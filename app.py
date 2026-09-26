"""
Live Audio Zero-Shot Classification Web Application using CLAP and Gradio.

This application uses the Hugging Face `transformers` implementation of CLAP
(`laion/clap-htsat-unfused`) to perform real-time, non-blocking zero-shot audio
classification on a continuous live microphone stream with a 48kHz rolling buffer.
"""

import os
import sys
import logging
import queue
import threading
import time
from typing import Dict, Tuple

import gradio as gr
import numpy as np
import sounddevice as sd
import torch
from transformers import ClapModel, ClapProcessor

# -----------------------------------------------------------------------------
# 0. Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("clap_app")

# -----------------------------------------------------------------------------
# 1. Model Initialization (Preserved Exactly)
# -----------------------------------------------------------------------------
CHECKPOINT = "laion/clap-htsat-unfused"
TARGET_SAMPLE_RATE = 48000  # Strictly required by the CLAP HTSAT feature extractor

# Select execution device (GPU if available, fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Target execution device: {device}")

logger.info(f"Loading CLAP processor from checkpoint '{CHECKPOINT}'...")
processor = ClapProcessor.from_pretrained(CHECKPOINT)

logger.info(f"Loading CLAP model from checkpoint '{CHECKPOINT}'...")
model = ClapModel.from_pretrained(CHECKPOINT)
model.to(device)
model.eval()
logger.info("CLAP model and processor loaded successfully.")

# -----------------------------------------------------------------------------
# 2. Continuous Streaming & Rolling Buffer Setup
# -----------------------------------------------------------------------------
BUFFER_DURATION = 2.0  # seconds of rolling audio history
BUFFER_SIZE = int(TARGET_SAMPLE_RATE * BUFFER_DURATION)  # 96,000 samples at 48kHz

# Thread-safe queue for non-blocking stream ingestion
audio_queue: queue.Queue = queue.Queue(maxsize=100)
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
buffer_lock = threading.Lock()

# Inference and streaming state management
is_streaming = threading.Event()
stream_handle = None
stream_lock = threading.Lock()

# Candidate labels state
current_candidate_labels = ["typing", "talking", "silence", "clapping"]
labels_lock = threading.Lock()

# Prediction results state
latest_predictions: Dict[str, float] = {}
predictions_lock = threading.Lock()
stream_status_msg = "🔴 Stream Inactive (Click 'Start Live Stream' to begin)"


def audio_callback(indata, frames, time_info, status):
    """
    Non-blocking audio callback invoked by sounddevice InputStream.
    Pushes incoming mono audio frames into the thread-safe queue.
    """
    if status:
        logger.debug(f"Audio stream status: {status}")
    try:
        # indata has shape (frames, channels); extract mono channel as float32
        audio_queue.put_nowait(indata[:, 0].copy())
    except queue.Full:
        pass


# -----------------------------------------------------------------------------
# 3. Asynchronous Background Inference Thread
# -----------------------------------------------------------------------------
def inference_worker():
    """
    Background worker thread that:
    1. Continuously drains audio_queue into the rolling buffer.
    2. Takes a snapshot of the last 1-2 seconds of 48kHz audio.
    3. Runs CLAP zero-shot classification against current candidate labels.
    4. Updates latest_predictions for real-time UI consumption.
    """
    global audio_buffer
    logger.info("Asynchronous CLAP inference worker started.")

    while True:
        if not is_streaming.is_set():
            time.sleep(0.1)
            continue

        # 1. Update rolling buffer from thread-safe queue
        with buffer_lock:
            while not audio_queue.empty():
                try:
                    chunk = audio_queue.get_nowait()
                except queue.Empty:
                    break
                chunk_len = len(chunk)
                if chunk_len >= BUFFER_SIZE:
                    audio_buffer[:] = chunk[-BUFFER_SIZE:]
                else:
                    audio_buffer[:-chunk_len] = audio_buffer[chunk_len:]
                    audio_buffer[-chunk_len:] = chunk
            snapshot = audio_buffer.copy()

        # 2. Retrieve current candidate prompts
        with labels_lock:
            labels = list(current_candidate_labels)

        if not labels:
            time.sleep(0.1)
            continue

        # 3. Core CLAP Inference (retained from original logic)
        try:
            inputs = processor(
                text=labels,
                audio=snapshot,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_audio = outputs.logits_per_audio
                probs = logits_per_audio.softmax(dim=-1)[0].cpu().numpy()

            results = {label: float(prob) for label, prob in zip(labels, probs)}

            with predictions_lock:
                latest_predictions.clear()
                latest_predictions.update(results)

        except Exception as e:
            logger.error(f"Inference error in worker: {e}")

        # Throttle loop to balance CPU load with low-latency updates (~10-15 FPS)
        time.sleep(0.08)


# Launch background inference thread as daemon
inference_thread = threading.Thread(target=inference_worker, daemon=True)
inference_thread.start()


# -----------------------------------------------------------------------------
# 4. Stream Control & UI Helper Functions
# -----------------------------------------------------------------------------
def start_stream() -> str:
    """Starts the non-blocking sounddevice microphone stream."""
    global stream_handle, stream_status_msg
    with stream_lock:
        if not is_streaming.is_set():
            try:
                # Clear queue & buffer for a clean start
                while not audio_queue.empty():
                    audio_queue.get_nowait()
                with buffer_lock:
                    audio_buffer.fill(0.0)

                stream_handle = sd.InputStream(
                    samplerate=TARGET_SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    blocksize=2048,
                    callback=audio_callback
                )
                stream_handle.start()
                is_streaming.set()
                stream_status_msg = "🟢 Live Stream Active (48kHz Rolling Buffer, 2.0s Window)"
                logger.info("Live microphone stream started at 48kHz.")
            except Exception as e:
                logger.error(f"Failed to start stream: {e}")
                stream_status_msg = f"❌ Error starting audio stream: {e}"
        return stream_status_msg


def stop_stream() -> str:
    """Stops the sounddevice microphone stream."""
    global stream_handle, stream_status_msg
    with stream_lock:
        if is_streaming.is_set():
            is_streaming.clear()
            if stream_handle is not None:
                try:
                    stream_handle.stop()
                    stream_handle.close()
                except Exception as e:
                    logger.warning(f"Error stopping stream: {e}")
                stream_handle = None
            stream_status_msg = "🔴 Stream Inactive (Click 'Start Live Stream' to begin)"
            logger.info("Live microphone stream stopped.")
        return stream_status_msg


def update_prompts(prompts_text: str) -> str:
    """Updates candidate zero-shot prompts dynamically during live streaming."""
    global current_candidate_labels
    labels = [label.strip() for label in prompts_text.split(",") if label.strip()]
    if not labels:
        return "⚠️ Please provide at least one valid prompt separated by commas."
    with labels_lock:
        current_candidate_labels = labels
    return f"✅ Updated prompts ({len(labels)} classes): {', '.join(labels)}"


def get_live_state() -> Tuple[Dict[str, float], str]:
    """Polled periodically by gr.Timer to update UI in real-time."""
    with predictions_lock:
        preds = dict(latest_predictions)
    return preds, stream_status_msg


# -----------------------------------------------------------------------------
# 5. Gradio Interface Construction
# -----------------------------------------------------------------------------
custom_css = """
.gradio-container {
    max-width: 950px !important;
    margin: 0 auto !important;
}
.header-box {
    text-align: center;
    margin-bottom: 1.5rem;
}
.status-bar {
    font-weight: 600;
    font-size: 1.05rem;
}
"""

with gr.Blocks(title="CLAP Real-Time Streaming Audio Classifier") as demo:
    with gr.Column(elem_classes=["header-box"]):
        gr.Markdown(
            """
            # 🎙️ Live Streaming Audio Zero-Shot Classification
            ### Real-Time Contrastive Language-Audio Pretraining (`laion/clap-htsat-unfused`)
            Continuously ingests microphone audio into a 48kHz rolling buffer with non-blocking async inference.
            """
        )

    status_box = gr.Markdown(
        value=stream_status_msg,
        elem_classes=["status-bar"]
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                start_btn = gr.Button("▶️ Start Live Stream", variant="primary")
                stop_btn = gr.Button("⏹️ Stop Live Stream", variant="secondary")

            prompts_input = gr.Textbox(
                label="Candidate Text Prompts (comma-separated)",
                placeholder="e.g. typing, talking, silence, clapping, laughter, music",
                value=", ".join(current_candidate_labels),
                lines=2
            )
            update_btn = gr.Button("Update Prompts", variant="neutral")
            prompt_status = gr.Markdown(value="")

        with gr.Column(scale=1):
            output_label = gr.Label(
                num_top_classes=6,
                label="Live Class Probabilities"
            )

            with gr.Accordion("Technical Details & Architecture", open=False):
                gr.Markdown(
                    f"""
                    - **Stream Ingestion:** Non-blocking `sounddevice.InputStream` callback -> thread-safe `queue.Queue`.
                    - **Rolling Buffer:** Continuous circular buffer maintaining the last **{BUFFER_DURATION} seconds** ({BUFFER_SIZE:,} samples at **{TARGET_SAMPLE_RATE} Hz**).
                    - **Asynchronous Inference:** Dedicated background daemon thread running `ClapModel` predictions independently from the audio capture thread.
                    - **Model:** `laion/clap-htsat-unfused` on `{device.type.upper()}`.
                    """
                )

    # Event handlers
    start_btn.click(
        fn=start_stream,
        inputs=[],
        outputs=[status_box]
    )

    stop_btn.click(
        fn=stop_stream,
        inputs=[],
        outputs=[status_box]
    )

    update_btn.click(
        fn=update_prompts,
        inputs=[prompts_input],
        outputs=[prompt_status]
    )

    prompts_input.submit(
        fn=update_prompts,
        inputs=[prompts_input],
        outputs=[prompt_status]
    )

    # Real-time polling timer for UI updates (every 250ms)
    live_timer = gr.Timer(value=0.25, active=True)
    live_timer.tick(
        fn=get_live_state,
        inputs=[],
        outputs=[output_label, status_box]
    )


# -----------------------------------------------------------------------------
# 6. Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[*] Starting Gradio CLAP streaming server on port {port}...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        theme=gr.themes.Soft(),
        css=custom_css,
        share=False
    )
