"""
WebRTC demo server with:
  A. Video echo  — relays the browser's video track back so it appears on screen.
  B. Whisper ASR — accumulates audio, detects speech, transcribes, and sends
                   the text back via a WebRTC data channel.
"""

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay

# --- Whisper (optional) ---
try:
    import whisper

    logger_init = logging.getLogger(__name__)
    logger_init.info("Loading Whisper model (base)...")
    whisper_model = whisper.load_model("base")
    ASR_AVAILABLE = True
except ImportError:
    whisper_model = None
    ASR_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

if not ASR_AVAILABLE:
    logger.warning("openai-whisper not installed — ASR disabled. Install with: pip install -U openai-whisper")
else:
    logger.info("Whisper model loaded.")

ROOT = Path(__file__).parent
pcs = set()
relay = MediaRelay()

WHISPER_SR = 16000


# ---------------------------------------------------------------------------
# A. Video: log frames (echo is handled by pc.addTrack in the offer handler)
# ---------------------------------------------------------------------------
async def consume_video(track):
    count = 0
    while True:
        try:
            frame = await track.recv()
        except Exception:
            break
        count += 1
        if count % 30 == 0:
            img = frame.to_ndarray(format="bgr24")
            logger.info(f"Video frame #{count}: {img.shape}")


# ---------------------------------------------------------------------------
# B. Audio: VAD → accumulate → Whisper → data channel
# ---------------------------------------------------------------------------
def transcribe(audio_np):
    """Run Whisper on a float32 numpy array (16 kHz)."""
    result = whisper_model.transcribe(audio_np, language="en")
    return result["text"].strip()


async def consume_audio(track, dc_ref):
    """Simple energy-based VAD → Whisper ASR → send text via data channel."""

    # If Whisper is not available, fall back to logging only
    if not ASR_AVAILABLE:
        total = 0
        while True:
            try:
                frame = await track.recv()
            except Exception:
                break
            total += frame.samples
            if total % 16000 < frame.samples:
                logger.info(f"Audio: ~{total // 16000}s ({total} samples)")
        return

    buffer = []
    speech_active = False
    silence_count = 0
    speech_count = 0

    ENERGY_THRESHOLD = 0.03   # higher to ignore background noise
    SILENCE_LIMIT = 40        # consecutive silent frames to end a segment (~0.8s)
    MIN_SPEECH_FRAMES = 10    # ignore very short bursts
    MAX_BUFFER_SECONDS = 15   # hard cap to prevent runaway accumulation

    # Calibrate noise floor from first ~50 frames
    noise_samples = []
    CALIBRATION_FRAMES = 50

    while True:
        try:
            frame = await track.recv()
        except Exception:
            break

        samples = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        energy = float(np.sqrt(np.mean(samples ** 2)))
        src_sr = frame.sample_rate or 48000

        # Auto-calibrate threshold from initial noise floor
        if len(noise_samples) < CALIBRATION_FRAMES:
            noise_samples.append(energy)
            if len(noise_samples) == CALIBRATION_FRAMES:
                noise_floor = sum(noise_samples) / len(noise_samples)
                ENERGY_THRESHOLD = max(0.03, noise_floor * 3.0)
                logger.info(f"Audio calibrated: noise floor={noise_floor:.4f}, threshold={ENERGY_THRESHOLD:.4f}")
            continue

        # Hard cap: flush buffer if it gets too long
        total_buffered = sum(len(b) for b in buffer) / src_sr if buffer else 0
        if total_buffered > MAX_BUFFER_SECONDS:
            logger.warning(f"Buffer exceeded {MAX_BUFFER_SECONDS}s, flushing")
            buffer.clear()
            speech_active = False
            silence_count = 0
            speech_count = 0
            continue

        if energy > ENERGY_THRESHOLD:
            speech_active = True
            silence_count = 0
            speech_count += 1
            buffer.append(samples)
        elif speech_active:
            buffer.append(samples)
            silence_count += 1

            if silence_count >= SILENCE_LIMIT and speech_count >= MIN_SPEECH_FRAMES:
                # Resample to 16 kHz
                audio_full = np.concatenate(buffer)
                n_out = int(len(audio_full) * WHISPER_SR / src_sr)
                indices = np.round(np.linspace(0, len(audio_full) - 1, n_out)).astype(int)
                audio_16k = audio_full[indices]

                duration = len(audio_16k) / WHISPER_SR
                logger.info(f"Processing {duration:.1f}s of speech...")

                text = await asyncio.to_thread(transcribe, audio_16k)

                if text:
                    logger.info(f"ASR: {text}")
                    dc = dc_ref[0]
                    if dc and dc.readyState == "open":
                        dc.send(json.dumps({"type": "asr", "text": text}))

                buffer.clear()
                speech_active = False
                silence_count = 0
                speech_count = 0


# ---------------------------------------------------------------------------
# Signaling
# ---------------------------------------------------------------------------
async def offer(request):
    params = await request.json()
    sdp_offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    # Mutable ref so consume_audio can access the data channel once it opens
    dc_ref = [None]

    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info(f"Data channel opened: {channel.label}")
        dc_ref[0] = channel

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        logger.info(f"Received {track.kind} track")

        if track.kind == "video":
            # A: Echo video back to the browser
            pc.addTrack(relay.subscribe(track))
            # Also log frames
            asyncio.ensure_future(consume_video(relay.subscribe(track)))

        elif track.kind == "audio":
            # B: Run ASR and send text via data channel
            asyncio.ensure_future(consume_audio(relay.subscribe(track), dc_ref))

        @track.on("ended")
        async def on_ended():
            logger.info(f"{track.kind} track ended")

    await pc.setRemoteDescription(sdp_offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    })


async def index(request):
    return web.FileResponse(ROOT / "index.html")


async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)

    logger.info("WebRTC demo server starting on http://localhost:8080")
    web.run_app(app, host="localhost", port=8080)
