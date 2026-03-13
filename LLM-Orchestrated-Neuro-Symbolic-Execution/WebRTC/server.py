"""
WebRTC demo server with:
  A. Video echo  — relays the browser's video track back so it appears on screen.
  B. Whisper ASR — VAD-based speech detection, transcribes with faster-whisper,
                   sends text back via a WebRTC data channel.
"""

import asyncio
import json
import logging
from pathlib import Path

import av
import numpy as np
import webrtcvad
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from faster_whisper import WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- faster-whisper (int8 on CPU for speed) ---
logger.info("Loading faster-whisper model (tiny, int8)...")
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
logger.info("Model loaded.")

# --- WebRTC VAD (aggressiveness 2 out of 0-3) ---
vad = webrtcvad.Vad(2)

ROOT = Path(__file__).parent
pcs = set()
relay = MediaRelay()


# ---------------------------------------------------------------------------
# A. Video: echo + log
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
# B. Audio: av.AudioResampler → WebRTC VAD → faster-whisper
# ---------------------------------------------------------------------------
async def consume_audio(track, dc_ref):
    """Resample to 16kHz mono, use VAD to detect speech, transcribe on silence."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

    audio_buffer = bytearray()  # speech audio for transcription
    vad_buffer = bytearray()    # accumulates resampled data for VAD framing
    silent_count = 0
    speaking = False
    full_text = ""

    MAX_SILENT_FRAMES = 20   # ~600ms silence triggers transcription
    MAX_BUFFER_BYTES = 16000 * 2 * 10  # 10s hard cap (16kHz * 2 bytes * 10s)
    VAD_FRAME_BYTES = 960    # 30ms at 16kHz, 16-bit mono = 480 samples * 2

    while True:
        try:
            frame = await track.recv()
        except Exception:
            break

        # Resample to 16kHz mono s16
        resampled_frames = resampler.resample(frame)
        for rf in resampled_frames:
            vad_buffer.extend(rf.to_ndarray().tobytes())

        # Process accumulated data in 30ms VAD frames
        while len(vad_buffer) >= VAD_FRAME_BYTES:
            chunk = bytes(vad_buffer[:VAD_FRAME_BYTES])
            del vad_buffer[:VAD_FRAME_BYTES]

            try:
                is_speech = vad.is_speech(chunk, 16000)
            except Exception:
                is_speech = False

            if is_speech:
                audio_buffer.extend(chunk)
                silent_count = 0
                speaking = True
            elif speaking:
                audio_buffer.extend(chunk)
                silent_count += 1

            # Trigger transcription: speech ended or buffer too long
            if speaking and (silent_count > MAX_SILENT_FRAMES or len(audio_buffer) > MAX_BUFFER_BYTES):
                buf_dur = len(audio_buffer) / (16000 * 2)
                logger.info(f"Transcribing {buf_dur:.1f}s of speech...")

                audio_np = np.frombuffer(bytes(audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0

                segments, _ = await asyncio.to_thread(
                    whisper_model.transcribe, audio_np, beam_size=5, language="en"
                )
                text = "".join(s.text for s in segments).strip()

                if text:
                    full_text = (full_text + text) if full_text else text
                    logger.info(f"ASR: {text}")
                    dc = dc_ref[0]
                    if dc and dc.readyState == "open":
                        dc.send(json.dumps({"type": "asr", "text": full_text}))

                audio_buffer.clear()
                silent_count = 0
                speaking = False


# ---------------------------------------------------------------------------
# Signaling
# ---------------------------------------------------------------------------
async def offer(request):
    params = await request.json()
    sdp_offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

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
            pc.addTrack(relay.subscribe(track))
            asyncio.ensure_future(consume_video(relay.subscribe(track)))
        elif track.kind == "audio":
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
