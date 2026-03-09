# WebRTC Integration

Adds WebRTC native media streaming alongside the existing WebSocket control channel. Built with [aiortc](https://github.com/aiortc/aiortc) (pure Python, no external server required).

---

## Motivation

The current system uses WebSocket for everything: video frames (manual JPEG encode + 4-byte header), audio clips (manual WAV encode + 4-byte header), and control messages. This module offloads the media transport to WebRTC native tracks, while keeping WebSocket for lightweight control signaling.

## WebRTC vs WebSocket: Division of Responsibilities

| Transport           | What it carries                                                                | Why                                        |
| :------------------ | :----------------------------------------------------------------------------- | :----------------------------------------- |
| **WebRTC**    | Video track, audio track                                                       | Native codec, low latency, NAT traversal   |
| **WebSocket** | Control messages (`imageAnalysis_interval`, `imageAnalysis_quality`, etc.) | Lightweight, reliable, already implemented |

## Architecture

```
Browser / Android
       │
       │  1. HTTP: fetch SDP offer
       │  2. WebRTC: media tracks
       ▼
┌─────────────────────┐
│  aiortc server      │
│  (signaling + media)│
│                     │
│  on_track(audio) ───┼──► Whisper ASR ──► Qwen-VL ──► F5-TTS
│  on_track(video) ───┼──► frame buffer                  │
│                     │                                  │
│  add_track(tts) ◄───┼──────────────────────────────────┘
└─────────────────────┘
```

Signaling uses a simple HTTP endpoint (`aiohttp`). No separate signaling server or LiveKit infrastructure needed.

## Files

```
WebRTC/
├── server.py    # aiortc WebRTC server (signaling + media consumption)
├── index.html   # Browser client (camera + mic → WebRTC tracks)
└── README.md
```

## Quick Start

```bash
pip install aiortc aiohttp
cd LLM-Orchestrated-Neuro-Symbolic-Execution/WebRTC
python server.py
```

Open http://localhost:8080 in your browser, click **Connect**, and grant camera/microphone access. The server logs received video frames and audio samples:

```
2026-03-09 12:00:01,234 - INFO - Received video track
2026-03-09 12:00:01,235 - INFO - Received audio track
2026-03-09 12:00:02,456 - INFO - Video frame #30: (720, 1280, 3)
2026-03-09 12:00:03,012 - INFO - Audio: ~1s received (16000 samples)
```

## Integration with Existing Pipeline

The `consume_video` and `consume_audio` functions in `server.py` are placeholders. To integrate with the full Egocentric Co-Pilot pipeline, replace them with calls to the existing inference logic in `main.py`:

```python
# In consume_video:
frame_bgr = frame.to_ndarray(format="bgr24")
# → feed into Qwen2-VL / YOLO / image_queue

# In consume_audio:
audio_array = frame.to_ndarray()
# → accumulate and feed into Whisper ASR
```
