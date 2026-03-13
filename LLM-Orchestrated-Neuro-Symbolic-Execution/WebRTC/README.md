# WebRTC Integration

Adds WebRTC native media streaming alongside the existing WebSocket control channel. Built with [aiortc](https://github.com/aiortc/aiortc) (pure Python, no external server required).

---

## Features

- **Video echo** — relays the browser's camera track back for verification
- **Real-time ASR** — WebRTC VAD detects speech boundaries, [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2, int8) transcribes, results sent back via data channel

## Architecture

```
Browser (camera + mic)
       │
       │  1. HTTP POST /offer  (SDP exchange)
       │  2. WebRTC media tracks + data channel
       ▼
┌──────────────────────────┐
│  aiortc server           │
│                          │
│  audio track ──► resample (av) ──► WebRTC VAD ──► faster-whisper
│                                                        │
│  video track ──► echo back to browser          data channel ◄─┘
└──────────────────────────┘
```

## Files

```
WebRTC/
├── server.py    # aiortc server: signaling, video echo, VAD + ASR
├── index.html   # browser client: camera/mic capture, transcript display
└── README.md
```

## Quick Start

```bash
pip install aiortc aiohttp faster-whisper webrtcvad av
cd LLM-Orchestrated-Neuro-Symbolic-Execution/WebRTC
python server.py
```

Open http://localhost:8080, click **Connect**, grant camera/microphone access. Speak and the transcription appears in real time.

## How It Works

1. **Resampling** — `av.AudioResampler` converts aiortc's 48kHz stereo s16 frames to 16kHz mono
2. **VAD** — `webrtcvad` processes 30ms frames; speech segments are buffered, ~600ms of silence triggers transcription
3. **ASR** — `faster-whisper` (tiny model, int8 on CPU) transcribes the buffered speech; results accumulate and are sent as full text via WebRTC data channel
4. **Frontend** — the browser replaces the transcript div on each update, so it reads as one continuous stream

## Configuration

In `server.py`:

| Variable | Default | Description |
|:---------|:--------|:------------|
| `model_size` | `"tiny"` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `language` | `"en"` | Target language for ASR (in `whisper_model.transcribe` call) |
| `Vad(2)` | aggressiveness 2 | VAD sensitivity 0–3 (higher = more aggressive filtering) |
| `MAX_SILENT_FRAMES` | 20 | ~600ms silence before triggering transcription |
| `MAX_BUFFER_BYTES` | 320000 | 10s hard cap on speech buffer |

## Integration with Full Pipeline

The `consume_video` and `consume_audio` functions can be extended to feed into the Egocentric Co-Pilot pipeline:

```python
# In consume_video:
frame_bgr = frame.to_ndarray(format="bgr24")
# → feed into Qwen2-VL / YOLO / image_queue

# In consume_audio (after ASR):
# text → feed into LLM orchestrator for intent/action
```
