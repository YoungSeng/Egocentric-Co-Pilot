# On-Device Perception and Interaction

The Android front-end application for **Egocentric Co-Pilot**, built with [CameraX](https://developer.android.com/training/camerax) and [Jetpack Compose](https://developer.android.com/jetpack/compose). It handles real-time visual perception, voice activity detection, and WebSocket communication with the back-end server.

---

## Overview

This module runs on Android smart glasses (or any Android device with a camera and microphone). It continuously captures video frames and audio, then streams them to the [`LLM-Orchestrated-Neuro-Symbolic-Execution`](../LLM-Orchestrated-Neuro-Symbolic-Execution) back-end via WebSocket for egocentric reasoning and response generation.

```
┌────────────────────────────────────────────────┐
│                Android Device                  │
│                                                │
│  CameraX ──► Frame Capture ──► JPEG Encode ─┐  │
│                                             │  │
│  AudioRecord ──► VAD ──► WAV Encode ────────┤  │
│                                             │  │
│             WebSocket Client ◄──────────────┘  │
│                 │           ▲                  │
└─────────────────┼───────────┼──────────────────┘
                  │           │
           image/audio     TTS audio
             frames        response
                  │           │
                  ▼           │
        ┌─────────────────────┴──┐
        │    Back-end Server     │
        │   (WebSocket :5000)   │
        └────────────────────────┘
```

---

## Operating Modes

The app supports multiple operating modes, configured via the `mode` variable in `MainActivity.kt`:

|    Mode    | Description                                         |      Transport      |
| :---------: | :-------------------------------------------------- | :-----------------: |
|      0      | VAD-triggered audio recording + photo capture       |        HTTP        |
|      1      | Real-time video frame streaming                     |      WebSocket      |
|      3      | Continuous photo capture (no audio)                 |        HTTP        |
| **4** | **Video stream + continuous audio (default)** | **WebSocket** |

---

## Project Structure

```
On-Device-Perception-and-Interaction/
├── MainActivity.kt      # Core activity: camera, audio, WebSocket, and playback logic
├── CameraPreview.kt     # Jetpack Compose CameraX preview component
└── README.md
```

These files should be placed in the following path within your Android Studio project:

```
app/src/main/java/com/plcoding/cameraxguide/
├── MainActivity.kt
└── CameraPreview.kt
```

---

## Dependencies

| Library                                                                                                 | Purpose                                     |
| :------------------------------------------------------------------------------------------------------ | :------------------------------------------ |
| [CameraX](https://developer.android.com/training/camerax)                                                  | Camera capture and image analysis           |
| [Camera2 Interop](https://developer.android.com/reference/androidx/camera/camera2/interop/package-summary) | Fine-grained camera control (e.g., AE lock) |
| [Jetpack Compose](https://developer.android.com/jetpack/compose)                                           | Declarative UI                              |
| [OkHttp](https://square.github.io/okhttp/)                                                                 | WebSocket and HTTP client                   |
| Android AudioRecord                                                                                     | Low-level audio capture with VAD            |

---

## Key Configuration

All configurable parameters are defined in the `companion object` and class-level properties of `MainActivity.kt`:

### Server

| Parameter       | Default       | Description                                   |
| :-------------- | :------------ | :-------------------------------------------- |
| `SERVER_IP`   | `127.0.0.1` | Back-end server IP (use with `adb reverse`) |
| `SERVER_PORT` | `5000`      | Back-end server port                          |

### Camera

| Parameter                  | Default      | Description                       |
| :------------------------- | :----------- | :-------------------------------- |
| `targetResolution`       | 1920 × 1080 | Capture resolution                |
| `imageAnalysis_interval` | 200 ms       | Min interval between frame sends  |
| `imageAnalysis_quality`  | 25           | JPEG compression quality (0–100) |
| `defaultZoomRatio`       | 1.75×       | Camera zoom level                 |

### Audio

| Parameter                     | Default  | Description                              |
| :---------------------------- | :------- | :--------------------------------------- |
| `sampleRate`                | 16000 Hz | Audio sample rate                        |
| `amplitudeThreshold_input`  | 1200     | Voice activity detection threshold       |
| `amplitudeThreshold_output` | 6000     | Threshold to interrupt TTS playback      |
| `silenceDuration`           | 450 ms   | Silence duration before ending recording |
| `minRecordingDuration`      | 755 ms   | Minimum recording length to send         |

---

## WebSocket Protocol

Communication uses a simple binary protocol with a **4-byte header** to distinguish data types:

| Header           | Type        | Payload    |
| :--------------- | :---------- | :--------- |
| `[1, 1, 1, 1]` | Image frame | JPEG bytes |
| `[0, 0, 0, 0]` | Audio clip  | WAV bytes  |

The server responds with raw WAV audio bytes (TTS output) via WebSocket, which are played back immediately on the device. In HTTP mode (mode 0), the server returns a JSON response containing an `audio_url` for playback instead.

---

## Getting Started

### Prerequisites

- **Android Studio** 2024.2.1 Patch 3 or later
- Android device or emulator with camera and microphone permissions
- Back-end server running (see [`LLM-Orchestrated-Neuro-Symbolic-Execution`](../LLM-Orchestrated-Neuro-Symbolic-Execution))

### Setup

1. Open your Android Studio project and place `MainActivity.kt` and `CameraPreview.kt` into:

   ```
   app/src/main/java/com/plcoding/cameraxguide/
   ```
2. Connect your Android device via USB and set up port forwarding:

   ```bash
   adb reverse tcp:5000 tcp:5000
   ```
3. Click **Run** in Android Studio to build and deploy the app.
4. *(Optional)* Mirror the device screen for debugging:

   ```bash
   scrcpy
   ```
