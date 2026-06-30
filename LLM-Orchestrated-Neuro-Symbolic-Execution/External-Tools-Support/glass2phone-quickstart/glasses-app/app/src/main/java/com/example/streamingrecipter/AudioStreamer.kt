package com.example.streamingrecipter

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import okhttp3.*
import okio.ByteString.Companion.toByteString
import java.util.concurrent.TimeUnit

// Interface to communicate events back to the ViewModel/UI
interface AudioStreamListener {
    fun onConnected()
    fun onMessage(text: String)
    fun onDisconnected()
    fun onError(error: String)
}

class AudioStreamer(private val listener: AudioStreamListener) {

    private companion object {
        const val TAG = "AudioStreamer"
        const val RECORDER_SAMPLERATE = 16000
        val RECORDER_CHANNELS: Int = AudioFormat.CHANNEL_IN_MONO
        val RECORDER_AUDIO_ENCODING: Int = AudioFormat.ENCODING_PCM_16BIT
    }

    private var audioRecord: AudioRecord? = null
    private var webSocket: WebSocket? = null
    private var client: OkHttpClient = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    private val coroutineScope = CoroutineScope(Dispatchers.IO)
    private var recordingJob: Job? = null

    // --- CHANGE 1: Add a volatile flag for our own state management ---
    @Volatile
    private var isStreamingActive = false

    private val bufferSize = AudioRecord.getMinBufferSize(
        RECORDER_SAMPLERATE,
        RECORDER_CHANNELS,
        RECORDER_AUDIO_ENCODING
    )

    @SuppressLint("MissingPermission")
    fun startStreaming(url: String) {
        if (isStreaming()) {
            Log.w(TAG, "Stream is already running")
            return
        }

        // --- CHANGE 2: Set our flag to true before we start ---
        isStreamingActive = true

        val request = Request.Builder().url(url).build()

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            RECORDER_SAMPLERATE,
            RECORDER_CHANNELS,
            RECORDER_AUDIO_ENCODING,
            bufferSize
        )

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                super.onOpen(webSocket, response)
                Log.d(TAG, "WebSocket Opened")
                listener.onConnected()

                // It's crucial that startRecording is called before the loop starts
                audioRecord?.startRecording()
                startRecordingLoop(webSocket)
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                super.onMessage(webSocket, text)
                Log.d(TAG, "Message received: $text")
                listener.onMessage(text)
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                super.onClosing(webSocket, code, reason)
                Log.d(TAG, "WebSocket Closing: $code / $reason")
                // Ensure streaming stops if the server closes the connection
                stopStreaming()
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                super.onFailure(webSocket, t, response)
                Log.e(TAG, "WebSocket Failure: ${t.message}")
                listener.onError("Connection failed: ${t.message}")
                stopStreaming()
            }
        })
    }

    private fun startRecordingLoop(socket: WebSocket) {
        recordingJob = coroutineScope.launch {
            val buffer = ByteArray(bufferSize)
            try {
                // --- CHANGE 3: Use our reliable flag for the loop condition ---
                while (isStreamingActive) {
                    val bytesRead = audioRecord?.read(buffer, 0, buffer.size) ?: -1
                    if (bytesRead > 0) {
                        socket.send(buffer.toByteString(0, bytesRead))
                    }
                }
            } catch (e: Exception) {
                if (isStreamingActive) { // Only report error if we were supposed to be streaming
                    Log.e(TAG, "Recording loop error", e)
                    listener.onError("Streaming error: ${e.message}")
                    stopStreaming()
                }
            }
        }
    }

    fun stopStreaming() {
        if (!isStreamingActive) return

        Log.d(TAG, "Stopping stream...")
        // --- CHANGE 4: Set the flag to false first to signal the loop to stop ---
        isStreamingActive = false

        recordingJob?.cancel()
        recordingJob = null

        // It's good practice to check the state before stopping
        if (audioRecord?.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
            audioRecord?.stop()
        }
        audioRecord?.release()
        audioRecord = null

        webSocket?.close(1000, "Client stopped streaming")
        webSocket = null
        listener.onDisconnected()
    }

    // --- CHANGE 5: The public method now returns our flag's state ---
    fun isStreaming(): Boolean {
        return isStreamingActive
    }
}