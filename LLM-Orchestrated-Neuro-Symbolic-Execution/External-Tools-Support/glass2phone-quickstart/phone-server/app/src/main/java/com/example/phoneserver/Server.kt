package com.example.phoneserver

import android.content.Context
import io.ktor.server.application.*
import io.ktor.server.cio.*
import io.ktor.server.engine.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import io.ktor.server.websocket.*
import io.ktor.websocket.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.suspendCancellableCoroutine
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import org.vosk.android.StorageService
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.time.Duration.Companion.seconds


object TranscriptionState {
    val partial = MutableStateFlow("")
    val final = MutableStateFlow("")
}

// The startServer function accepts the context from the Service.
fun startServer(context: Context) {
    // We define the module as a lambda directly inside the embeddedServer call.
    // This allows the code inside to "capture" the 'context' variable.
    embeddedServer(CIO, port = 8765, host = "0.0.0.0") {
        // --- This is our Application Module ---

        install(WebSockets) {
            pingPeriod = 15.seconds
            timeout = 15.seconds
            maxFrameSize = Long.MAX_VALUE
            masking = false
        }

        routing {
            get("/") {
                call.respondText("Hello World!")
            }

            webSocket("/ws") {
                var model: Model? = null
                try {
                    // Use the 'context' captured from the outer scope.
//                    model = loadVoskModel(context, "vosk-model-en-us-0.22-lgraph")

                    model = loadVoskModel(context, "vosk-model-small-en-us-0.15")
                    Recognizer(model, 16000.0f).use { recognizer ->
                        for (frame in incoming) {
                            if (frame is Frame.Binary) {
                                val audioBytes = frame.readBytes()

                                // =================================================================
                                // THE FIX: Capitalize the 'F' in acceptWaveForm
                                // =================================================================
                                if (recognizer.acceptWaveForm(audioBytes, audioBytes.size)) {
                                    val resultJson = recognizer.result
                                    val transcript = JSONObject(resultJson).getString("text")
                                    if (transcript.isNotBlank()) {
                                        println("Final Transcript: $transcript")
//                                        TranscriptionState.transcribedText.value = "Received audio data: ${transcript.size} bytes"
                                        TranscriptionState.final.value = transcript
                                        // clear partial once we have a final for that segment
                                        TranscriptionState.partial.value = ""
                                    }
                                } else {
                                    val partialResultJson = recognizer.partialResult
                                    val partialTranscript = JSONObject(partialResultJson).getString("partial")
                                    if (partialTranscript.isNotBlank()) {
                                        println("Partial: $partialTranscript")
//                                        TranscriptionState.final.value = transcript
                                        // clear partial once we have a final for that segment
                                        TranscriptionState.partial.value = partialTranscript
                                    }
                                }

                            }
                        }

                        val finalResultJson = recognizer.finalResult
                        val finalTranscript = JSONObject(finalResultJson).getString("text")
                        if (finalTranscript.isNotBlank()) {
                            println("Final Transcript (on disconnect): $finalTranscript")
                        }
                    }
                } catch (e: Exception) {
                    println("An error occurred in the WebSocket session: ${e.message}")
                    e.printStackTrace()
                } finally {
                    println("Closing model...")
                    model?.close()
                }
            }
        }
    }.start(wait = true)
}

/**
 * A helper suspend function to bridge Vosk's callback-based model loading
 * with Kotlin Coroutines.
 */
private suspend fun loadVoskModel(context: Context, modelNameInAssets: String): Model = suspendCancellableCoroutine { continuation ->
    StorageService.unpack(context, modelNameInAssets, "model",
        { model ->
            continuation.resume(model)
        },
        { exception ->
            continuation.resumeWithException(exception)
        }
    )
}