package com.example.simpleserver

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import io.ktor.server.application.*
import io.ktor.server.cio.CIO
import io.ktor.server.engine.embeddedServer
import io.ktor.server.plugins.origin
import io.ktor.server.request.ApplicationRequest
import io.ktor.server.routing.*
import io.ktor.server.websocket.*
import io.ktor.websocket.*
import java.time.Duration
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

class ServerService : Service() {

    private val job = SupervisorJob()
    private val scope = CoroutineScope(Dispatchers.IO + job)

    private val server by lazy {
        embeddedServer(CIO, port = 8765, host = "0.0.0.0") {
            install(WebSockets)
//            {
//                pingPeriod = Duration.ofSeconds(15)
//                timeout = Duration.ofSeconds(15)
//                maxFrameSize = Long.MAX_VALUE
//                masking = false
//            }
            routing {
                webSocket("/echo") {
                    // UPDATED: Get client IP address using modern Ktor API
                    val clientIp = call.request.local.remoteAddress
                    try {
                        println("SUCCESS: Client connected! IP: $clientIp")
                        for (frame in incoming) {
                            if (frame is Frame.Text) {
                                val receivedText = frame.readText()
                                println("Server received: $receivedText")
                                outgoing.send(Frame.Text("You said: $receivedText"))
                            }
                        }
                    } catch (e: Exception) {
                        println("ERROR during WebSocket session: ${e.localizedMessage}")
                        e.printStackTrace()
                    } finally {
                        println("INFO: Client disconnected! IP: $clientIp")
                    }
                }
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        println("SERVER_SERVICE: onStartCommand received. Starting foreground service.")
        // Set the state to running
        _isServiceRunning.value = true
        startForeground(NOTIFICATION_ID, createNotification())
        scope.launch {
            println("SERVER_SERVICE: Ktor server starting...")
            server.start(wait = false)
        }
        println("SERVER_SERVICE: onStartCommand finished.")
        return START_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        println("SERVER_SERVICE: onDestroy called. Stopping server.")
        // Set the state to not running
        _isServiceRunning.value = false
        server.stop(1_000, 5_000)
        job.cancel()
    }

    private fun createNotification(): Notification {
        val channelId = "server_service_channel"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channelName = "WebSocket Server Service"
            val notificationManager = getSystemService(NotificationManager::class.java)
            val channel = NotificationChannel(
                channelId,
                channelName,
                NotificationManager.IMPORTANCE_DEFAULT
            )
            notificationManager.createNotificationChannel(channel)
        }
        return NotificationCompat.Builder(this, channelId)
            .setContentTitle("WebSocket Server")
            .setContentText("Server is running...")
            .setSmallIcon(R.mipmap.ic_launcher) // Make sure this drawable exists
            .build()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    companion object {
        private const val NOTIFICATION_ID = 1

        // CORRECTED: Define StateFlow here to be publicly accessible
        private val _isServiceRunning = MutableStateFlow(false)
        val isServiceRunning = _isServiceRunning.asStateFlow()
    }
}