package com.example.phoneserver

import android.app.Service
import android.content.Intent
import android.os.IBinder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class ServerService : Service() {
//    private val serverJob = CoroutineScope(Dispatchers.IO).launch {
//        startServer()
//    }
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        CoroutineScope(Dispatchers.IO).launch {
            // Pass the context directly when starting the server
            startServer(applicationContext)
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? {
        return null
    }

//    override fun onDestroy() {
//        super.onDestroy()
//        serverJob.cancel()
//    }
}