package com.example.simpleserver

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.example.simpleserver.ui.theme.SimpleServerTheme
import java.net.Inet4Address
import java.net.NetworkInterface

class MainActivity : ComponentActivity() {

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            // Permission has been granted, now we can safely start the service.
            startServerService()
        } else {
            // Handle the case where the user denies the permission.
            // You could show a Snackbar or a dialog explaining why the permission is needed.
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            SimpleServerTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    // Pass the start server action down to the composable
                    ServerControlScreen(onStartServer = {
                        // This is the action that will be triggered by the button click
                        checkAndStartServer()
                    })
                }
            }
        }
    }

    private fun checkAndStartServer() {
        // This is only required for API level 33+ (Android 13)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            when {
                ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.POST_NOTIFICATIONS
                ) == PackageManager.PERMISSION_GRANTED -> {
                    // You have the permission, you can start the service.
                    startServerService()
                }
                shouldShowRequestPermissionRationale(Manifest.permission.POST_NOTIFICATIONS) -> {
                    // Optionally, show a rationale dialog to the user explaining why
                    // you need the permission before asking again.
                    requestPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                }
                else -> {
                    // Directly ask for the permission.
                    requestPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                }
            }
        } else {
            // For older versions (below Android 13), no runtime permission is needed.
            startServerService()
        }
    }

    private fun startServerService() {
        val intent = Intent(this, ServerService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }
    }
}

@Composable
fun ServerControlScreen(onStartServer: () -> Unit) { // Accept the start server action
    val context = LocalContext.current
    val isServerRunning by ServerService.isServiceRunning.collectAsState()

    fun getLocalIpAddress(): String {
        return try {
            NetworkInterface.getNetworkInterfaces().toList().mapNotNull { networkInterface ->
                networkInterface.inetAddresses.toList().find {
                    !it.isLoopbackAddress && it is Inet4Address
                }?.hostAddress
            }.firstOrNull() ?: "Not connected"
        } catch (e: Exception) {
            "Error getting IP"
        }
    }

    val ipAddress by remember { mutableStateOf(getLocalIpAddress()) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(text = "In-App WebSocket Server", style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(32.dp))

        Text(
            text = "Server Status: ${if (isServerRunning) "Running" else "Stopped"}",
            fontSize = 18.sp,
            color = if (isServerRunning) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "Connect clients to:",
            style = MaterialTheme.typography.bodyLarge
        )
        Text(
            text = "ws://$ipAddress:8765/echo",
            style = MaterialTheme.typography.bodyLarge.copy(fontWeight = FontWeight.Bold),
            textAlign = TextAlign.Center
        )
        Spacer(modifier = Modifier.height(24.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            Button(
                onClick = {
                    // Call the action that was passed down from the Activity
                    onStartServer()
                },
                enabled = !isServerRunning
            ) {
                Text("Start Server")
            }

            Button(
                onClick = {
                    val intent = Intent(context, ServerService::class.java)
                    context.stopService(intent)
                },
                enabled = isServerRunning
            ) {
                Text("Stop Server")
            }
        }
    }
}