package com.example.phoneserver

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.phoneserver.ui.theme.PhoneServerTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            // Start the Ktor server service
            PhoneServerTheme {
                // This is the correct way to start the service in Compose
                ServerStarter()

                // Your UI can go here, for example:

//                // Greeting("Android")
//                val serviceIntent = Intent(this, ServerService::class.java)
//                startService(serviceIntent)
                TranscriptView(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(16.dp)
                )
            }
        }
    }
}



@Composable
fun ServerStarter() {
    // We get the context here, as it's the standard Compose way
    val context = LocalContext.current

    // LaunchedEffect will run this block only once because its key (Unit) never changes.
    // This is the modern, correct way to handle one-off side effects.
    LaunchedEffect(Unit) {
        val serviceIntent = Intent(context, ServerService::class.java)
        context.startService(serviceIntent)
    }
}

@Composable
fun TranscriptView(modifier: Modifier = Modifier) {
    // Collect StateFlows as Compose state
    val partial by TranscriptionState.partial.collectAsState()
    val final by TranscriptionState.final.collectAsState()

    Card(modifier = modifier) {
        Column(Modifier.padding(16.dp)) {
            Text("Live Transcript", style = MaterialTheme.typography.titleLarge)
            if (partial.isNotBlank()) {
                Text(
                    text = "Partial: $partial",
                    style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier.padding(top = 8.dp)
                )
            }
            if (final.isNotBlank()) {
                Text(
                    text = "Final: $final",
                    style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier.padding(top = 12.dp)
                )
            }
            if (partial.isBlank() && final.isBlank()) {
                Text(
                    text = "Waiting for audio...",
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(top = 8.dp)
                )
            }
        }
    }
}