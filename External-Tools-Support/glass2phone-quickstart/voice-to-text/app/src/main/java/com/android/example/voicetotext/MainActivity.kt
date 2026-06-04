package com.android.example.voicetotext

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.android.example.voicetotext.ui.theme.VoiceTotextTheme
import java.util.Locale

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            VoiceTotextTheme {
                Scaffold(
                    modifier = Modifier.fillMaxSize(),
                    topBar = { AppBar() }) { innerPadding ->
                    VoiceRecognitionScreen(
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppBar() {
    TopAppBar(
        title = {
            Text(text = "Continuous Voice to Text", color = Color.White)
        },
        colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.primary)
    )
}

@Composable
fun VoiceRecognitionScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var spokenText by remember { mutableStateOf("") }
    var isListening by remember { mutableStateOf(false) }
    val speechRecognizer = remember { SpeechRecognizer.createSpeechRecognizer(context) }

    DisposableEffect(Unit) {
        val recognitionListener = object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) { isListening = true }
            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty()) {
                    spokenText = if (spokenText.isEmpty()) matches[0] else "$spokenText ${matches[0]}"
                }
                startListening(speechRecognizer)
            }
            override fun onError(error: Int) {
                isListening = false
                if (error == SpeechRecognizer.ERROR_NO_MATCH || error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT) {
                    startListening(speechRecognizer)
                }
            }
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() { isListening = false }
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        }
        speechRecognizer.setRecognitionListener(recognitionListener)

        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(context as Activity, arrayOf(Manifest.permission.RECORD_AUDIO), 101)
        } else {
            startListening(speechRecognizer)
        }

        onDispose {
            speechRecognizer.destroy()
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        BasicTextField(
            value = spokenText,
            onValueChange = { spokenText = it },
            // *** CHANGE 1: Added a TextStyle to set the color and font size ***
            textStyle = TextStyle(
                color = Color.White,
                fontSize = 18.sp
            ),
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                // *** CHANGE 2: Added a background color for readability ***
                .background(Color.DarkGray)
                .border(1.dp, MaterialTheme.colorScheme.primary)
                .padding(8.dp)
                .verticalScroll(rememberScrollState()),
            decorationBox = { innerTextField ->
                if (spokenText.isEmpty()) {
                    Text(
                        text = if (isListening) "Listening..." else "Press 'Start Listening'",
                        color = Color.LightGray // Changed placeholder to light gray for contrast
                    )
                }
                innerTextField()
            }
        )

        Button(
            onClick = {
                if (isListening) {
                    speechRecognizer.stopListening()
                } else {
                    startListening(speechRecognizer)
                }
            },
            modifier = Modifier.padding(top = 16.dp)
        ) {
            Text(if (isListening) "Stop Listening" else "Start Listening")
        }
    }
}

private fun startListening(speechRecognizer: SpeechRecognizer) {
    val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
        putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
    }
    speechRecognizer.startListening(intent)
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    VoiceTotextTheme {
        VoiceRecognitionScreen()
    }
}