package com.example.streamingrecipter

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class UiState(
    val isStreaming: Boolean = false,
    val statusText: String = "Ready to stream",
    val serverResponse: String = ""
)

class MainViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val audioStreamer: AudioStreamer

    init {
        val listener = object : AudioStreamListener {
            override fun onConnected() {
                viewModelScope.launch {
                    _uiState.update { it.copy(isStreaming = true, statusText = "Connected and streaming...") }
                }
            }
            override fun onMessage(text: String) {
                viewModelScope.launch {
                    _uiState.update { it.copy(serverResponse = "Server: $text") }
                }
            }
            override fun onDisconnected() {
                viewModelScope.launch {
                    _uiState.update { it.copy(isStreaming = false, statusText = "Disconnected. Ready to stream.") }
                }
            }
            override fun onError(error: String) {
                viewModelScope.launch {
                    _uiState.update { it.copy(isStreaming = false, statusText = "Error: $error") }
                }
            }
        }
        audioStreamer = AudioStreamer(listener)
    }

    fun toggleStreaming() {
        if (_uiState.value.isStreaming) {
            audioStreamer.stopStreaming()
        } else {
            // !!! IMPORTANT !!!
            // Replace this with your actual WebSocket server URL
            val webSocketUrl = "ws://127.0.0.1:8765/ws"
//            val webSocketUrl = "ws://192.168.1.84:8765/ws"

            _uiState.update { it.copy(statusText = "Connecting...") }
            audioStreamer.startStreaming(webSocketUrl)
        }
    }

    override fun onCleared() {
        super.onCleared()
        // Ensure resources are released when ViewModel is destroyed
        audioStreamer.stopStreaming()
    }
}