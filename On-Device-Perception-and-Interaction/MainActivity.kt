package com.plcoding.cameraxguide

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaPlayer
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.util.Size
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.*
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okio.ByteString
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

/**
 * Main Activity handling CameraX, Audio Recording, and WebSocket/HTTP Streaming.
 */
class MainActivity : AppCompatActivity() {

    // ------------------------------------------------------------------------
    // Configuration & Constants
    // ------------------------------------------------------------------------
    companion object {
        private const val TAG = "MainActivity"

        // Server Configuration - Modify these for your environment
        private const val SERVER_IP = "127.0.0.1"
        private const val SERVER_PORT = "5000"

        // Constructed URLs
        private const val WS_URL = "ws://$SERVER_IP:$SERVER_PORT/ws"
        private const val HTTP_UPLOAD_PHOTO_URL = "http://$SERVER_IP:$SERVER_PORT/process_image"
        private const val HTTP_UPLOAD_ALL_URL = "http://$SERVER_IP:$SERVER_PORT/process"

        private val APP_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.INTERNET
        )
    }

    // ------------------------------------------------------------------------
    // Camera & Image Config
    // ------------------------------------------------------------------------
    private lateinit var controller: LifecycleCameraController

    // Preview control
    private val showPreview = true

    // Mode Selection:
    // 0 => Audio + Photo (Legacy)
    // 1 => Real-time Video Stream (Simulated via frames)
    // 2 => CameraX Native Video Recording
    // 3 => Continuous Photo Capture
    // 4 => Video Stream + Continuous Audio (Current default)
    private val mode = 4

    // Resolution & Quality
    private val targetResolution = Size(1920, 1080)
    private var imageAnalysis_interval = 200        // Frame interval
    private var imageAnalysis_quality = 25          // JPEG quality
    private val defaultZoomRatio = 1.75f
    private val resizedImageWidth = 1920
    private val resizedImageHeight = 1080
    private val topCropRatio = 1000f

    // Capture utilities
    private val captureInterval = 2000L
    private var startTime = 0L

    // ------------------------------------------------------------------------
    // Audio Config
    // ------------------------------------------------------------------------
    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private val bufferSize by lazy {
        AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
    }

    // Thresholds & Timings
    private val amplitudeThreshold_input = 1200
    private val amplitudeThreshold_output = 6000
    private val silenceDuration = 450L
    private val minRecordingDuration = 755L

    // Audio State
    private var audioRecord: AudioRecord? = null
    private var isAudioThreadActive = AtomicBoolean(false)
    private var isWritingToFile = false
    private var fileStartTime = 0L
    private var audioFilePath: String? = null
    private var ringBuffer: ByteArray? = null
    private var ringBufferPos = 0
    private val ringBufferSize = 16000
    private var fileOutputStream: FileOutputStream? = null
    private var silenceTimer: Handler? = null

    // ------------------------------------------------------------------------
    // Network & Playback
    // ------------------------------------------------------------------------
    private val wsClient by lazy { OkHttpClient() }
    private var webSocket: WebSocket? = null
    private var mediaPlayer: MediaPlayer? = null
    private var lastFrameSendTime = 0L

    private val cameraProviderFuture by lazy {
        ProcessCameraProvider.getInstance(this)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Check permissions
        if (!hasRequiredPermissions()) {
            ActivityCompat.requestPermissions(this, APP_PERMISSIONS, 0)
        }

        setupCamera()
        setupUI()

        // Mode Initialization Logic
        when (mode) {
            0 -> {
                ringBuffer = ByteArray(ringBufferSize)
                startContinuousAudioReading()
            }
            1 -> {
                initWebSocket()
                startVideoStreaming()
            }
            3 -> {
                startContinuousPhotoCapture()
            }
            4 -> {
                initWebSocket()
                startVideoStreaming()
                ringBuffer = ByteArray(ringBufferSize)
                startContinuousAudioReading()
            }
        }
    }

    private fun setupCamera() {
        controller = LifecycleCameraController(applicationContext).apply {
            bindToLifecycle(this@MainActivity)
            // Configure Use Cases based on mode
            when (mode) {
                0, 3 -> setEnabledUseCases(CameraController.IMAGE_CAPTURE)
                1, 4 -> setEnabledUseCases(CameraController.IMAGE_ANALYSIS)
                // 2 -> setEnabledUseCases(CameraController.VIDEO_CAPTURE) // Reserved
            }
            isTapToFocusEnabled = false
        }
    }

    private fun setupUI() {
        setContent {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black)
            ) {
                // Camera Preview (controlled by alpha)
                Box(
                    modifier = Modifier
                        .size(100.dp)
                        .align(Alignment.BottomEnd)
                        .padding(16.dp)
                        .alpha(if (showPreview) 1f else 0f)
                ) {
                    CameraPreview(
                        controller = controller,
                        modifier = Modifier.fillMaxSize()
                    )
                }

                // Tap to close overlay
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .pointerInput(Unit) {
                            detectTapGestures(onTap = { finish() })
                        }
                )
            }
        }
    }

    // ------------------------------------------------------------------------
    // WebSocket Logic
    // ------------------------------------------------------------------------
    private fun initWebSocket() {
        val request = Request.Builder()
            .url(WS_URL)
            .build()

        webSocket = wsClient.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d("WebSocket", "Connection established")
            }

            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                Log.d("WebSocket", "Received audio data, size: ${bytes.size}")
                playAudioFromBytes(bytes.toByteArray())
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.d("WebSocket", "Received text: $text")
                try {
                    val jsonObject = JSONObject(text)
                    if (jsonObject.has("imageAnalysis_interval") && jsonObject.has("imageAnalysis_quality")) {
                        imageAnalysis_interval = jsonObject.getInt("imageAnalysis_interval")
                        imageAnalysis_quality = jsonObject.getInt("imageAnalysis_quality")
                        Log.d("WebSocket", "Updated config: Interval=$imageAnalysis_interval, Quality=$imageAnalysis_quality")
                    }
                } catch (e: Exception) {
                    Log.e("WebSocket", "JSON parse error: ${e.message}")
                }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                Log.d("WebSocket", "Closing: $code / $reason")
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e("WebSocket", "Connection failed", t)
            }
        })
    }

    // ------------------------------------------------------------------------
    // Video / Image Streaming Logic
    // ------------------------------------------------------------------------
    private fun startVideoStreaming() {
        val resolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(
                    targetResolution,
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                )
            )
            .build()

        val imageAnalysisBuilder = ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)

        // Lock Auto Exposure
        Camera2Interop.Extender(imageAnalysisBuilder)
            .setCaptureRequestOption(android.hardware.camera2.CaptureRequest.CONTROL_AE_LOCK, true)

        val imageAnalysis = imageAnalysisBuilder.build()

        imageAnalysis.setAnalyzer(
            Executors.newSingleThreadExecutor(),
            ImageAnalysis.Analyzer { imageProxy ->
                try {
                    val bitmap = imageProxy.toBitmap()
                    if (shouldSendFrame()) {
                        uploadVideoFrame(bitmap)
                    }
                } finally {
                    imageProxy.close()
                }
            }
        )

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            cameraProvider.unbindAll()
            try {
                val camera = cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    imageAnalysis
                )

                // Set Zoom
                camera.cameraControl.setZoomRatio(defaultZoomRatio)
                    .addListener({
                        Log.d("Camera", "Zoom set to: $defaultZoomRatio")
                    }, ContextCompat.getMainExecutor(this))

            } catch (e: Exception) {
                Log.e("Camera", "Binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun uploadVideoFrame(bitmap: Bitmap) {
        try {
            val originalWidth = bitmap.width
            val originalHeight = bitmap.height
            val target169Height = (originalWidth * 9f / 16f).toInt()

            // Crop logic
            val croppedBitmap = if (target169Height < originalHeight) {
                val cropHeightDiff = originalHeight - target169Height
                val bottomCropHeight = (cropHeightDiff / (topCropRatio + 1)).toInt()
                val topCropHeight = cropHeightDiff - bottomCropHeight
                val cropOffsetY = topCropHeight

                Bitmap.createBitmap(
                    bitmap, 0, cropOffsetY, originalWidth, target169Height
                )
            } else {
                bitmap
            }

            // Resize logic
            val resizedBitmap = Bitmap.createScaledBitmap(
                croppedBitmap, resizedImageWidth, resizedImageHeight, true
            )

            val bos = ByteArrayOutputStream()
            resizedBitmap.compress(Bitmap.CompressFormat.JPEG, imageAnalysis_quality, bos)
            val imageBytes = bos.toByteArray()

            webSocket?.let { ws ->
                // Header: 1, 1, 1, 1 (Image Flag)
                val header = ByteArray(4) { 1 }
                val dataWithHeader = ByteArray(4 + imageBytes.size)
                System.arraycopy(header, 0, dataWithHeader, 0, 4)
                System.arraycopy(imageBytes, 0, dataWithHeader, 4, imageBytes.size)

                if (ws.send(ByteString.of(*dataWithHeader))) {
                    Log.d("WebSocket", "Sent image frame, size: ${dataWithHeader.size}")
                }
            } ?: initWebSocket()

            bitmap.recycle()
            if (bitmap != croppedBitmap) croppedBitmap.recycle()
            resizedBitmap.recycle()

        } catch (e: Exception) {
            Log.e("ImageProcess", "Error sending image", e)
        }
    }

    private fun shouldSendFrame(): Boolean {
        val now = System.currentTimeMillis()
        return if (now - lastFrameSendTime > imageAnalysis_interval) {
            lastFrameSendTime = now
            true
        } else {
            false
        }
    }

    // ------------------------------------------------------------------------
    // Audio Processing Logic
    // ------------------------------------------------------------------------
    private fun startContinuousAudioReading() {
        if (isAudioThreadActive.get()) return

        audioRecord = AudioRecord.Builder()
            .setAudioSource(android.media.MediaRecorder.AudioSource.VOICE_RECOGNITION)
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setEncoding(audioFormat)
                    .setChannelMask(channelConfig)
                    .build()
            )
            .setBufferSizeInBytes(bufferSize)
            .build()

        audioRecord?.startRecording()
        isAudioThreadActive.set(true)

        CoroutineScope(Dispatchers.Default).launch {
            val readBuffer = ByteArray(bufferSize)
            while (isAudioThreadActive.get() && audioRecord?.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                val bytesRead = audioRecord?.read(readBuffer, 0, readBuffer.size) ?: 0
                if (bytesRead > 0) {
                    val amplifiedBuffer = amplifyAudio(readBuffer, bytesRead)
                    val amplitude = computeAmplitude(amplifiedBuffer, bytesRead)

                    withContext(Dispatchers.Main) {
                        handleAmplitude(amplitude)
                    }

                    storeInRingBuffer(amplifiedBuffer, bytesRead)

                    if (isWritingToFile) {
                        writeDataToFile(amplifiedBuffer, bytesRead)
                    }
                }
            }
        }
    }

    private fun amplifyAudio(buffer: ByteArray, bytesRead: Int): ByteArray {
        val gain = 5.0f
        val amplifiedBuffer = buffer.clone()

        var i = 0
        while (i < bytesRead - 1) {
            val low = amplifiedBuffer[i].toInt() and 0xFF
            val high = amplifiedBuffer[i + 1].toInt()
            var sample = (high shl 8) or low

            sample = (sample * gain).toInt().coerceIn(-32768, 32767)

            amplifiedBuffer[i] = (sample and 0xFF).toByte()
            amplifiedBuffer[i + 1] = (sample shr 8).toByte()
            i += 2
        }
        return amplifiedBuffer
    }

    private fun computeAmplitude(data: ByteArray, length: Int): Int {
        var max = 0
        var i = 0
        while (i < length - 1) {
            val low = data[i].toInt() and 0xFF
            val high = data[i + 1].toInt()
            val sample = (high shl 8) or low
            val absVal = abs(sample)
            if (absVal > max) max = absVal
            i += 2
        }
        return max
    }

    private fun handleAmplitude(amplitude: Int) {
        if (amplitude > amplitudeThreshold_output) {
            if (mediaPlayer?.isPlaying == true) {
                mediaPlayer?.stop()
                mediaPlayer?.release()
                mediaPlayer = null
            }
        } else {
            if (amplitude > amplitudeThreshold_input) {
                if (!isWritingToFile) startWritingFile()
                silenceTimer?.removeCallbacksAndMessages(null)
                silenceTimer = null
            } else {
                if (isWritingToFile && silenceTimer == null) {
                    silenceTimer = Handler(Looper.getMainLooper())
                    silenceTimer?.postDelayed({ stopWritingFile() }, silenceDuration)
                }
            }
        }
    }

    private fun storeInRingBuffer(data: ByteArray, length: Int) {
        val rb = ringBuffer ?: return
        val toCopy = if (length > rb.size) rb.size else length
        if (toCopy <= 0) return

        if (ringBufferPos + toCopy <= rb.size) {
            System.arraycopy(data, 0, rb, ringBufferPos, toCopy)
            ringBufferPos += toCopy
        } else {
            val firstPart = rb.size - ringBufferPos
            System.arraycopy(data, 0, rb, ringBufferPos, firstPart)
            val remaining = toCopy - firstPart
            System.arraycopy(data, firstPart, rb, 0, remaining)
            ringBufferPos = remaining
        }
    }

    private fun startWritingFile() {
        if (isWritingToFile) return
        isWritingToFile = true
        fileStartTime = System.currentTimeMillis()

        val audioFile = File(getExternalFilesDir(null), "recorded_audio.wav")
        audioFilePath = audioFile.absolutePath
        if (audioFile.exists()) audioFile.delete()

        fileOutputStream = FileOutputStream(audioFile)
        writeWavHeader(fileOutputStream!!, sampleRate, channelConfig, AudioFormat.ENCODING_PCM_16BIT)

        ringBuffer?.let {
            fileOutputStream!!.write(it, ringBufferPos, it.size - ringBufferPos)
            fileOutputStream!!.write(it, 0, ringBufferPos)
        }
        Log.d(TAG, "Started writing audio file")
    }

    private fun writeDataToFile(data: ByteArray, length: Int) {
        try {
            fileOutputStream?.write(data, 0, length)
        } catch (e: IOException) {
            Log.e(TAG, "Write file error", e)
        }
    }

    private fun stopWritingFile() {
        if (!isWritingToFile) return
        isWritingToFile = false

        silenceTimer?.removeCallbacksAndMessages(null)
        silenceTimer = null

        val duration = System.currentTimeMillis() - fileStartTime
        fileStartTime = 0

        fileOutputStream?.flush()
        fileOutputStream?.close()
        fileOutputStream = null

        if (duration < minRecordingDuration) {
            audioFilePath?.let { File(it).takeIf { f -> f.exists() }?.delete() }
            return
        }

        audioFilePath?.let { path ->
            val file = File(path)
            fixWavHeader(file)

            if (mode == 4) {
                sendAudioFileViaWebSocket(file)
            } else {
                takePhoto(controller) { bitmap ->
                    uploadAudioAndPhoto(file, bitmap)
                }
            }
        }
    }

    private fun sendAudioFileViaWebSocket(audioFile: File) {
        try {
            val audioBytes = audioFile.readBytes()
            webSocket?.let { ws ->
                // Header: 0, 0, 0, 0 (Audio Flag)
                val header = ByteArray(4) { 0 }
                val dataWithHeader = ByteArray(4 + audioBytes.size)
                System.arraycopy(header, 0, dataWithHeader, 0, 4)
                System.arraycopy(audioBytes, 0, dataWithHeader, 4, audioBytes.size)

                if (ws.send(dataWithHeader.toByteString())) {
                    Log.d("WebSocket", "Sent audio file, size: ${dataWithHeader.size}")
                }
            }
            audioFile.delete()
        } catch (e: Exception) {
            Log.e("WebSocket", "Error sending audio", e)
        }
    }

    // ------------------------------------------------------------------------
    // Playback Logic
    // ------------------------------------------------------------------------
    private fun playAudioFromBytes(audioBytes: ByteArray) {
        mediaPlayer?.release()
        mediaPlayer = MediaPlayer().apply {
            try {
                val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
                val maxVolume = audioManager.getStreamMaxVolume(AudioManager.STREAM_ALARM)
                audioManager.setStreamVolume(AudioManager.STREAM_ALARM, maxVolume, 0)

                @Suppress("DEPRECATION")
                setAudioStreamType(AudioManager.STREAM_ALARM)

                val tempFile = File.createTempFile("audio_ws", ".wav", cacheDir).apply {
                    deleteOnExit()
                }
                FileOutputStream(tempFile).use { it.write(audioBytes) }

                setDataSource(tempFile.path)
                prepareAsync()
                setOnPreparedListener { it.start() }
                setOnCompletionListener {
                    it.release()
                    mediaPlayer = null
                }
                setOnErrorListener { mp, _, _ ->
                    mp.release()
                    mediaPlayer = null
                    true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Play audio error", e)
                release()
                mediaPlayer = null
            }
        }
    }

    private fun playOutput(outputUrl: String) {
        // Logic similar to playAudioFromBytes but from URL
        mediaPlayer?.release()
        mediaPlayer = MediaPlayer().apply {
            try {
                // Audio Manager setup repeated for safety
                val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
                val maxVolume = audioManager.getStreamMaxVolume(AudioManager.STREAM_ALARM)
                audioManager.setStreamVolume(AudioManager.STREAM_ALARM, maxVolume, 0)

                @Suppress("DEPRECATION")
                setAudioStreamType(AudioManager.STREAM_ALARM)

                setDataSource(outputUrl)
                prepareAsync()
                setOnPreparedListener { it.start() }
                setOnCompletionListener {
                    it.release()
                    mediaPlayer = null
                }
            } catch (e: Exception) {
                release()
                mediaPlayer = null
            }
        }
    }

    // ------------------------------------------------------------------------
    // Photo & HTTP Upload Logic
    // ------------------------------------------------------------------------
    private fun startContinuousPhotoCapture() {
        startTime = System.currentTimeMillis()
        val interval = 200L
        CoroutineScope(Dispatchers.IO).launch {
            while (true) {
                takePhoto(controller) { bitmap ->
                    uploadPhoto(bitmap)
                }
                delay(interval)
            }
        }
    }

    private fun takePhoto(
        controller: LifecycleCameraController,
        onPhotoTaken: (Bitmap) -> Unit
    ) {
        controller.takePicture(
            ContextCompat.getMainExecutor(applicationContext),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val matrix = Matrix().apply {
                        postRotate(image.imageInfo.rotationDegrees.toFloat())
                    }
                    val rotatedBitmap = Bitmap.createBitmap(
                        image.toBitmap(), 0, 0, image.width, image.height, matrix, true
                    )
                    onPhotoTaken(rotatedBitmap)
                    image.close()
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e("Camera", "Photo capture failed", exception)
                }
            }
        )
    }

    private fun uploadPhoto(bitmap: Bitmap) {
        val client = getOkHttpClient()
        val bos = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, bos)
        val imageRequestBody = bos.toByteArray().toRequestBody("image/jpeg".toMediaTypeOrNull())

        val multipartBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("videoFrame", "photo.jpg", imageRequestBody)
            .build()

        val request = Request.Builder()
            .url(HTTP_UPLOAD_PHOTO_URL)
            .post(multipartBody)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e("Upload", "Photo upload failed", e)
            }
            override fun onResponse(call: Call, response: Response) {
                response.close() // Ensure we close the body
            }
        })
    }

    private fun uploadAudioAndPhoto(audioFile: File, bitmap: Bitmap) {
        val client = getOkHttpClient()
        val bos = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, bos)

        val audioRequestBody = audioFile.asRequestBody("audio/wav".toMediaTypeOrNull())
        val imageRequestBody = bos.toByteArray().toRequestBody("image/jpeg".toMediaTypeOrNull())

        val multipartBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("audio", audioFile.name, audioRequestBody)
            .addFormDataPart("image", "photo.jpg", imageRequestBody)
            .build()

        val request = Request.Builder()
            .url(HTTP_UPLOAD_ALL_URL)
            .post(multipartBody)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e("Upload", "Upload failed", e)
            }

            override fun onResponse(call: Call, response: Response) {
                response.use {
                    if (it.isSuccessful) {
                        val jsonString = it.body?.string() ?: ""
                        val jsonObject = JSONObject(jsonString)
                        val outputUrl = jsonObject.optString("audio_url", "")
                        if (outputUrl.isNotEmpty()) {
                            runOnUiThread { playOutput(outputUrl) }
                        }
                    }
                }
            }
        })
    }

    private fun getOkHttpClient(): OkHttpClient {
        return OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .build()
    }

    // ------------------------------------------------------------------------
    // WAV Header Utilities
    // ------------------------------------------------------------------------
    private fun writeWavHeader(out: FileOutputStream, sampleRate: Int, channelConfig: Int, audioFormat: Int) {
        val channels = if (channelConfig == AudioFormat.CHANNEL_IN_MONO) 1 else 2
        val bitsPerSample = if (audioFormat == AudioFormat.ENCODING_PCM_16BIT) 16 else 8
        val byteRate = sampleRate * channels * (bitsPerSample / 8)

        out.write("RIFF".toByteArray())
        out.write(intToByteArray(0))
        out.write("WAVE".toByteArray())
        out.write("fmt ".toByteArray())
        out.write(intToByteArray(16))
        out.write(shortToByteArray(1))
        out.write(shortToByteArray(channels.toShort()))
        out.write(intToByteArray(sampleRate))
        out.write(intToByteArray(byteRate))
        out.write(shortToByteArray((channels * bitsPerSample / 8).toShort()))
        out.write(shortToByteArray(bitsPerSample.toShort()))
        out.write("data".toByteArray())
        out.write(intToByteArray(0))
    }

    private fun fixWavHeader(wavFile: File) {
        RandomAccessFile(wavFile, "rw").use { raf ->
            val fileSize = raf.length().toInt()
            raf.seek(4)
            raf.write(intToByteArray(fileSize - 8))
            raf.seek(40)
            raf.write(intToByteArray(fileSize - 44))
        }
    }

    private fun shortToByteArray(value: Short): ByteArray =
        ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(value).array()

    private fun intToByteArray(value: Int): ByteArray =
        ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array()

    // ------------------------------------------------------------------------
    // Lifecycle & Permissions
    // ------------------------------------------------------------------------
    override fun onDestroy() {
        super.onDestroy()
        silenceTimer?.removeCallbacksAndMessages(null)
        isAudioThreadActive.set(false)

        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null

        mediaPlayer?.release()
        webSocket?.close(1000, "Activity Destroyed")
        wsClient.dispatcher.executorService.shutdown()
    }

    private fun hasRequiredPermissions(): Boolean {
        return APP_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(applicationContext, it) == PackageManager.PERMISSION_GRANTED
        }
    }
}