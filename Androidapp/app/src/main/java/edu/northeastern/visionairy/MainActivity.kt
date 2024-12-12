package edu.northeastern.visionairy

import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import edu.northeastern.visionairy.ui.theme.NeonRed
import io.socket.client.IO
import io.socket.client.Socket
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.os.Handler
import android.os.Looper


class MainActivity : ComponentActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private var socket: Socket? = null
    private val CAMERA_PERMISSION = android.Manifest.permission.CAMERA

    // ActivityResultLauncher for requesting permissions
    private val requestPermissionsLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                // Permission granted, initialize the camera
                initializeCamera()
            } else {
                // Permission denied, handle appropriately
                Log.e("Permissions", "Camera permission denied.")
                finish()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Check camera permission before proceeding
        if (checkSelfPermission(CAMERA_PERMISSION) != PackageManager.PERMISSION_GRANTED) {
            requestPermissionsLauncher.launch(CAMERA_PERMISSION)
        } else {
            // Permission already granted
            initializeCamera()
        }
    }

    private fun initializeCamera() {
        // Initialize the camera executor
        cameraExecutor = Executors.newSingleThreadExecutor()

        val cameraType = intent.getStringExtra("camera") ?: "back"

        setContent {
            CameraPreview(
                cameraType = cameraType,
                onExitClick = {
                    stopStreaming()
                    val intent = Intent(this, HomeActivity::class.java)
                    intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
                    startActivity(intent)
                },
                onCameraReady = { imageAnalysis -> startStreaming(imageAnalysis) }
            )
        }
    }

    private fun startStreaming(imageAnalysis: ImageAnalysis) {
        try {
            val opts = IO.Options()
            opts.forceNew = true
            opts.reconnection = true
            socket = IO.socket("http://10.0.0.3:5001", opts)
        } catch (e: Exception) {
            Log.e("SocketIO", "Error creating socket: ${e.message}")
            return
        }

        socket?.on(Socket.EVENT_CONNECT) {
            Log.d("SocketIO", "Connected to server.")
        }

        socket?.on(Socket.EVENT_CONNECT_ERROR) {
            Log.e("SocketIO", "Connection error: ${it.joinToString()}")
        }

        socket?.connect()

        // Handler and interval for throttling
        val throttleIntervalMillis = 333L // For ~3 frames per second
        var lastEmittedTime = 0L

        imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
            val currentTime = System.currentTimeMillis()
            if (currentTime - lastEmittedTime >= throttleIntervalMillis) {
                lastEmittedTime = currentTime

                try {
                    val byteArray = convertToByteArray(imageProxy)
                    if (byteArray.isNotEmpty()) {
                        Log.d("SocketIO", "Emitting frame of size: ${byteArray.size} bytes")
                        socket?.emit("frame", byteArray)
                    } else {
                        Log.e("ImageAnalysis", "Frame conversion failed; skipping frame.")
                    }
                } catch (e: Exception) {
                    Log.e("ImageAnalysis", "Error during frame analysis: ${e.message}", e)
                } finally {
                    imageProxy.close()
                }
            } else {
                imageProxy.close() // Close the frame if throttled
            }
        }
    }


    private fun convertToByteArray(image: ImageProxy): ByteArray {
        Log.d("ImageConversion", "Starting conversion for ImageProxy with dimensions: ${image.width}x${image.height}")

        try {
            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            Log.d("ImageConversion", "Buffer sizes - Y: $ySize, U: $uSize, V: $vSize")

            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = YuvImage(
                nv21,
                ImageFormat.NV21,
                image.width,
                image.height,
                null
            )

            val byteArrayOutputStream = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 85, byteArrayOutputStream)

            Log.d("ImageConversion", "Successfully compressed ImageProxy to JPEG. Size: ${byteArrayOutputStream.size()} bytes")
            return byteArrayOutputStream.toByteArray()
        } catch (e: Exception) {
            Log.e("ImageConversion", "Error during conversion: ${e.message}", e)
            return ByteArray(0)
        } finally {
            image.close()
        }
    }


    private fun stopStreaming() {
        socket?.disconnect()
        socket?.close()
        socket = null
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        stopStreaming()
    }
}

@Composable
fun CameraPreview(
    cameraType: String,
    onExitClick: () -> Unit,
    onCameraReady: (ImageAnalysis) -> Unit
) {
    val context = LocalContext.current

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { ctx ->
                val previewView = PreviewView(ctx)
                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder().build()
                    val imageAnalysis = ImageAnalysis.Builder().build()

                    preview.surfaceProvider = previewView.surfaceProvider
                    val cameraSelector = if (cameraType == "front") {
                        CameraSelector.DEFAULT_FRONT_CAMERA
                    } else {
                        CameraSelector.DEFAULT_BACK_CAMERA
                    }

                    cameraProvider.bindToLifecycle(
                        context as ComponentActivity,
                        cameraSelector,
                        preview,
                        imageAnalysis
                    )
                    onCameraReady(imageAnalysis)
                }, ContextCompat.getMainExecutor(ctx))
                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        IconButton(
            onClick = onExitClick,
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(16.dp)
        ) {
            Icon(
                painter = painterResource(id = android.R.drawable.ic_menu_close_clear_cancel),
                contentDescription = "Exit",
                modifier = Modifier.size(50.dp),
                tint = NeonRed
            )
        }
    }
}
