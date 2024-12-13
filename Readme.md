# visionAIry

---

## Team Members

**Teammate names:** Anirudha Shastri, Josef LaFranchise, Elio Khouri, Venkata Satya Naga Sai Karthik Koduru  
**Time Travel Days Used:** 0

---

## System Info

- **Operating System:** Windows 11, macOS
- **IDE:** Visual Studio Code, Android Studio, Jupyter Notebook

---

## Project Description

**Visionary** is an application designed to help visually impaired individuals navigate indoor spaces. The app uses real-time object detection with YOLO11n, depth estimation through Depth Anything V2, and natural language processing with a Large Language Model (LLM). Audio feedback is provided via Yapper Text-to-Speech.
The system processes live video streams from a mobile device, with computational tasks handled by a Python-based server backend. Users interact through a simple Android app. Optimizations like downscaled depth calculations and frame skipping ensure smooth performance.

**Key Features:**

**Object Detection:** Detects objects and doors using YOLO11n.

**Depth Estimation:** Measures object distances with Depth Anything V2.

**Language Processing:** Uses an LLM for generating responses.

**Audio Feedback:** Provides responses through Yapper TTS.

This project addresses challenges in assistive technology by providing reliable, real-time feedback. Future updates aim to add directional guidance and outdoor navigation.

---

## Instructions to Run the Files

### Peliminary Setup (Mac)
1. If on a Mac, Use this command to set the environment variables so you can run the models on cpu: ```export PYTORCH_ENABLE_MPS_FALLBACK=1```. Use this command to check the variable is set: ```echo $PYTORCH_ENABLE_MPS_FALLBACK```.

### Depth Anything V2:

1. Change to the ```./depth_anything/Depth_Anything_V2/``` directory.
2. Install the Depth-Anything-V2-Small for indoor enviroments (Hypersim) https://drive.google.com/file/d/1sH69FufnBDQmAJkZIf4lwVhBFr2bOBjE/view?usp=drive_link  Original Download Source: https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth
3. Create a folder called checkpoints and place the downloaded model inside of it.
4. Create a anaconda enviroment and install the libaries in requirements.txt.
5. Activate the enviroment.

#### Run with webcam

1. Run ```python run_webcam_metric_combined_flask.py --inputsource webcam --encoder vits```

#### Run with video

1. Download the common folder from the google drive folder: https://drive.google.com/drive/u/0/folders/1Szij42KgBW6JFPqv2YAGnSqQbktg6dK2
2. Place it at the top level of the repo.
3. Run ```python run_webcam_metric_combined.py --encoder vits``` 



### Setting up the Flask Server:
 
1. **Run the Depth-Anything V2 Flask Server**
 
   To start the server and enable streaming, follow these steps:
 
   - Navigate to the `Depth-Anything-V2` directory:
 
     ```bash
     cd .../visionAIry/depth_anything/Depth-Anything-V2
     ```
 
   - Run the `run_webcam_metric_combined_flask.py` script:
 
     ```bash
     python run_webcam_metric_combined_flask.py --encoder vits --inputsource phone
     ```
 
     The output will include details about the Flask server, such as:
 
     ```
     [+] Flask server is starting...
     [+] Server running on:
         - Local:   http://127.0.0.1:5001
         - Network: http://192.168.1.174:5001
     ```
 
     Note the `Network` address (e.g., `http://192.168.1.174:5001`).
 
2. **Update Android App Network Configuration**
 
   To integrate the Depth-Anything V2 server with the Android app, make sure you have android studio installed with the app opened.
 
   Open the AndroidApp folder at the top level of the repo.
   Open the following two files:
 
   - **MainActivity.kt**
 
     Edit the socket URL in `MainActivity.kt`:
 
     Path: `.../visionairy/app/src/main/java/edu/northeastern/visionairy/MainActivity.kt`
 
     ```kotlin
     private fun startStreaming(imageAnalysis: ImageAnalysis) {
         try {
             val opts = IO.Options()
             opts.forceNew = true
             opts.reconnection = true
             socket = IO.socket("http://192.168.1.174:5001", opts)
         } catch (e: Exception) {
             Log.e("SocketIO", "Error creating socket: ${e.message}")
             return
         }
     }
     ```
 
     Replace `http://192.168.1.174:5001` with the Network address from the Flask server output.
 
   - **network_security_config.xml**
 
     Update the network security configuration to allow traffic to the server:
 
     Path: `.../visionairy/app/src/main/res/xml/network_security_config.xml`
 
     ```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
<domain-config cleartextTrafficPermitted="true">
<domain includeSubdomains="true">192.168.1.174</domain>
</domain-config>
</network-security-config>
     ```
 
     Replace `192.168.1.174` with the IP address from the Flask server output. Note that the port number is not required in this file.

     
 
3. **Run the Android App**
 
   - Build and run the Android app in Android Studio.
   - The app will connect to the Flask server using the specified network address.
 
#### Run with flask and Android Studio

1. Run ```python run_webcam_metric_combined_flask.py --inputsource phone --encoder vits ```


---

## Required Libraries:

- matplotlib
- opencv-python
- torch
- torchvision
- flask
- flask-cors
- ultralytics
- numpy
- Pillow
- scipy
- scikit-image
- tqdm
- Flask-SocketIO
- eventlet
- groq
- yapper-tts

---

## File Descriptions
### Description for run_webcam_metric_combined_flask.py 
 
This script combines YOLO-based object detection with Depth Anything V2 for monocular depth estimation. It processes video streams from a phone camera(using Flask server) or webcam, overlays detected objects with bounding boxes and depth values, and optimizes performance through frame skipping and image downscaling. Detected data is sent to an LLM for generating context-aware responses, which are delivered as audio feedback via Yapper TTS, enabling dynamic spatial awareness and navigation assistance.
 
### Description for run_webcam_metric_combined.py
 
This script integrates object detection, depth estimation, and audio feedback . Using YOLO models for object and door detection, and Depth Anything V2 for depth estimation, it identifies objects and calculates their distances in real-time. Combined data is processed by a large language model (LLM) to generate verbal responses via Yapper TTS. Supporting video or webcam input, the system optimizes performance through frame downscaling and GPU acceleration, ensuring efficient and dynamic real-time feedback for enhanced spatial awareness

### Description for yoloTransferLearning.ipynb
This notebook demonstrates transfer learning using the YOLO object detection model. It includes steps for dataset preparation, model loading, training, evaluation, and visualization of predictions. The workflow fine-tunes a pre-trained YOLO model on a custom dataset to achieve accurate object detection results.

---

## File Structure

```plaintext
visionAIry/
├── Androidapp/
│   ├── app/
│   │   ├── src/
│   │   ├── build.gradle.kts
│   │   ├── proguard-rules.pro
│   ├── gradle/
│   │   ├── wrapper/
│   │   ├── libs.versions.toml
│   ├── build.gradle.kts
│   ├── gradle.properties
│   ├── gradlew
│   ├── gradlew.bat
│   ├── settings.gradle.kts
├── common/
│   ├── photos/
│   │   ├── ground_truth/
│   │   ├── predicted/
│   ├── videos/
│   │   ├── 20241104_221555.mp4
│   │   ├── 20241104_221732.mp4
├── DEPTH/
│   ├── depth.ipynb
├── depth_anything/
│   ├── datasets/
│   │   ├── coco8/
│   ├── Depth-Anything-V2/
│   │   ├── depth_anything_v2/
│   │   ├── metric_depth/
│   │   ├── runs/
│   │   ├── venv/
│   │   ├── __pycache__/
│   │   ├── app.py
│   │   ├── assistant.py
│   │   ├── bus.jpg
│   │   ├── DA-2K.md
│   │   ├── LICENSE
│   │   ├── LLMassistant.py
│   │   ├── phone_camera_input_flask.py
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── run.py
│   │   ├── run_video.py
│   │   ├── run_webcam.py
│   │   ├── run_webcam_metric.py
│   │   ├── run_webcam_metric_combined.py
│   │   ├── run_webcam_metric_combined_flask.py
│   │   ├── server.py
│   │   ├── train_yolo.py
│   │   ├── yolo11n.pt
│   │   ├── yolo_custom_weights.pt
├── YOLO/
│   ├── yolo11n.pt
│   ├── yoloTransferLearning.ipynb
├── Readme.md
├── yolo11n.pt
├── yolo_custom_weights.pt
```
