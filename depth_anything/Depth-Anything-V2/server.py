"""
Authors: Elio Khouri, Anirudha Shastri, Josef LaFranchise, Karthik Koduru
Date: 11/22/2024
CS 7180: Advanced Perception
server.py
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from the Android app

# Load YOLO model
yolo_model = YOLO("yolo11n.pt")

# Load Depth Anything model
depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
depth_model.load_state_dict(torch.load("path_to_depth_model.pth", map_location="cpu"))
depth_model = depth_model.eval()

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Get the image from the request
    file = request.files.get('frame')
    if not file:
        return jsonify({'error': 'No frame provided'}), 400

    # Read and decode the image
    np_img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # YOLO object detection
    yolo_results = yolo_model.predict(frame)
    detections = []
    for result in yolo_results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
        conf = result.conf[0].item()
        cls = int(result.cls[0])
        detections.append({
            'label': yolo_model.names[cls],
            'confidence': conf,
            'bbox': [x1, y1, x2, y2]
        })

    # Depth estimation
    depth = depth_model.infer_image(frame, frame.shape[1])
    center_depth = float(depth[depth.shape[0] // 2, depth.shape[1] // 2])  # Depth at the center of the frame

    # Prepare response
    response = {
        'detections': detections,
        'center_depth': center_depth
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Run the server on port 5000
