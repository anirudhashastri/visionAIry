'''
Author: Josef LaFranchise , Anirudha Shastri, elio khouri, Karthik Koduru
Adapted from run_video.py script from Depth Anything V2 Repo
Date: 11/22/2024
CS 7180: Advanced Perception
run_webcam_metric_combined_flask.py
'''
# FOR MPS
# import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# print("PYTORCH_ENABLE_MPS_FALLBACK set to:", os.environ["PYTORCH_ENABLE_MPS_FALLBACK"])

import argparse
import cv2
import matplotlib
import numpy as np
import torch
import threading

from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO
from assistant import generte_object_reponse
from flask import Flask, Response
from phone_camera_input_flask import get_latest_frame, start_server


if __name__ == '__main__':
    
    model_COCO = YOLO("yolo11n.pt")  # COCO model
    model_door = YOLO("yolo_custom_weights.pt")  # Door-specific model
    
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--nodepth', dest='no_depth', action='store_true', help='hide the depth map')
    parser.add_argument('--inputsource', type=str, default='webcam', choices=['webcam', 'phone'], help='Input source: webcam or phone camera') # For Flask support
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    if torch.cuda.is_available():
        print("[+] Using GPU Device")
    else:
        print("[-] No GPU Found")
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    
    dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20  # 20 for indoor, 80 for outdoor
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': max_depth})
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    
    # source = 0  # Set this for webcam
    # cap = cv2.VideoCapture(source)  # Initialize webcam
    
    # For Flask support
    if args.inputsource == 'phone':
        inputsource = 'phone'
        threading.Thread(target=start_server, daemon=True).start()  # Start Flask server for phone camera
        print("[+] Phone camera server started. Waiting for frames...")
    elif args.inputsource == 'webcam':
        inputsource = 0  # Default to webcam
        cap = cv2.VideoCapture(inputsource)
    else:
        raise ValueError(f"Unsupported source: {args.source}. Please use 'webcam' or 'phone'.")

    #print(f"Using source: {inputsource}")

    window_width = 1000
    window_height = 1000

    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    print("[+] Preparing to read video stream")
    
    frameIndex = 0
    
    """
    Capture Loop
    
    Process frames as fast as possible, whether from a video file or live camera feed.
    """
    
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    # For Flask support
    while True:
        if inputsource == 0:  # Webcam
            ret, frame = cap.read()
            if not ret:
                break
        elif inputsource == 'phone':  # Phone camera
            frame = get_latest_frame()
            if frame is None:
                continue  # Wait for the next frame
        else:
            raise ValueError("Unsupported source")
        
        frameIndex += 1
        #if frameIndex % 4 != 0:
        #    continue
        
        resized_frame = cv2.resize(frame, (window_width, window_height))
        
        # YOLO COCO model inference
        coco_results = model_COCO.predict(resized_frame)

        # YOLO Door model inference
        door_results = model_door.predict(resized_frame)

        # Combine predictions: filter door model to include only "door" class
        combined_results = []

        for box in coco_results[0].boxes:
            combined_results.append({"box": box, "source": "COCO"})

        for box in door_results[0].boxes:
            cls = int(box.cls[0])
            if model_door.names[cls] == "door":  # Only include door class
                combined_results.append({"box": box, "source": "DOOR"})

        # Depth estimation
        downscale_factor = 2
        down_scaled_frame = cv2.resize(frame, (window_width // downscale_factor, window_height // downscale_factor))

        depth = depth_anything.infer_image(down_scaled_frame, window_width // downscale_factor)
        center_depth = depth[depth.shape[0] // 2, depth.shape[1] // 2]
        
        # Upscale to original size
        depth = cv2.resize(depth, (window_width, window_height))
        original_depth = depth.copy()

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # Object Detection Loop
        for result in combined_results:
            box = result["box"]
            source = result["source"]
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            min_depth = np.min(original_depth[y1:y2, x1:x2])
            
            label = f"{model_COCO.names[cls] if source == 'COCO' else 'door'} {conf:.2f} - {min_depth:.2f} m"
            

            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            
            # Generate object response
            object_name = model_COCO.names[cls] if source == "COCO" else "door"
            generte_object_reponse(object_name, min_depth, conf)

        cv2.putText(resized_frame, f"{center_depth:.2f} m", (depth.shape[0]//2,depth.shape[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        
        # Display results
        cv2.imshow("Depth Estimate", resized_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # print(f"Using source: {inputsource}")
    

    # cap.release()
    # cv2.destroyAllWindows()

    # For Flask support
    if inputsource == 0:  # Webcam
        cap.release()
    cv2.destroyAllWindows()
    print("[+] Successfully Cleaned Up Resources")
