"""
Authors: Josef LaFranchise, Anirudha Shastri, Elio Khouri, Karthik Koduru
Adapted from run_video.py script from Depth Anything V2.
Added YOLO integration.
Date: 11/22/2024
CS 7180: Advanced Perception
run_webcam_metric_combined.py
"""

import argparse
import cv2
import matplotlib
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2

from ultralytics import YOLO

if __name__ == '__main__':
    """
    Main entry point of the script. Loads a pretrained YOLO model to detect objects in the scene and
    add bounding boxes. Also runs Depth Anything V2 on the image to produce a depth map which we then
    transfer the bounding boxes to.
    """ 
    
    # Initialize YOLO
    model = YOLO("yolo11n.pt")

    results = model.train(data="coco8.yaml", epochs = 3)
    results = model.val()
    results = model("https://ultralytics.com/images/bus.jpg")
    
    # Parse the provided arguments.
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    # Setup GPU device to run with pytorch (if available)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # source = 0 # Uncomment for webcam
    source = "../../common/videos/20241104_221555.mp4"
        
    # Initialize webcam or video
    cap = cv2.VideoCapture(source)  

    window_width = 600
    window_height = 600

    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    print("[+] Preparing to read video stream")
    
    frameIndex = 0
    
    """
    Capture Loop
    
    Process frames as fast as possible, whether from a video file or live camera feed. Terminates when
    the video ends, user quits the application, or camera is disconnected.
    """
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # We keep track of the frame index to allow us to allow us to process every nth frame.
        frameIndex += 1
        if frameIndex % 6 != 0:
            continue
        
        # Run YOLO and Depth Anything V2 on the frame.
        resized_frame = cv2.resize(frame , (window_width, window_height))
        results = model.predict(resized_frame)

        depth = depth_anything.infer_image(resized_frame, 600)
            
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
            
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # We loop over all of the object bounding and transfer them to the depth map image.
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
            conf = result.conf[0]
            cls = int(result.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            
            cv2.rectangle(depth, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(depth, label, (x1, y1 -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Display results with bounding boxes.
        cv2.imshow("Depth Estimate", depth)

        # Break loop on 'q' key press and exit the application.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    print("[+] Successfully Cleaned Up Resources")
