"""
Authors: Josef LaFranchise, Anirudha Shastri, Elio Khouri, Karthik Koduru
Adapted from run_video.py script from Depth Anything V2.
Added YOLO integration for metric depth. Added downscaling for depth calculations.
Date: 11/22/2024
CS 7180: Advanced Perception
run_webcam_metric_combined.py
"""

import argparse
import cv2
import matplotlib
import numpy as np
import torch

from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

from ultralytics import YOLO

from assistant import generte_object_reponse

if __name__ == '__main__':
    """
    Main entry point of the script. Loads a pretrained YOLO model to detect objects in the scene and
    add bounding boxes. Also runs the Depth Anything V2 metric model on the image to produce a depth which
    is used to determine the distance of object identified by YOLO.
    """ 
    
    # Initialize YOLO
    model = YOLO("yolo11n.pt")
    
    # Parse the provided arguments.
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--nodepth', dest='no_depth', action='store_true', help='hide the depth map')
    
    args = parser.parse_args()
    
    # Setup GPU device to run with pytorch (if available)
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
    
    dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20 # 20 for indoor, 80 for outdoor
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': max_depth})
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{args.encoder}.pth', map_location='cpu'))
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
        if frameIndex % 4 != 0:
            continue
        
        # Run YOLO and Depth Anything V2 on the frame.
        resized_frame = cv2.resize(frame , (window_width, window_height))
        results = model.predict(resized_frame)

        # Downscale the image for faster depth calculations.
        downscale_factor = 2
        down_scaled_frame = cv2.resize(frame, (window_width//downscale_factor, window_height//downscale_factor))

        depth = depth_anything.infer_image(down_scaled_frame, window_width//downscale_factor)   
        center_depth = depth[depth.shape[0]//2,depth.shape[1]//2]
        
        # Upscale the depth map back to original size.
        depth = cv2.resize(depth, (window_width, window_height))
        original_depth = depth.copy()
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)  
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        """ 
        Object Detection Loop
        
        Iterates over all of the bounding boxes of the objects in the scene to draw those bounding boxes
        over the depth image. Also displays the confidence for each object as well as the minimum distance
        for that object in the scene.    
        """
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
            conf = result.conf[0]
            cls = int(result.cls[0])
            
            min_depth = np.min(original_depth[x1:x2,y1:y2])
            
            label = f"{model.names[cls]} {conf:.2f} - {min_depth:.2f} m"
            
            if args.no_depth:
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(resized_frame, label, (x1, y1 -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            else:
                cv2.rectangle(depth, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(depth, label, (x1, y1 -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            generte_object_reponse(model.names[cls], min_depth, conf)

        # Lable the center depth of the image.
        if args.no_depth:
            cv2.putText(resized_frame, f"{center_depth:.2f} m", (depth.shape[0]//2,depth.shape[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        else:
            cv2.putText(depth, f"{center_depth:.2f} m", (depth.shape[0]//2,depth.shape[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Display results in a window.
        if args.no_depth:
            cv2.imshow("Depth Estimate", resized_frame)
        else:
            cv2.imshow("Video Feed", depth)

        # Break loop on 'q' key press and exit the application.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    print("[+] Successfully Cleaned Up Resources")
