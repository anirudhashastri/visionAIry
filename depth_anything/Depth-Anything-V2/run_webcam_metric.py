import argparse
import cv2
import matplotlib
import numpy as np
import torch

from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

from ultralytics import YOLO

if __name__ == '__main__':
    
    model = YOLO("yolo11n.pt")
    
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
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

    # source = 0
    source = "../../common/videos/20241104_221555.mp4"
        
    cap = cv2.VideoCapture(source)  # Initialize webcam

    window_width = 600
    window_height = 600

    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    print("[+] Preparing to read video stream")
    
    frameIndex = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frameIndex += 1
        if frameIndex % 4 != 0:
            continue
        
        resized_frame = cv2.resize(frame , (window_width, window_height))
        results = model.predict(resized_frame)

        downscale_factor = 2
        # Downscale the image
        down_scaled_frame = cv2.resize(frame, (window_width//downscale_factor, window_height//downscale_factor))

        depth = depth_anything.infer_image(down_scaled_frame, window_width//downscale_factor)   
        center_depth = depth[depth.shape[0]//2,depth.shape[1]//2]
        
        # Upscale to original size
        depth = cv2.resize(depth, (window_width, window_height))
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
            
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
            conf = result.conf[0]
            cls = int(result.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            
            cv2.rectangle(depth, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(depth, label, (x1, y1 -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.putText(depth, f"{center_depth:.2f} m", (depth.shape[0]//2,depth.shape[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Display results
        cv2.imshow("Depth Estimate", depth)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[+] Successfully Cleaned Up Resources")
