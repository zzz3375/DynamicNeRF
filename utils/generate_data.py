import os
import numpy as np
import imageio
import glob
import torch
import torchvision
import skimage.morphology
import argparse
import cv2
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def multi_view_multi_time(args):
    """Extract frames 300 to 500 from video and save with masks"""
    
    # Create output directories
    output_paths = ['images', 'images_colmap', 'background_mask']
    for path in output_paths:
        create_dir(os.path.join(args.data_dir, args.outputname, path))

    # Open video and set start position
    cap = cv2.VideoCapture(args.videopath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 230)
    
    # Process 200 frames
    for idx in range(100):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert and process frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W, _ = img.shape
        
        # Save images
        cv2.imwrite(os.path.join(args.data_dir, args.outputname, 'images', f"{idx:03d}.png"), 
                   cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.data_dir, args.outputname, 'images_colmap', f"{idx:03d}.jpg"), 
                   cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Save blank background mask
        background_mask = torch.ones(H, W, dtype=torch.float32, device=device)
        background_mask_np = ((background_mask.cpu().numpy() > 0.1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.data_dir, args.outputname, 'background_mask', f"{idx:03d}.jpg.png"), 
                   background_mask_np)
        
        print(f"Processed frame {idx}")

    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--videopath", type=str,
                        help='video path')
    parser.add_argument("--data_dir", type=str, default='../data/',
                        help='where to store data')
    parser.add_argument("--outputname", type=str)

    args = parser.parse_args()

    multi_view_multi_time(args)
