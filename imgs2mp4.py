import cv2
import os

def simple_image_to_video(image_folder, output_file='output.mp4', fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png', '.jpeg'))]
    images.sort()
    
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    
    video.release()
    print(f"Video created: {output_file}")

# Usage
simple_image_to_video("scene_flow_analysis/DJI_20250516151729_0005_V-100f-fps30/images", 'output.mp4', 30)