import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import glob
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14

# Define the data structure
data = {
    "NeRF": {
        "rgb": "/home/zzz2004/DynamicNeRF/render_results/DJI_20250516151729_0005_V-no_combine/rgb_map_s",
    },
    "InstantNGP": {
        "rgb": "/mnt/c/Users/13694/instantNGP/output/Field-vibration",
    },
    "Gaussian-splatting": {
        "rgb": "/mnt/c/Users/13694/gaussian-splatting/output/wind-blade-vibration-capture3-fps30-100/train/ours_30000/renders",
        "depth": "/mnt/c/Users/13694/gaussian-splatting/output/wind-blade-vibration-capture3-fps30-100/train/ours_30000/depths"
    },
    "Yang et al": {
        "rgb": "/mnt/c/Users/13694/Deformable-3D-Gaussians/output/capture3-fps30-100/train/ours_40000/renders",
        "depth": "/mnt/c/Users/13694/Deformable-3D-Gaussians/output/capture3-fps30-100/train/ours_40000/depth"
    },
    "Gao et al": {
        "rgb": "/home/zzz2004/DynamicNeRF/render_results/DJI_20250516151729_0005_V-no_combine-original/rgb_map_full",
        "depth": "/home/zzz2004/DynamicNeRF/render_results/DJI_20250516151729_0005_V-no_combine-original/depth_map_full"
    },
    "Ours": {
        "rgb": "/home/zzz2004/DynamicNeRF/render_results/DJI_20250516151729_0005_V-no_combine-fullflow/rgb_map_d",
        "depth": "/home/zzz2004/DynamicNeRF/render_results/DJI_20250516151729_0005_V-no_combine-fullflow/depth_map_d"
    },
    "GT": {
        "rgb": "/home/zzz2004/DynamicNeRF/data/DJI_20250516151729_0005_V-100f-fps30/images"
    }
}

# Define the indices we want to display
indices = [0, 33, 66, 99]

# Function to find the correct image file for a given index
def find_image_file(directory, index):
    # Get all image files in the directory
    image_extensions = ['png', 'jpg', 'jpeg']
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(directory, f"*.{ext}")))
    
    # Sort files to ensure consistent ordering
    files.sort()
    
    # Check if we have at least 100 files
    if len(files) < 100:
        print(f"Warning: Only {len(files)} images found in {directory}")
        return None
    
    # Return the file at the requested index
    if index < len(files):
        return files[index]
    else:
        print(f"Warning: Index {index} out of range for {directory}")
        return None

# Calculate the number of rows needed
n_methods = len(data)
n_cols = len(indices)

# Count rows (each method has RGB, some have depth too)
n_rows = 0
for method, paths in data.items():
    n_rows += 1  # For RGB
    if "depth" in paths:
        n_rows += 1  # For depth if available

# Create the figure with appropriate size
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 1.75))

if n_rows == 1:
    axes = np.array([axes])  # Ensure axes is 2D

# Track current row
current_row = 0

# For each method, load and display images
for method, paths in data.items():
    # Display RGB images
    rgb_path = paths["rgb"]
    for col_idx, idx in enumerate(indices):
        img_path = find_image_file(rgb_path, idx)
        
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path)
            ax = axes[current_row, col_idx]
            ax.imshow(np.array(img))
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add column headers for the first row
            if current_row == 0:
                ax.set_title(f"View {col_idx+1}")
                
            # Add row label for the first column
            if col_idx == 0:
                ax.set_ylabel(f"{method}\nRGB", rotation=90, )
        else:
            print(f"Warning: File not found for index {idx} in {rgb_path}")
            # Create an empty image
            ax = axes[current_row, col_idx]
            ax.imshow(np.zeros((100, 100, 3)))
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(f"{method}\nRGB", rotation=90, )
    
    current_row += 1
    
    # Display depth images if available
    if "depth" in paths:
        depth_path = paths["depth"]
        for col_idx, idx in enumerate(indices):
            depth_img_path = find_image_file(depth_path, idx)
            
            if depth_img_path and os.path.exists(depth_img_path):
                depth_img = Image.open(depth_img_path)
                ax = axes[current_row, col_idx]
                # Convert to grayscale if it's not already
                if depth_img.mode != 'L':
                    depth_img = depth_img.convert('L')
                ax.imshow(np.array(depth_img), cmap='viridis')
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add row label for the first column
                if col_idx == 0:
                    ax.set_ylabel(f"{method}\nDepth", rotation=90)
            else:
                print(f"Warning: Depth file not found for index {idx} in {depth_path}")
                # Create an empty image
                ax = axes[current_row, col_idx]
                ax.imshow(np.zeros((100, 100)), cmap='viridis')
                ax.set_xticks([])
                ax.set_yticks([])
                if col_idx == 0:
                    ax.set_ylabel(f"{method}\nDepth", rotation=90)
        
        current_row += 1

plt.tight_layout()
plt.savefig("method_comparison.png", dpi=300, bbox_inches='tight')
plt.show()