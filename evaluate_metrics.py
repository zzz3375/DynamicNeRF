"""
a python script that evaluate masked PSNR, SSIM, LPIPS
input: result_dir, gt_dir, mask_dir all is png format, the name is sequential and all folder have the same total image number
if shape not same, cv2 resize the output to gt shape 
use mask as a weight, do not use bbox of the mask
output: (pd datafame 1*6) overall PSNR,overall SSIM, overall LPIPS, masked PSNR, masked SSIM, masked LPIPS
rewrite the psnr and ssim, implement it by ourself not skimage
save to xlsx
"""
import os
import cv2
import numpy as np
import pandas as pd
# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch
from PIL import Image
import argparse
import re

def natural_sort_key(s):
    """Natural sort key function for numerical ordering"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_image(path, target_shape=None):
    """Load image and optionally resize to target shape"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    
    if target_shape is not None and img.shape[:2] != target_shape:
        img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    
    return img

def load_mask(path, target_shape=None):
    """Load mask and optionally resize to target shape"""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {path}")
    
    if target_shape is not None and mask.shape[:2] != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Normalize mask to 0-1
    mask = mask.astype(np.float32) / 255.0
    return mask

def psnr(img_1, img_2, mask=None, data_range=255):
    """
    Calculate PSNR between two images, optionally using a mask.
    
    Args:
        img_1: First image (numpy array)
        img_2: Second image (numpy array)
        mask: Binary mask (2D array) where 1 indicates valid pixels
        data_range: The dynamic range of the images (default 255 for uint8)
    
    Returns:
        PSNR value in dB
    """
    from math import log10
    img_1 = img_1.astype(np.float64)
    img_2 = img_2.astype(np.float64)
    
    if mask is not None:
        # Ensure mask has the same spatial dimensions
        if mask.ndim == 2 and img_1.ndim == 3:
            mask = mask[:, :, np.newaxis]
        
        # Apply mask
        masked_img1 = img_1 * mask
        masked_img2 = img_2 * mask
        
        # Calculate MSE only on masked regions
        if mask.sum() == 0:
            return float('inf')  # No valid pixels to compare
        
        mse = np.sum((masked_img1 - masked_img2) ** 2) / mask.sum()
    else:
        # Calculate MSE on entire image
        mse = np.mean((img_1 - img_2) ** 2)
    
    if mse == 0:
        return float('inf')  # Perfect match
    
    psnr_val = 20 * log10(data_range) - 10 * log10(mse)
    return psnr_val

def ssim(img_1, img_2, mask=None, data_range=255, channel_axis=2):
    """
    Calculate SSIM between two images, optionally using a mask.
    
    Args:
        img_1: First image (numpy array)
        img_2: Second image (numpy array)
        mask: Binary mask (2D array) where 1 indicates valid pixels
        data_range: The dynamic range of the images
        channel_axis: Axis containing color channels
    
    Returns:
        SSIM value
    """
    img_1 = img_1.astype(np.float64)
    img_2 = img_2.astype(np.float64)
    
    # Constants for stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    if mask is not None:
        # Ensure mask has the same spatial dimensions
        if mask.ndim == 2 and img_1.ndim == 3:
            mask_3d = mask[:, :, np.newaxis]
        else:
            mask_3d = mask
        
        # Apply mask
        masked_img1 = img_1 * mask_3d
        masked_img2 = img_2 * mask_3d
        
        # Calculate statistics only on masked regions
        if mask.sum() == 0:
            return 1.0  # No valid pixels, return perfect score
        
        # Mean values
        mu1 = np.sum(masked_img1, axis=(0, 1)) / mask.sum()
        mu2 = np.sum(masked_img2, axis=(0, 1)) / mask.sum()
        
        # Variance and covariance
        var1 = np.sum(mask_3d * (img_1 - mu1) ** 2, axis=(0, 1)) / mask.sum()
        var2 = np.sum(mask_3d * (img_2 - mu2) ** 2, axis=(0, 1)) / mask.sum()
        covar = np.sum(mask_3d * (img_1 - mu1) * (img_2 - mu2), axis=(0, 1)) / mask.sum()
    else:
        # Calculate statistics on entire image
        mu1 = np.mean(img_1, axis=(0, 1))
        mu2 = np.mean(img_2, axis=(0, 1))
        
        var1 = np.var(img_1, axis=(0, 1))
        var2 = np.var(img_2, axis=(0, 1))
        covar = np.mean((img_1 - mu1) * (img_2 - mu2), axis=(0, 1))
    
    # Calculate SSIM for each channel
    ssim_channels = ((2 * mu1 * mu2 + C1) * (2 * covar + C2)) / \
                   ((mu1**2 + mu2**2 + C1) * (var1 + var2 + C2))
    
    # Return mean SSIM across channels
    return np.mean(ssim_channels)

def calculate_psnr_ssim(img1, img2, mask=None):
    """Calculate PSNR and SSIM with optional masking"""
    if mask is not None:
        # Apply mask
        masked_img1 = img1 
        masked_img2 = img2 
        
        # Calculate metrics only on masked regions
        psnr_val = psnr(masked_img1, masked_img2,mask, data_range=255)
        ssim_val = ssim(masked_img1, masked_img2,mask,  data_range=255, channel_axis=2)
    else:
        psnr_val = psnr(img1, img2, data_range=255)
        ssim_val = ssim(img1, img2, data_range=255, channel_axis=2)
    
    return psnr_val, ssim_val

def calculate_lpips(img1, img2, mask=None, loss_fn=None):
    """Calculate LPIPS with optional masking"""
    # Convert to tensors
    transform = lpips.im2tensor
    img1_tensor = transform(img1).to(device)
    img2_tensor = transform(img2).to(device)
    
    if mask is not None:
        # Create mask tensor
        mask_tensor = torch.from_numpy(mask).float().to(device)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Apply mask to the feature maps
        with torch.no_grad():
            # Get features from the network
            feat0 = loss_fn.net.forward(img1_tensor)
            feat1 = loss_fn.net.forward(img2_tensor)
            
            # Resize mask to match each feature map size
            lpips_val = 0.0
            total_weight = 0.0
            
            for f0, f1 in zip(feat0, feat1):
                # Resize mask to match feature map size
                current_mask = torch.nn.functional.interpolate(
                    mask_tensor, size=f0.shape[2:], mode='nearest'
                )
                
                # Calculate LPIPS for this layer
                diff = (f0 - f1) ** 2
                layer_lpips = torch.sum(diff * current_mask) / (torch.sum(current_mask) + 1e-8)/ f0.shape[1]
                
                lpips_val += layer_lpips.item()
                total_weight += 1.0
            
            lpips_val /= total_weight
    else:
        lpips_val = loss_fn.forward(img1_tensor, img2_tensor).item()
    
    return lpips_val

def get_sorted_files(directory, extension='.png'):
    """Get files sorted numerically by their index"""
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    # Sort files numerically based on the numbers in the filename
    files.sort(key=natural_sort_key)
    return files

def evaluate_directory(result_dir, gt_dir, mask_dir):
    """Evaluate all images in the directories"""
    # Get sorted file lists
    result_files = get_sorted_files(result_dir)
    gt_files = get_sorted_files(gt_dir)
    mask_files = get_sorted_files(mask_dir)
    
    # Check if all directories have the same number of files
    if len(result_files) != len(gt_files) or len(result_files) != len(mask_files):
        print(f"Warning: Directory file counts don't match. Results: {len(result_files)}, GT: {len(gt_files)}, Masks: {len(mask_files)}")
        print("Using the minimum number of files across all directories")
        num_files = min(len(result_files), len(gt_files), len(mask_files))
        result_files = result_files[:num_files]
        gt_files = gt_files[:num_files]
        mask_files = mask_files[:num_files]
    
    print(f"Evaluating {len(result_files)} images...")
    print(f"First few files - Results: {result_files[:3]}, GT: {gt_files[:3]}, Masks: {mask_files[:3]}")
    
    # Initialize metrics
    overall_psnr, overall_ssim, overall_lpips = [], [], []
    masked_psnr, masked_ssim, masked_lpips = [], [], []
    
    # Initialize LPIPS
    loss_fn = lpips.LPIPS(net='alex').to(device)
    
    for i in range(len(result_files)):
        result_filename = result_files[i]
        gt_filename = gt_files[i]
        mask_filename = mask_files[i]
        
        try:
            # Load images
            gt_path = os.path.join(gt_dir, gt_filename)
            result_path = os.path.join(result_dir, result_filename)
            mask_path = os.path.join(mask_dir, mask_filename)
            
            # Load ground truth first to get target shape
            gt_img = load_image(gt_path)
            result_img = load_image(result_path, target_shape=gt_img.shape[:2])
            mask = load_mask(mask_path, target_shape=gt_img.shape[:2])
            
            if mask.sum() == 0: 
                print(f"{mask_filename} is full zero: reset to full one")
                mask = np.ones_like(mask)
            
            # Convert to float for calculations
            gt_img = gt_img.astype(np.float32)
            result_img = result_img.astype(np.float32)
            
            # Calculate overall metrics
            psnr_o, ssim_o = calculate_psnr_ssim(gt_img, result_img)
            lpips_o = calculate_lpips(gt_img, result_img, loss_fn=loss_fn)
            
            # Calculate masked metrics
            psnr_m, ssim_m = calculate_psnr_ssim(gt_img, result_img, mask)
            lpips_m = calculate_lpips(gt_img, result_img, mask, loss_fn=loss_fn)
            
            # Store results
            overall_psnr.append(psnr_o)
            overall_ssim.append(ssim_o)
            overall_lpips.append(lpips_o)
            masked_psnr.append(psnr_m)
            masked_ssim.append(ssim_m)
            masked_lpips.append(lpips_m)
            
            if i % 10 == 0:
                print(f"Processed {i+1}/{len(result_files)} images")
            
        except Exception as e:
            print(f"Error processing {result_filename}, {gt_filename}, {mask_filename}: {e}")
            continue
    
    # Calculate averages
    metrics = {
        'overall_PSNR': "{:.4g}".format(np.mean(overall_psnr)) if overall_psnr else 0,
        'overall_SSIM': "{:.4g}".format(np.mean(overall_ssim)) if overall_ssim else 0,
        'overall_LPIPS': "{:.4g}".format(np.mean(overall_lpips)) if overall_lpips else 0,
        'masked_PSNR': "{:.4g}".format(np.mean(masked_psnr)) if masked_psnr else 0,
        'masked_SSIM': "{:.4g}".format(np.mean(masked_ssim)) if masked_ssim else 0,
        'masked_LPIPS': "{:.4g}".format(np.mean(masked_lpips)) if masked_lpips else 0
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate image quality metrics with masking')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory with result images')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory with ground truth images')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory with mask images')
    parser.add_argument('--output', type=str, default='evaluation_results.xlsx', help='Output Excel file name')
    
    args = parser.parse_args()
    
    # Check if directories exist
    for dir_path in [args.result_dir, args.gt_dir, args.mask_dir]:
        if not os.path.exists(dir_path):
            raise ValueError(f"Directory does not exist: {dir_path}")
    
    # Evaluate
    metrics = evaluate_directory(args.result_dir, args.gt_dir, args.mask_dir)
    
    # Create DataFrame
    df = pd.DataFrame([metrics])
    
    # Save to Excel
    df.to_excel(args.output, index=False)
    print(f"Results saved to {args.output}")
    print("\nEvaluation Results:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    # Set device for LPIPS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    main()