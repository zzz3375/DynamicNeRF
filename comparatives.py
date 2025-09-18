"""
a python script that evaluate masked PSNR, SSIM, LPIPS
input: result_dir, gt_dir, mask_dir
output: (pd datafame 1*6) overall PSNR,overall SSIM, overall LPIPS, masked PSNR, masked SSIM, masked LPIPS
save to xlsx
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings('ignore')

class ImageQualityEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
    
    def load_image(self, path):
        """Load image and convert to numpy array in range [0, 255]"""
        img = Image.open(path).convert('RGB')
        return np.array(img)
    
    def load_mask(self, path):
        """Load mask and convert to binary mask"""
        mask = Image.open(path).convert('L')
        mask = np.array(mask)
        # Convert to binary mask (0 and 1)
        mask = (mask > 128).astype(np.float32)
        return mask
    
    def calculate_overall_psnr(self, img1, img2):
        """Calculate PSNR for entire image"""
        return psnr(img1, img2, data_range=255)
    
    def calculate_overall_ssim(self, img1, img2):
        """Calculate SSIM for entire image"""
        # SSIM requires single channel or multi-channel specification
        return ssim(img1, img2, data_range=255, channel_axis=2, win_size=7)
    
    def calculate_overall_lpips(self, img1, img2):
        """Calculate LPIPS for entire image"""
        # Convert to torch tensors
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        
        img1_tensor = img1_tensor.to(self.device)
        img2_tensor = img2_tensor.to(self.device)
        
        with torch.no_grad():
            lpips_score = self.lpips_model(img1_tensor, img2_tensor)
        
        return lpips_score.item()
    
    def calculate_masked_psnr(self, img1, img2, mask):
        """Calculate PSNR only in masked region"""
        # Apply mask
        masked_img1 = img1 * mask[..., np.newaxis]
        masked_img2 = img2 * mask[..., np.newaxis]
        
        # Get only the masked pixels
        mask_flat = mask.flatten().astype(bool)
        img1_flat = masked_img1.reshape(-1, 3)[mask_flat]
        img2_flat = masked_img2.reshape(-1, 3)[mask_flat]
        
        if len(img1_flat) == 0:
            return 0  # No masked pixels
        
        return psnr(img1_flat, img2_flat, data_range=255)
    
    def calculate_masked_ssim(self, img1, img2, mask):
        """Calculate SSIM only in masked region"""
        # Create a bounding box around the mask to avoid processing entire image
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return 0  # No masked pixels
            
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        
        # Crop images and mask to the bounding box
        img1_crop = img1[y_min:y_max, x_min:x_max]
        img2_crop = img2[y_min:y_max, x_min:x_max]
        mask_crop = mask[y_min:y_max, x_min:x_max]
        
        # Calculate SSIM on the cropped region
        ssim_val = ssim(img1_crop, img2_crop, data_range=255, channel_axis=2, win_size=7)
        
        return ssim_val
    
    def calculate_masked_lpips(self, img1, img2, mask):
        """Calculate LPIPS only in masked region"""
        # Convert mask to tensor and apply to images
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        mask_tensor = mask_tensor.to(self.device)
        
        # Convert images to tensors
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        
        img1_tensor = img1_tensor.to(self.device)
        img2_tensor = img2_tensor.to(self.device)
        
        # Apply mask to feature maps (approximate approach)
        with torch.no_grad():
            # Get features from LPIPS model
            feat1 = self.lpips_model.net.forward_once(img1_tensor)
            feat2 = self.lpips_model.net.forward_once(img2_tensor)
            
            # Resize mask to match feature map size
            current_mask = mask_tensor
            masked_dists = []
            
            for f1, f2 in zip(feat1, feat2):
                # Resize mask to current feature map size
                if current_mask.shape[2:] != f1.shape[2:]:
                    current_mask = torch.nn.functional.interpolate(
                        current_mask, size=f1.shape[2:], mode='nearest'
                    )
                
                # Apply mask and calculate distance
                f1_masked = f1 * current_mask
                f2_masked = f2 * current_mask
                
                dist = (f1_masked - f2_masked) ** 2
                masked_dists.append(dist.mean())
            
            # Weighted average as in original LPIPS
            lpips_score = sum(masked_dists) / len(masked_dists)
        
        return lpips_score.item()
    
    def evaluate_single_pair(self, result_path, gt_path, mask_path):
        """Evaluate a single image pair"""
        # Load images and mask
        result_img = self.load_image(result_path)
        gt_img = self.load_image(gt_path)
        mask = self.load_mask(mask_path)
        
        # Calculate metrics
        metrics = {}
        
        # Overall metrics
        metrics['overall_psnr'] = self.calculate_overall_psnr(result_img, gt_img)
        metrics['overall_ssim'] = self.calculate_overall_ssim(result_img, gt_img)
        metrics['overall_lpips'] = self.calculate_overall_lpips(result_img, gt_img)
        
        # Masked metrics
        metrics['masked_psnr'] = self.calculate_masked_psnr(result_img, gt_img, mask)
        metrics['masked_ssim'] = self.calculate_masked_ssim(result_img, gt_img, mask)
        metrics['masked_lpips'] = self.calculate_masked_lpips(result_img, gt_img, mask)
        
        return metrics
    
    def evaluate_directory(self, result_dir, gt_dir, mask_dir):
        """Evaluate all images in directories"""
        # Get all image files
        result_files = sorted([f for f in os.listdir(result_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Ensure files match
        assert len(result_files) == len(gt_files) == len(mask_files), "Number of files in directories don't match"
        
        all_metrics = []
        
        for result_file, gt_file, mask_file in zip(result_files, gt_files, mask_files):
            result_path = os.path.join(result_dir, result_file)
            gt_path = os.path.join(gt_dir, gt_file)
            mask_path = os.path.join(mask_dir, mask_file)
            
            try:
                metrics = self.evaluate_single_pair(result_path, gt_path, mask_path)
                all_metrics.append(metrics)
                print(f"Processed: {result_file}")
            except Exception as e:
                print(f"Error processing {result_file}: {e}")
                continue
        
        # Create DataFrame and compute averages
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            avg_metrics = df.mean().to_dict()
            
            # Create final 1x6 DataFrame
            final_df = pd.DataFrame([avg_metrics], columns=[
                'overall_psnr', 'overall_ssim', 'overall_lpips',
                'masked_psnr', 'masked_ssim', 'masked_lpips'
            ])
            
            return final_df
        else:
            raise ValueError("No valid images were processed")

def main(result_dir, gt_dir, mask_dir):
    """Main function to run evaluation"""
    evaluator = ImageQualityEvaluator()
    results_df = evaluator.evaluate_directory(result_dir, gt_dir, mask_dir)
    return results_df

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual directories
    result_directory = "path/to/result/images"
    gt_directory = "path/to/ground/truth/images"
    mask_directory = "path/to/mask/images"
    
    results = main(result_directory, gt_directory, mask_directory)
    print("\nFinal Results:")
    print(results)
    print(f"\nAverage Metrics:")
    for col, value in results.iloc[0].items():
        print(f"{col}: {value:.4f}")