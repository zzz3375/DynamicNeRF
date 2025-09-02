import os
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm
import open3d as o3d
from load_llff import load_llff_data
from run_nerf import config_parser, create_nerf
from render_utils import *
from run_nerf_helpers import *


def depth_to_3d_points(depth_map, intrinsic_matrix, extrinsic_matrix=None):
    """
    Convert depth map to 3D points in world coordinates
    
    Args:
        depth_map: [H, W] depth values
        intrinsic_matrix: [3, 3] camera intrinsic matrix
        extrinsic_matrix: [4, 4] camera to world transformation (optional)
    
    Returns:
        points_3d: [H*W, 3] 3D points
    """
    H, W = depth_map.shape
    depth_map = depth_map.reshape(-1)
    
    # Create pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.reshape(-1)
    v = v.reshape(-1)
    
    # Convert to normalized device coordinates
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    # Backproject to 3D camera coordinates
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map
    
    points_camera = np.stack([x, y, z], axis=1)
    
    # Transform to world coordinates if extrinsic matrix is provided
    if extrinsic_matrix is not None:
        # Add homogeneous coordinate
        points_camera_homogeneous = np.concatenate([points_camera, np.ones((len(points_camera), 1))], axis=1)
        points_world_homogeneous = points_camera_homogeneous @ extrinsic_matrix.T
        points_3d = points_world_homogeneous[:, :3] / points_world_homogeneous[:, 3:]
    else:
        points_3d = points_camera
    
    return points_3d


def create_point_cloud_from_render(rgb_map, depth_map, intrinsic_matrix, extrinsic_matrix, 
                                  max_points=1000000, downsample_factor=1):
    """
    Create point cloud from rendered RGB and depth maps
    
    Args:
        rgb_map: [H, W, 3] RGB values
        depth_map: [H, W] depth values
        intrinsic_matrix: [3, 3] camera intrinsic matrix
        extrinsic_matrix: [4, 4] camera to world transformation
        max_points: maximum number of points to keep
        downsample_factor: factor to downsample the point cloud
    
    Returns:
        pcd: open3d point cloud object
    """
    H, W = depth_map.shape[:2]
    
    # Downsample if needed
    if downsample_factor > 1:
        depth_map = depth_map[::downsample_factor, ::downsample_factor]
        rgb_map = rgb_map[::downsample_factor, ::downsample_factor]
        H, W = depth_map.shape[:2]
    
    # Convert depth map to 3D points
    points_3d = depth_to_3d_points(depth_map, intrinsic_matrix, extrinsic_matrix)
    
    # Reshape RGB to match points
    colors = rgb_map.reshape(-1, 3)
    
    # Remove invalid points (infinite or NaN depth)
    valid_mask = np.isfinite(points_3d).all(axis=1) & (depth_map.reshape(-1) > 0)
    points_3d = points_3d[valid_mask]
    colors = colors[valid_mask]
    
    # Limit number of points if too many
    if len(points_3d) > max_points:
        indices = np.random.choice(len(points_3d), max_points, replace=False)
        points_3d = points_3d[indices]
        colors = colors[indices]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize to [0, 1]
    
    return pcd


def load_latest_checkpoint(args, render_kwargs_train):
    ckpt_dir = os.path.join(args.basedir, args.expname)
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.tar')))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    latest_ckpt = ckpts[-1]
    print(f"Loading checkpoint: {latest_ckpt}")
    ckpt = torch.load(latest_ckpt)

    if args.N_importance > 0:
        raise NotImplementedError("Support for fine network not implemented.")
    else:
        render_kwargs_train['network_fn_d'].load_state_dict(ckpt['network_fn_d_state_dict'])
        render_kwargs_train['network_fn_s'].load_state_dict(ckpt['network_fn_s_state_dict'])


def render_and_save_ply(args, output_dir):
    """
    Render depth maps and create PLY files for all training views
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load LLFF data
    images, invdepths, masks, poses, bds, render_poses, render_focals, grids = load_llff_data(
        args, args.datadir, None, frame2dolly=args.frame2dolly,
        recenter=True, bd_factor=.9, spherify=args.spherify)

    hwf = poses[0, :3, -1]
    H, W, focal = map(int, hwf)
    poses = poses[:, :3, :4]
    num_img = poses.shape[0]

    # Create intrinsic matrix
    intrinsic_matrix = np.array([
        [focal, 0, W/2],
        [0, focal, H/2],
        [0, 0, 1]
    ])

    # Create model
    render_kwargs_train, render_kwargs_test, _, _, _ = create_nerf(args)
    render_kwargs_test.update({'near': 0., 'far': 1., 'num_img': num_img})

    # Load latest checkpoint
    load_latest_checkpoint(args, render_kwargs_test)

    # Create combined point cloud for all views
    combined_pcd = o3d.geometry.PointCloud()

    for i in tqdm(range(num_img), desc="Rendering and creating point clouds"):
        pose = torch.Tensor(poses[i]) if not args.camera_still else torch.Tensor(poses[num_img//2])
        time_val = i / float(num_img) * 2. - 1.0

        with torch.no_grad():
            ret = render(
                time_val, False, H, W, focal, chunk=args.chunk,
                c2w=pose, **render_kwargs_test
            )

        # Get RGB and depth maps
        rgb_map = to8b(ret['rgb_map_d'].cpu().numpy())
        depth_map = ret['depth_map_d'].cpu().numpy()

        # Create extrinsic matrix (camera to world transformation)
        pose_np = pose.cpu().numpy()
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :4] = pose_np

        # Create point cloud for this view
        pcd = create_point_cloud_from_render(
            rgb_map, depth_map, intrinsic_matrix, extrinsic_matrix,
            max_points=args.max_points_per_view, downsample_factor=args.downsample_factor
        )

        # Save individual point cloud
        ply_path = os.path.join(output_dir, f"view_{i:03d}.ply")
        o3d.io.write_point_cloud(ply_path, pcd)

        # Add to combined point cloud
        combined_pcd += pcd

        # Also save depth and RGB images for reference
        # depth_image_path = os.path.join(output_dir, f"depth_{i:03d}.png")
        # rgb_image_path = os.path.join(output_dir, f"rgb_{i:03d}.png")
        
        # imageio.imwrite(depth_image_path, to8b(normalize_depth(ret['depth_map_d']).cpu().numpy()))
        # imageio.imwrite(rgb_image_path, rgb_map)

    # Save combined point cloud
    combined_ply_path = os.path.join(output_dir, "combined_pointcloud.ply")
    o3d.io.write_point_cloud(combined_ply_path, combined_pcd)
    
    print(f"Saved combined point cloud with {len(combined_pcd.points)} points to {combined_ply_path}")


if __name__ == '__main__':
    parser = config_parser()
    parser.add_argument('--output_dir', type=str, default='ply_results', required=True, 
                       help="Folder to save PLY files and images")
    parser.add_argument('--max_points_per_view', type=int, default=500000,
                       help="Maximum number of points per view")
    parser.add_argument('--downsample_factor', type=int, default=2,
                       help="Downsampling factor for point cloud generation")
    parser.add_argument('--camera_still', action='store_true', help="UAV suspension")
    args = parser.parse_args()

    # Install open3d if not available
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "open3d"])
        import open3d as o3d

    torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')
    render_and_save_ply(args, args.output_dir)