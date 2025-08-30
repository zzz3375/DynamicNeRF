import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from run_nerf import config_parser, create_nerf
from render_utils import get_rays, ndc_rays, NDC2world, render_rays
from run_nerf_helpers import get_embedder

def main():
    # Parse configuration
    parser = config_parser()
    args = parser.parse_args()
    
    # Override chunk settings for lower memory usage
    args.chunk = 16384
    args.netchunk = 16384
    args.render_only = True
    
    # Load dataset information
    if args.dataset_type == 'llff':
        from load_llff import load_llff_data
        images, invdepths, masks, poses, bds, render_poses, render_focals, grids = load_llff_data(
            args, args.datadir, None, recenter=True, bd_factor=.9, spherify=args.spherify)
        
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        num_img = float(poses.shape[0])
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        
        # Use all views to train
        i_train = np.array([i for i in np.arange(int(images.shape[0]))])
        
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
    
    # Create NeRF model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    
    # Extract only the parameters that render_rays expects
    render_rays_kwargs = {
        'network_fn_d': render_kwargs_test['network_fn_d'],
        'network_fn_s': render_kwargs_test['network_fn_s'],
        'network_query_fn_d': render_kwargs_test['network_query_fn_d'],
        'network_query_fn_s': render_kwargs_test['network_query_fn_s'],
        'N_samples': render_kwargs_test['N_samples'],
        'num_img': num_img,
        'DyNeRF_blending': render_kwargs_test['DyNeRF_blending'],
        'pretrain': False,
        'lindisp': args.lindisp,
        'perturb': render_kwargs_test['perturb'],
        'N_importance': render_kwargs_test['N_importance'],
        'raw_noise_std': render_kwargs_test['raw_noise_std'],
        'inference': render_kwargs_test['inference']
    }
    
    # Load the latest checkpoint
    basedir = args.basedir
    expname = args.expname
    ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]
    
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        print('Loading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        
        # Load model weights
        render_kwargs_train['network_fn_d'].load_state_dict(ckpt['network_fn_d_state_dict'])
        render_kwargs_train['network_fn_s'].load_state_dict(ckpt['network_fn_s_state_dict'])
    
    # Move data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    
    # Select the first frame
    frame_idx = 0
    t = frame_idx / num_img * 2. - 1.0
    target_image = images[frame_idx].cpu().numpy()
    pose = poses[frame_idx, :3, :4]
    
    # Display the first image and select pixels
    cv2.namedWindow("Select Pixels (Press 'q' when done)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Pixels (Press 'q' when done)", 800, 600)
    
    selected_pixels = []
    display_image = target_image.copy()
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Draw a red circle at the selected pixel
            cv2.circle(display_image, (x, y), 2, (1, 0, 0), -1)
            selected_pixels.append((y, x))  # Store as (row, col)
            cv2.imshow("Select Pixels (Press 'q' when done)", display_image)
    
    cv2.setMouseCallback("Select Pixels (Press 'q' when done)", mouse_callback)
    cv2.imshow("Select Pixels (Press 'q' when done)", display_image)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    if not selected_pixels:
        print("No pixels selected. Exiting.")
        return
    
    print(f"Selected {len(selected_pixels)} pixels")
    
    # Get rays for selected pixels
    rays_o, rays_d = get_rays(H, W, focal, pose)
    selected_rays_o = rays_o[tuple(zip(*selected_pixels))].reshape(-1, 3)
    selected_rays_d = rays_d[tuple(zip(*selected_pixels))].reshape(-1, 3)
    
    # Create ray batch
    near_arr = near * torch.ones(selected_rays_o.shape[0], 1).to(device)
    far_arr = far * torch.ones(selected_rays_o.shape[0], 1).to(device)
    
    if args.use_viewdirs:
        viewdirs = selected_rays_d / torch.norm(selected_rays_d, dim=-1, keepdim=True)
        ray_batch = torch.cat([selected_rays_o, selected_rays_d, near_arr, far_arr, viewdirs], -1)
    else:
        ray_batch = torch.cat([selected_rays_o, selected_rays_d, near_arr, far_arr], -1)
    
    # Render the first frame to get initial 3D positions
    with torch.no_grad():
        ret = render_rays(t, False, ray_batch, **render_rays_kwargs)
    
    # Get the 3D points using the weights and sample positions
    N_rays = len(selected_pixels)
    N_samples = args.N_samples
    
    # Create sample positions along the ray
    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    if not args.lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    
    z_vals = z_vals.expand([N_rays, N_samples])
    
    # Points in space
    pts = selected_rays_o[..., None, :] + selected_rays_d[..., None, :] * z_vals[..., :, None]
    
    # Get the weighted average 3D position for each ray
    weights = ret['weights_full']
    pts_3d = torch.sum(weights[..., None] * pts, -2)
    
    # Convert from NDC to world coordinates if needed
    if render_kwargs_test['ndc']:
        pts_3d_world = NDC2world(pts_3d, H, W, focal)
    else:
        pts_3d_world = pts_3d
    
    # Initialize arrays to store scene flow and displacements
    N_p = len(selected_pixels)
    N_steps = int(num_img) - 1  # Number of consecutive frames to process
    
    scene_flow = np.zeros((N_p, N_steps, 3))
    displacement = np.zeros((N_p, N_steps + 1, 3))  # +1 for initial position
    
    # Store initial positions
    displacement[:, 0, :] = pts_3d_world.cpu().numpy()
    
    # Process consecutive frames
    for step in tqdm(range(N_steps), desc="Processing frames"):
        frame_idx = step + 1
        t = frame_idx / num_img * 2. - 1.0
        pose = poses[frame_idx, :3, :4]
        
        # Get rays for the same pixels in the new frame
        rays_o, rays_d = get_rays(H, W, focal, pose)
        current_rays_o = rays_o[tuple(zip(*selected_pixels))].reshape(-1, 3)
        current_rays_d = rays_d[tuple(zip(*selected_pixels))].reshape(-1, 3)
        
        # Create ray batch
        if args.use_viewdirs:
            viewdirs = current_rays_d / torch.norm(current_rays_d, dim=-1, keepdim=True)
            current_ray_batch = torch.cat([current_rays_o, current_rays_d, near_arr, far_arr, viewdirs], -1)
        else:
            current_ray_batch = torch.cat([current_rays_o, current_rays_d, near_arr, far_arr], -1)
        
        # Render the current frame
        with torch.no_grad():
            ret_current = render_rays(t, False, current_ray_batch, **render_rays_kwargs)
        
        # Get scene flow from the model (forward flow)
        sceneflow_f = ret_current['sceneflow_f']
        
        # Average the scene flow across samples using weights
        avg_sceneflow = torch.sum(ret_current['weights_full'][..., None] * sceneflow_f, -2)
        
        # Store scene flow
        scene_flow[:, step, :] = avg_sceneflow.cpu().numpy()
        
        # Calculate displacement
        if step == 0:
            displacement[:, step + 1, :] = displacement[:, step, :] + scene_flow[:, step, :]
        else:
            displacement[:, step + 1, :] = displacement[:, step, :] + scene_flow[:, step, :]
    
    # Create output directory
    dataset_name = os.path.basename(args.datadir.rstrip('/'))
    output_dir = f"./scene_flow_analysis/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save displacement data to CSV files
    time_steps = np.arange(N_steps + 1) / 30.0  # Assuming 30 FPS
    
    for dim, dim_name in enumerate(['x', 'y', 'z']):
        df_data = {'Time (s)': time_steps}
        
        for i in range(N_p):
            df_data[f'Sensor {i+1}'] = displacement[i, :, dim]
        
        df = pd.DataFrame(df_data)
        df.to_csv(f"{output_dir}/displacement_{dim_name}.csv", index=False)
    
    # Plot displacement for each sensor
    plt.figure(figsize=(12, 8))
    
    for i in range(N_p):
        plt.plot(time_steps, displacement[i, :, 0], label=f'Sensor {i+1} X')
        plt.plot(time_steps, displacement[i, :, 1], label=f'Sensor {i+1} Y')
        plt.plot(time_steps, displacement[i, :, 2], label=f'Sensor {i+1} Z')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement')
    plt.title('Displacement over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/displacement_plot.png")
    plt.show()
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main()