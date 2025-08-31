# ./analysis_scene_flow.py
import os
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from run_nerf import config_parser, create_nerf, induce_flow
from run_nerf_helpers import get_rays
from render_utils import render
from render_results import load_latest_checkpoint


def main():
    parser = config_parser()
    parser.set_defaults(chunk=16384, netchunk=16384)
    parser.add_argument("--output_dir", type=str, default="scene_flow_analysis", help="output folder")
    args = parser.parse_args()

    save_dir = os.path.join(args.output_dir, os.path.basename(args.datadir))
    os.makedirs(save_dir, exist_ok=True)
    
    # Create images subfolder
    images_dir = os.path.join(save_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    torch.set_default_tensor_type("torch.cuda.FloatTensor" if torch.cuda.is_available() else "torch.FloatTensor")

    # Create model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    load_latest_checkpoint(args, render_kwargs_test)

    # Load dataset (like in render_results)
    from load_llff import load_llff_data
    images, invdepths, masks, poses, bds, render_poses, render_focals, grids = load_llff_data(
        args, args.datadir, None, frame2dolly=args.frame2dolly,
        recenter=True, bd_factor=.9, spherify=args.spherify)

    hwf = poses[0, :3, -1]
    H, W, focal = map(int, hwf)
    poses = poses[:, :3, :4]
    num_img = poses.shape[0]
    render_kwargs_test.update({'near': 0., 'far': 1., 'num_img': num_img})
    
    # ---- Step 1: show first frame ----
    first_pose = torch.Tensor(poses[0])
    t0 = 0.0 / float(num_img) * 2.0 - 1.0
    with torch.no_grad():
        ret = render(t0, False, H, W, focal, chunk=args.chunk, c2w=first_pose, **render_kwargs_test)

    
    first_img = (ret['rgb_map_d'].cpu().numpy() * 255).astype(np.uint8)
    current_img = first_img
    cv2.imshow("Select Pixels", first_img[:, :, ::-1])  # BGR
    selected_pixels = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_pixels.append((x, y))
            cv2.circle(first_img, (x, y), 2, (0, 0, 255), -1)
            cv2.imshow("Select Pixels", first_img[:, :, ::-1])

    cv2.setMouseCallback("Select Pixels", click_event)
    print("Click pixels to select (press any key when done)...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    N_p = len(selected_pixels)
    print(f"Selected {N_p} pixels")

    # ---- Step 2: compute 3D coords for first frame ----
    rays_o, rays_d = get_rays(H, W, focal, first_pose)
    rays_o, rays_d = rays_o.cpu().numpy(), rays_d.cpu().numpy()

    depth_map = ret["depth_map_d"].cpu().numpy()
    pt_3d = []
    for (x, y) in selected_pixels:
        d = depth_map[y, x]
        ro = rays_o[y, x]
        rd = rays_d[y, x]
        pt = ro + d * rd
        pt_3d.append(pt)
    pt_3d = np.array(pt_3d)  # [N_p, 3]

    # ---- Step 3: trace scene flow across frames ----
    scene_flows = np.zeros((N_p, num_img, 3))
    displacements = np.zeros((N_p, num_img, 3))

    prev_pts = pt_3d.copy()
    
    # # Save first frame with points
    # first_img_with_points = first_img.copy()
    # for (x, y) in selected_pixels:
    #     cv2.circle(first_img_with_points, (x, y), 3, (0, 0, 255), -1)  # Red dot
    # cv2.imwrite(os.path.join(images_dir, f"frame_000.png"), first_img_with_points[:, :, ::-1])
    
    for i in tqdm(range(0, num_img - 1), desc="Tracing scene flow"):

        
        pose = torch.Tensor(poses[i])
        grid = grids[i]
        t_val = i / float(num_img) * 2.0 - 1.0
        with torch.no_grad():
            ret = render(t_val, False, H, W, focal, chunk=args.chunk, c2w=pose, **render_kwargs_test)

        current_img = (ret['rgb_map_d'].cpu().numpy() * 255).astype(np.uint8)
        for (x, y) in selected_pixels:
            cv2.circle(current_img, (x, y), 3, (0, 0, 255), -1)  # Red dot
        cv2.imwrite(os.path.join(images_dir, f"frame_{i:03d}.png"), current_img[:, :, ::-1])

        depth_map = ret["depth_map_d"].cpu().numpy()
        rays_o, rays_d = get_rays(H, W, focal, pose)
        rays_o, rays_d = rays_o.cpu().numpy(), rays_d.cpu().numpy()

        # update selected pixels
        pose_f = poses[min(i + 1, num_img - 1), :3, :4]
        induced_flow_f = induce_flow(
            H, W, focal,
            torch.Tensor(pose_f),
            ret['weights_d'],
            ret['raw_pts_f'],
            torch.Tensor(grid[..., :2])
        ).cpu().numpy()  
        new_pixels = []
        for (x, y) in selected_pixels:
            flow = induced_flow_f[y, x]
            new_x = int(np.clip(x + flow[0], 0, W - 1))
            new_y = int(np.clip(y + flow[1], 0, H - 1))
            new_pixels.append((new_x, new_y))
        selected_pixels = new_pixels  
        
        # calculate 3D points
        cur_pts = []
        for (x, y) in selected_pixels:
            d = depth_map[y, x]
            cur_pts.append(rays_o[y, x] + d * rays_d[y, x])
        cur_pts = np.array(cur_pts)

        scene_flows[:, i - 1, :] = cur_pts - prev_pts
        displacements[:, i, :] = displacements[:, i - 1, :] + (cur_pts - prev_pts)
        prev_pts = cur_pts
        
        # Draw and save current frame with tracked points



    # ---- Step 4: plot displacement ----
    time_axis = np.arange(num_img) / 30.0  # seconds
    for dim, label in enumerate(["x", "y", "z"]):
        plt.figure()
        for p in range(N_p):
            plt.plot(time_axis, displacements[p, :, dim], label=f"Sensor {p+1}")
        plt.xlabel("Time (s)")
        plt.ylabel(f"Displacement {label}")
        plt.legend()
        plt.title(f"Displacement along {label}")
        plt.savefig(os.path.join(save_dir, f"displacement_{label}.png"))
        plt.close()

        # Save to CSV
        df = pd.DataFrame(displacements[:, :, dim].T, columns=[f"Sensor {i+1}" for i in range(N_p)])
        df.insert(0, "Time (s)", time_axis)

        df.to_csv(os.path.join(save_dir, f"displacement_{label}.csv"), index=False)


if __name__ == "__main__":
    main()