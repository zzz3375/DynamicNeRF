import os
import glob
import torch
import argparse
import imageio
import numpy as np
from tqdm import tqdm

from load_llff import load_llff_data
from run_nerf import config_parser, create_nerf

from render_utils import *
from run_nerf_helpers import *


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


def render_all_training_views(args, output_dir, render_mode='color', make_video=False):
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load LLFF data
    images, invdepths, masks, poses, bds, render_poses, render_focals, grids = load_llff_data(
        args, args.datadir, None, frame2dolly=args.frame2dolly,
        recenter=True, bd_factor=.9, spherify=args.spherify)

    hwf = poses[0, :3, -1]
    H, W, focal = map(int, hwf)
    poses = poses[:, :3, :4]
    num_img = poses.shape[0]

    # Create model
    render_kwargs_train, render_kwargs_test, _, _, _ = create_nerf(args)
    render_kwargs_test.update({'near': 0., 'far': 1., 'num_img': num_img})

    # Load latest checkpoint
    # load_latest_checkpoint(args, render_kwargs_train)
    del render_kwargs_train
    load_latest_checkpoint(args, render_kwargs_test)  # usually the same networks are shared

    frames = []

    for i in tqdm(range(num_img), desc="Rendering training views"):
        pose = torch.Tensor(poses[i])
        time_val = i / float(num_img) * 2. - 1.0

        with torch.no_grad():
            ret = render(
                time_val, False, H, W, focal, chunk=args.chunk,
                c2w=pose, **render_kwargs_test
            )

        if render_mode == 'color':
            image = to8b(ret['rgb_map_full'].cpu().numpy())
        elif render_mode == 'depth':
            image = to8b(normalize_depth(ret['depth_map_full']).cpu().numpy())
        else:
            raise ValueError("render_mode must be 'color' or 'depth'")

        fname = os.path.join(output_dir, f"{i:03d}.png")
        imageio.imwrite(fname, image)
        frames.append(image)

    if make_video:
        import cv2
        import os

        # 设置参数
        image_folder = output_dir # 图片文件夹路径
        video_name = 'output_video.mp4'  # 输出视频文件名
        fps = 30  # 帧率（每秒多少张图片）

        # 获取所有图片并按文件名排序
        images = [img for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png', '.jpeg'))]
        images.sort()  # 确保图片按顺序排列

        # 读取第一张图片，获取尺寸
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        # 创建视频写入对象
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # 逐帧写入视频
        for image in images:
            img_path = os.path.join(image_folder, image)
            frame = cv2.imread(img_path)
            video.write(frame)

        # 释放资源
        video.release()
        print(f"视频已生成: {video_name}")


if __name__ == '__main__':
    parser = config_parser()
    parser.add_argument('--output_dir', type=str, default='render_results', required=True, help="Folder to save images and video")
    parser.add_argument('--render_mode', choices=['color', 'depth'], default='depth', help="Which map to render")
    parser.add_argument('--make_video', action='store_true', help="Combine images into a video")
    args = parser.parse_args()

    torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')
    render_all_training_views(args, args.output_dir, args.render_mode, args.make_video)
    #  python render_results.py --output_dir render_results/plain --render_mode depth --make_video --config configs/config-WTB-inservice.txt --chunk 16384
