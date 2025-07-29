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
    """
    Generating multi view multi time data using cv2 for image I/O
    """

    Maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device).eval()
    threshold = 0.5

    videoname, ext = os.path.splitext(os.path.basename(args.videopath))

    cap = cv2.VideoCapture(args.videopath)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # step = max(1, int(np.ceil(frame_count / 385)))
    step = 1
    total_frames = 10000

    create_dir(os.path.join(args.data_dir, args.outputname, 'images'))
    create_dir(os.path.join(args.data_dir, args.outputname, 'images_colmap'))
    create_dir(os.path.join(args.data_dir, args.outputname, 'background_mask'))

    idx = 0
    frame_idx = 0
    idx_cut = 0
    while True:
        ret, frame = cap.read()
        # frame = frame[100:,:,:]
        if frame_idx < idx_cut:
            frame_idx += 1
            continue
        if not ret:
            break
        if idx >= total_frames:
            break
        if frame_idx % step == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # img = frame
            H, W, _ = img.shape
            # max_size = int(1920 / 2)
            # max_size = 1600
            # scale = 1.0
            # if scale < 1.0:
            #     img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
            #     H, W, _ = img.shape

            print(idx)
            # Save images using cv2 (convert RGB to BGR)
            cv2.imwrite(os.path.join(args.data_dir, args.outputname, 'images', str(idx).zfill(3) + '.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(args.data_dir, args.outputname, 'images_colmap', str(idx).zfill(3) + '.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Get coarse background mask
            img_tensor = torchvision.transforms.functional.to_tensor(img).to(device)
            background_mask = torch.ones(H, W, dtype=torch.float32, device=device)
            # objPredictions = Maskrcnn([img_tensor])[0]

            # for intMask in range(len(objPredictions['masks'])):
            #     if objPredictions['scores'][intMask].item() > threshold:
            #         if objPredictions['labels'][intMask].item() == 1:  # person
            #             background_mask[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

            background_mask_np = ((background_mask.cpu().numpy() > 0.1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.data_dir, args.outputname, 'background_mask', str(idx).zfill(3) + '.jpg.png'), background_mask_np)
            idx += 1
        frame_idx += 1

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
