import argparse
import cv2
import glob
import numpy as np
import os
import torch
from tqdm import tqdm

import sys
sys.path.append('./')
from depth_anything_v2.dpt import DepthAnythingV2

"""
Generating pseudo depth for KITTI 360 dataset. 
Only the depths of perspective images (image_00, image_01) will be generated.
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')

    parser.add_argument('--img_dir', type=str, default='data/KITTI-360/data_2d_raw')
    parser.add_argument('--encoder', type=str, default='vitl')
    parser.add_argument('--load-from', type=str,
                        default='data_preprocess/pseudo_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=80)
    args = parser.parse_args()

    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to('cuda').eval()

    filenames = glob.glob(os.path.join(args.img_dir, '2013_05_28_drive_00*_sync/image_0[01]/data_192x640/*.png'))
    filenames = sorted(filenames)

    for filename in tqdm(filenames):
        output_path_png = filename.replace('data_2d_raw', 'depth')
        output_path_npy = output_path_png.replace('.png', '.npy')
        os.makedirs(os.path.dirname(output_path_png), exist_ok=True)
        if os.path.exists(output_path_png) and os.path.exists(output_path_npy):
            continue
        raw_image = cv2.imread(filename)
        depth = depth_anything.infer_image(raw_image)

        # save png
        depth_png = depth / 80.0 * 255.0
        depth_png = depth_png.astype(np.uint8)
        cv2.imwrite(output_path_png, depth_png)

        # save npy
        np.save(output_path_npy, depth)


