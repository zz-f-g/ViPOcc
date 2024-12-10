import argparse
import sys

sys.path.append(".")
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset


def main():
    parser = argparse.ArgumentParser("KITTI 360 Preprocessing")
    parser.add_argument("--data_path", default='data/KITTI-360')
    parser.add_argument("--save_png", default=False, action='store_true')
    args = parser.parse_args()

    data_path = Path(args.data_path)

    print("Setting up dataset")
    dataset = Kitti360Dataset(
        data_path=data_path,
        pose_path=data_path / "data_poses",
        split_path=None,
        return_stereo=True,
        frame_count=1,
    )
    print("Setting up folders...")

    for i in tqdm(range(len(dataset))):
        sequence, id, is_right = dataset._datapoints[i]
        img_id = dataset._img_ids[sequence][id]
        save_path = data_path / 'depth_gt' / sequence / f"{img_id:010d}"
        if is_right or (save_path.with_suffix('.npy').exists() and save_path.with_suffix('.png').exists()):
            continue
        save_path.parent.mkdir(exist_ok=True, parents=True)
        depth = dataset.load_depth(sequence, img_id, is_right)[0].astype(np.float32)  # [192, 640]
        np.save(save_path.with_suffix('.npy'), depth)

        if args.save_png:
            # depth_img = (depth - depth.min()) / (depth.max() - depth.min())
            depth_img = (depth / 80.0 * 255.0).astype(np.uint8)
            cv2.imwrite(str(save_path.with_suffix('.png')), depth_img)


if __name__ == "__main__":
    main()
