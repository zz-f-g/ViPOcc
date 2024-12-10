import argparse
import json
import os.path
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import multivariate_normal


def get_sample_probabilities(areas, background_sample_ratio):
    areas = np.array(areas)
    normalized_areas = np.log(areas)
    normalized_areas = normalized_areas / normalized_areas.sum()
    sample_probabilities = np.append((1 - background_sample_ratio) * normalized_areas, background_sample_ratio)
    return sample_probabilities


class GaussianSampling:
    def __init__(self, box, img_size, patch_size):
        self.img_size = img_size
        self.patch_size = patch_size
        self.half_ps = patch_size // 2

        x1, y1, x2, y2 = box
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.width = x2 - x1
        self.height = y2 - y1
        self.cov = np.array([[self.width ** 2 / 16, 0], [0, self.height ** 2 / 16]])  # 95.45% samples in the rectangle

    def gen_sample(self):
        h, w = self.img_size
        cnt = 0
        while True:
            cnt += 1
            sample = np.random.multivariate_normal(mean=self.center, cov=self.cov, size=1)
            sample = np.round(sample).astype(np.int32)
            if (0 + self.half_ps < sample[0, 0] < w - self.half_ps and
                    0 + self.half_ps < sample[0, 1] < h - self.half_ps):
                return sample
            elif cnt > 10:  # If the sampler cannot be collected within 10 attempts, return None
                return None

    def get_pdf(self):
        u, v = np.meshgrid(np.arange(self.img_size[0]), np.arange(self.img_size[1]), indexing='ij')
        pos = np.dstack((u, v)).astype(np.float64)
        center = (self.center[1], self.center[0])
        cov = np.array([[self.cov[1, 1], 0], [0, self.cov[0, 0]]])
        pdf = multivariate_normal.pdf(pos, mean=center, cov=cov)
        return pdf


class UniformSampling:
    def __init__(self, img_size, patch_size):
        self.img_size = img_size
        self.patch_size = patch_size
        self.half_ps = patch_size // 2

    def gen_sample(self):
        h, w = self.img_size
        x = np.random.randint(low=self.half_ps, high=w - self.half_ps, size=(1,))
        y = np.random.randint(low=self.half_ps, high=h - self.half_ps, size=(1,))
        return np.hstack((x, y))

    def get_pdf(self):
        h, w = self.img_size
        pdf = 1 / (h * w) * np.ones((h, w), dtype=np.float32)
        return pdf


def gen_snog_samples(json,
                     img_size,
                     patch_size,
                     patch_num,
                     background_sample_ratio,
                     gaussian_sample_phrases,
                     ):
    samples = []  # list of np.array([u,v]), where [u,v] is coordinates of the center anchor point.
    # phase 0: initialize samplers and probabilities
    gaussian_sampler_list = []
    areas = []

    for item in json:
        if item['label'] in gaussian_sample_phrases:
            gs_sampler = GaussianSampling(item['box'], img_size=img_size, patch_size=patch_size)
            gaussian_sampler_list.append(gs_sampler)
            areas.append(item['pixel_num'])

    uni_sampler = UniformSampling(img_size, patch_size)
    sampler_list = gaussian_sampler_list + [uni_sampler]
    prob_list = get_sample_probabilities(areas, background_sample_ratio)

    if len(gaussian_sampler_list) == 0 or len(json) == 0:
        prob_list = [1]

    # phase 1: choose sampler
    sampler_choices = np.random.choice(np.arange(len(sampler_list)), size=patch_num, replace=True, p=prob_list)

    # phase 2: choose samples from sampler
    cnt = 0
    for i in sampler_choices:
        while True:
            cnt += 1
            if cnt > 10:
                sample = sampler_list[-1].gen_sample()
            else:
                sample = sampler_list[i].gen_sample()
                if sample is None:  # If the sample is None, collect a random sample
                    sample = sampler_list[-1].gen_sample()
            # print(sample)
            if all(np.linalg.norm(sample - old_sample) >= np.sqrt(2) * patch_size for old_sample in samples):
                samples.append(sample)
                cnt = 0
                break

    all_samples = np.vstack(samples)
    return all_samples


def draw(anchors, mask, patch_size, patch_num, alpha, mask_color, output_path):
    patch_mask = np.zeros_like(mask)
    half_ps = patch_size // 2
    h, w, _ = mask.shape
    for i in range(patch_num):
        v, u = anchors[i]
        patch_mask[u - half_ps:u + half_ps + 1, v - half_ps:v + half_ps + 1] = 1

    anchor_rgb = np.zeros((h, w, 3))
    for i in range(patch_num):
        u, v = anchors[i]
        u -= half_ps
        v -= half_ps
        cv2.rectangle(anchor_rgb, (u, v), (u + patch_size, v + patch_size), color=mask_color, thickness=-1)

    masked_new = mask * (1 - patch_mask * alpha) + anchor_rgb * alpha
    cv2.imwrite(str(output_path), masked_new)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_path',
                        # default='data/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_192x640/0000000001.png',
                        default='data/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_192x640/0000000221.png',
                        # default='data/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_192x640/0000001144.png',
                        # default='data/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_192x640/0000001479.png',
                        # default='data/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_192x640/0000001534.png',
                        # default='data/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_192x640/0000001548.png',
                        # default='data/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_192x640/0000002714.png',
                        )
    parser.add_argument('--save_dir',
                        default='visualization/sampling_anchors',
                        type=str, help='directory to save')

    parser.add_argument('--background_sample_ratio', type=float, default=0.5)
    parser.add_argument('--mask_color', type=str, default=[255, 255, 0], help='color of the patch')
    parser.add_argument('--alpha', type=float, default=0.7, help='transparency of the patch')
    parser.add_argument('--patch_size', type=int, default=8, help='size of the patch')
    parser.add_argument('--patch_num', type=int, default=64, help='number of patches')
    parser.add_argument('--gaussian_sample_phrases', default=['car', 'person'])
    args = parser.parse_args()

    rgb_path = args.rgb_path
    mask_path = Path(rgb_path.replace('data_2d_raw', 'samples'))
    json_path = Path(rgb_path.replace('data_2d_raw', 'samples').replace('png', 'json'))

    json = json.load(open(json_path, 'r'))
    mask = cv2.imread(str(mask_path), -1)

    os.makedirs(args.save_dir, exist_ok=True)

    kwargs = {
        'img_size': mask.shape[:2],
        'patch_size': args.patch_size,
        'patch_num': args.patch_num,
        'background_sample_ratio': args.background_sample_ratio,
        'gaussian_sample_phrases': ['car', 'person'],
    }

    # ========== snog sampler visualization ==========
    snog_samples = gen_snog_samples(json, **kwargs)
    vis_args = {
        'patch_size': args.patch_size,
        'patch_num': args.patch_num,
        'mask_color': args.mask_color,
        'alpha': args.alpha,
        'output_path': Path(args.save_dir) / (mask_path.stem + '_snog_sampler.png')
    }
    draw(snog_samples, mask, **vis_args)

    # ========== random sampler visualization ==========
    random_samples = np.hstack([
        np.random.randint(low=0, high=mask.shape[1] - args.patch_size, size=(args.patch_num, 1)),
        np.random.randint(low=0, high=mask.shape[0] - args.patch_size, size=(args.patch_num, 1))]
    )
    vis_args = {
        'patch_size': args.patch_size,
        'patch_num': args.patch_num,
        'mask_color': args.mask_color,
        'alpha': args.alpha,
        'output_path': Path(args.save_dir) / (mask_path.stem + '_random_sampler.png')
    }
    draw(random_samples, mask, **vis_args)

    # save rgb
    cv2.imwrite(os.path.join(args.save_dir, mask_path.name), cv2.imread(rgb_path, -1))
