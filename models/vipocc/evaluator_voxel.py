import math
import os

import numpy as np
import torch
from torch import Tensor
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torch import nn
from torch.utils.data import DataLoader
from datasets.data_util import make_test_dataset
from models.common.render import NeRFRenderer
from models.vipocc.model.models_bts import BTSNet
from models.vipocc.model.ray_sampler import ImageRaySampler
from utils.base_evaluator import base_evaluation
from utils.metrics import MeanMetric
from utils.bbox import Bbox, bbox3d_collate_fn
from datasets.kitti_360.voxel import (
    VOXEL_ORIGIN,
    VOXEL_SIZE,
    VOXEL_RESOLUTION,
    Visualizer,
)
from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset

EPS = 1e-4


def process_unlabeled_voxel_under_ground(
    voxel: Tensor,
    unlabeled: int = 0,
    invalid: int = 255,
):
    """
    a straight-forward approach to deal with issue
    https://github.com/ai4ce/SSCBench/issues/4
    """
    assert voxel.shape == VOXEL_RESOLUTION
    X, Y, Z = voxel.shape

    valid_mask = (voxel != unlabeled) & (voxel != invalid)

    z_indices = torch.arange(Z, device=voxel.device).view(1, 1, -1).expand(X, Y, Z)
    surface_z = torch.where(valid_mask, z_indices, Z).amin(dim=-1)

    below_surface_mask = z_indices < surface_z.unsqueeze(-1)
    void_mask = (voxel == 0) & below_surface_mask

    return torch.where(void_mask, 255, voxel)


def compute_occlusion(scene: Tensor, observer: tuple[float, float, float]):
    device = scene.device
    X, Y, Z = scene.shape

    x, y, z = torch.meshgrid(
        torch.arange(X, device=device),
        torch.arange(Y, device=device),
        torch.arange(Z, device=device),
        indexing="ij",
    )

    dir_x = x - observer[0]
    dir_y = y - observer[1]
    dir_z = z - observer[2]

    max_steps = torch.max(torch.abs(torch.stack([dir_x, dir_y, dir_z])))

    norm = torch.sqrt(dir_x**2 + dir_y**2 + dir_z**2 + 1e-6)
    dir_x, dir_y, dir_z = dir_x / norm, dir_y / norm, dir_z / norm

    t = torch.arange(0, max_steps, device=device).float().view(-1, 1, 1, 1)

    sample_x = (observer[0] + t * dir_x).long().clamp(0, Z - 1)
    sample_y = (observer[1] + t * dir_y).long().clamp(0, Y - 1)
    sample_z = (observer[2] + t * dir_z).long().clamp(0, X - 1)

    occupancy_along_ray = scene[sample_z, sample_y, sample_x]
    blocked = occupancy_along_ray.cumsum(dim=0) > 1
    return blocked.any(dim=0)

def compute_occ_scores(
    is_occupied_pred: Tensor,
    is_occupied: Tensor,
    is_visible: Tensor,
    is_valid: Tensor,
):
    return (
        (is_occupied_pred == is_occupied)[is_valid].float().mean().item(),
        (is_occupied_pred == is_occupied)[(~is_visible) & is_valid]
        .float()
        .mean()
        .item(),
        (~is_occupied_pred)[(~is_occupied) & (~is_visible) & is_valid]
        .float()
        .mean()
        .item(),
    )


class BTSWrapper(nn.Module):
    def __init__(self, renderer, config) -> None:
        super().__init__()

        self.renderer = renderer

        self.z_near = config["z_near"]
        self.z_far = config["z_far"]
        self.query_batch_size = config.get("query_batch_size", 50000)
        self.occ_threshold = 0.5

        X, Y, Z = VOXEL_RESOLUTION
        vox_coords = torch.stack(
            torch.meshgrid(
                torch.arange(X), torch.arange(Y), torch.arange(Z), indexing="ij"
            ),
            dim=0,
        ).view(3, -1)
        points_velo = (
            (vox_coords + 0.5) * VOXEL_SIZE
            + torch.tensor(VOXEL_ORIGIN).view(3, 1)
        ).to(torch.float32)
        self.points_velo_h = torch.cat(
            [points_velo, torch.ones_like(points_velo[:1, ...])]
        ).to(
            "cuda"
        )  # [4, XYZ]
        self.sampler = ImageRaySampler(self.z_near, self.z_far, channels=3)
        self.count = 0

        # for debug visualization
        calib = Kitti360Dataset._load_calibs("/mnt/disk/KITTI-360/", (0, -15))
        self.visualizer = Visualizer(
            calib["im_size"],
            VOXEL_ORIGIN,
            VOXEL_SIZE,
            calib["K_perspective"],
            calib["T_velo_to_cam"]["00"],
            np.array(
                [
                    [1.0000000, 0.0000000, 0.0000000, 0],
                    [0.0000000, 0.9961947, -0.0871557, 0],
                    [0.0000000, 0.0871557, 0.9961947, 0],
                    [0.0000000, 000000000, 0.0000000, 1],
                ],
            ),
        )

    def forward(self, data):
        data = dict(data)
        images = torch.stack(data["imgs"], dim=1)  # n, v, c, h, w
        poses = torch.stack(data["poses"], dim=1)  # n, v, 4, 4 w2c
        projs = torch.stack(data["projs"], dim=1)  # n, v, 4, 4 (-1, 1)
        bboxes_3d = (
            [
                (
                    Bbox(
                        center=bbox_dict["center"],
                        whl=bbox_dict["whl"],
                        rotation=bbox_dict["rotation"],
                        label=bbox_dict["semanticId"],
                    )
                    if bbox_dict
                    else None # no 3d bboxes in this batch
                )
                for bbox_dict in data["3d_bboxes"]
            ] # eval w/ 3d bboxes, eval student or teacher
            if "3d_bboxes" in data
            else None # eval w/o 3d bboxes, eval student
        )

        self.count += 1

        world_transform = torch.inverse(poses[:, :1, :, :])  # transform to camera0
        # world_transform = cam_incl_adjust.to(images.device) @ world_transform  # add inclination
        # NOTE: already add incl in voxel.py
        poses = world_transform @ poses

        n, v, c, h, w = images.shape
        self.sampler.height = h
        self.sampler.width = w

        rays, _ = self.sampler.sample(
            None, poses[:, :1, :, :], projs[:, :1, :, :]
        )  # [1, 192*640=122880, 8]

        ids_encoder = [0]

        self.renderer.net.encode(
            images,
            projs,
            poses,
            ids_encoder=ids_encoder,
            ids_render=ids_encoder,
            images_alt=images * 0.5 + 0.5,
        )

        # compute voxel centers coordinates in cam system
        q_pts = (
            (data["voxellidar2c"] @ self.points_velo_h[None, ...].expand(n, -1, -1))[:, :3, :]
            .permute(0, 2, 1)
            .reshape(-1, 3)
        )

        # Query the density of the query points from the density field
        densities = []
        for i_from in range(0, len(q_pts), self.query_batch_size):
            i_to = min(i_from + self.query_batch_size, len(q_pts))
            q_pts_ = q_pts[i_from:i_to]
            _, _, densities_, _, _ = self.renderer.net(
                q_pts_.unsqueeze(0), bboxes_3d, only_density=False
            )
            densities.append(densities_.squeeze(0))
        densities = torch.cat(densities, dim=0).squeeze()
        is_occupied_pred = (densities > self.occ_threshold).reshape(
            n, *VOXEL_RESOLUTION
        )

        voxel = torch.stack(
            [process_unlabeled_voxel_under_ground(v) for v in data["voxel"]],
            dim=0,
        )
        q_pts_projected = q_pts[None, ...].expand(n, -1, -1) @ projs[
            :, 0, :, :
        ].permute(0, 2, 1)
        q_pts_uv = q_pts_projected[..., :2] / q_pts_projected[..., 2:]
        in_frustum = (
            (q_pts_uv > torch.tensor([-1, -1], device=q_pts_uv.device)).all(-1)
            & (q_pts_uv < torch.tensor([1, 1], device=q_pts_uv.device)).all(-1)
            & (q_pts_projected[..., 2] > EPS)
        ).view(1, *VOXEL_RESOLUTION)

        voxel = torch.stack(
            [process_unlabeled_voxel_under_ground(v) for v in data["voxel"]],
            dim=0,
        )
        is_occupied = (voxel != 0) & (voxel != 255)
        # for visualization
        if False:
            iop = is_occupied_pred.clone()
            iop[..., 16:] = 0
            self.visualizer.render(
                iop[0].cpu().numpy(),
                velo2cam=data["voxellidar2c"][0].cpu().numpy(),
                bbox_vertices_in_cam=(
                    data["3d_bboxes"][0]["vertices"].cpu().numpy()
                    if "3d_bboxes" in data
                    else None
                ),
            )
            self.visualizer.render(
                voxel[0].cpu().numpy(),
                velo2cam=data["voxellidar2c"][0].cpu().numpy(),
                bbox_vertices_in_cam=(
                    data["3d_bboxes"][0]["vertices"].cpu().numpy()
                    if "3d_bboxes" in data
                    else None
                ),
            )
            __import__('ipdb').set_trace()

        # o_acc, ie_acc, ie_rec = compute_occ_scores(is_occupied_pred, voxel, None, (voxel != 255))
        data["O_acc"] = 0
        data["IE_acc"] = 0
        data["IE_rec"] = 0
        data["IoU_I"] = (
            (is_occupied_pred & is_occupied & in_frustum & (voxel != 255))
            .float()
            .sum()
            .item()
        )
        data["IoU_U"] = (
            ((is_occupied_pred | is_occupied) & in_frustum & (voxel != 255))
            .float()
            .sum()
            .item()
        )
        data["IoU"] = (
            is_occupied_pred & is_occupied & in_frustum & (voxel != 255)
        ).float().sum().item() / (
            (is_occupied_pred | is_occupied) & in_frustum & (voxel != 255)
        ).float().sum().item()
        return data


def evaluation(local_rank, config):
    return base_evaluation(local_rank, config, get_dataflow, initialize, get_metrics)


def get_dataflow(config):
    test_dataset = make_test_dataset(config["data"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=config["num_workers"],
        shuffle=False,
        drop_last=False,
        collate_fn=(
            bbox3d_collate_fn if config["data"].get("return_3d_bboxes", True) else None
        ),
    )

    return test_loader


def get_metrics(config, device):
    names = [
        "O_acc",
        "IE_acc",
        "IE_rec",
        "IoU_I",
        "IoU_U",
        "IoU",
    ]
    metrics = {
        name: MeanMetric((lambda n: lambda x: x["output"][n])(name), device)
        for name in names
    }
    return metrics


def initialize(config: dict, logger=None):
    arch = config["model_conf"].get("arch", "BTSNet")
    net = globals()[arch](config["model_conf"])
    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()

    model = BTSWrapper(renderer, config["model_conf"])

    return model


def visualize(engine: Engine, logger: TensorboardLogger, step: int, tag: str):
    pass
