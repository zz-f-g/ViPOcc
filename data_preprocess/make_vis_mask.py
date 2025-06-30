import os
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from datasets.kitti_360 import voxel
from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset
from datasets.kitti_360.voxel import (
    VOXEL_ORIGIN,
    VOXEL_SIZE,
    VOXEL_RESOLUTION,
    Visualizer,
)


def meshgrid(h: int, w: int, device: torch.device):
    """
    >>> print(meshgrid(2, 3, "cpu"))
    tensor([[[0., 0.],
             [1., 0.],
             [2., 0.]],
    <BLANKLINE>
            [[0., 1.],
             [1., 1.],
             [2., 1.]]])
    """
    return torch.stack(
        torch.meshgrid(
            [
                torch.arange(w, device=device),
                torch.arange(h, device=device),
            ],
            indexing="xy",
        ),
        dim=-1,
    ).to(torch.float32)


def homogeneous(input: Tensor, dim:int):
    return torch.cat([input, torch.ones_like(input.narrow(dim, 0, 1))], dim=dim)


def make_cam_point(
    K: Tensor,
    h: int,
    w: int,
    voxel_size: float,
    voxel_res: (int, int, int),
    device: torch.device,
):
    assert K.shape == (3, 3)
    X, Y, Z = voxel_res
    pixel_h = homogeneous(meshgrid(h, w, device), dim=-1)
    ray_norm = F.normalize(pixel_h.view(h * w, 3) @ torch.inverse(K).T, dim=-1)
    n_points = int((X**2 + Y**2 + Z**2) ** 0.5) + 1
    return (
        ray_norm.view(h * w, 1, 3) * torch.arange(n_points, device=device).view(n_points, 1) * voxel_size
    )  # [hw, p, 3]


def voxelize_point(
    point: Tensor,
    voxel_size: float,
    voxel_res: (int, int, int),
):
    """
    >>> point = torch.tensor([[1.5, 2.5, 3.5], [0.5, 1.0, 2.0]])
    >>> voxel_size = 0.5
    >>> voxel_res = (10, 10, 10)
    >>> voxelized, valid = voxelize_point(point, voxel_size, voxel_res)
    >>> voxelized
    tensor([[3, 5, 7],
            [1, 2, 4]], dtype=torch.int32)
    >>> valid
    tensor([True, True])

    >>> point = torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    >>> voxelized, valid = voxelize_point(point, voxel_size, voxel_res)
    >>> voxelized
    tensor([[ 0,  0,  0],
            [10, 10, 10]], dtype=torch.int32)
    >>> valid
    tensor([False, False])

    >>> point = torch.tensor([2.0, 3.0, 4.0])
    >>> voxelized, valid = voxelize_point(point, voxel_size, voxel_res)
    >>> voxelized
    tensor([4, 6, 8], dtype=torch.int32)
    >>> valid
    tensor(True)

    >>> point = torch.tensor([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0]])
    >>> voxelized, valid = voxelize_point(point, voxel_size, voxel_res)
    >>> voxelized
    tensor([[-2,  2,  2],
            [ 2, -2,  2]], dtype=torch.int32)
    >>> valid
    tensor([False, False])
    """
    assert point.shape[-1] == 3
    X, Y, Z = voxel_res
    voxelized_point = (point / voxel_size).to(torch.int32)
    return (
        voxelized_point,  # [..., 3]
        (
            (voxelized_point[..., 0] > 0)
            & (voxelized_point[..., 0] < X)
            & (voxelized_point[..., 1] > 0)
            & (voxelized_point[..., 1] < Y)
            & (voxelized_point[..., 2] > 0)
            & (voxelized_point[..., 2] < Z)
        ),  # [...]
    )


def reduce_ray(is_occupied_each_ray: Tensor):
    """
    >>> _ = torch.manual_seed(0)
    >>> rayocc = torch.rand((5, 10)) < 0.3
    >>> print(rayocc)
    tensor([[False, False,  True,  True, False, False, False, False, False, False],
            [False, False,  True,  True,  True, False, False, False,  True,  True],
            [False, False, False, False, False, False, False,  True,  True, False],
            [False, False,  True,  True,  True,  True,  True, False, False, False],
            [False,  True, False,  True,  True,  True, False, False,  True, False]])
    >>> print(reduce_ray(rayocc))
    tensor([[ True,  True,  True, False, False, False, False, False, False, False],
            [ True,  True,  True, False, False, False, False, False, False, False],
            [ True,  True,  True,  True,  True,  True,  True,  True, False, False],
            [ True,  True,  True, False, False, False, False, False, False, False],
            [ True,  True, False, False, False, False, False, False, False, False]])
    """
    assert is_occupied_each_ray.dtype == torch.bool
    occupied_cumsum = torch.cumsum(is_occupied_each_ray.to(torch.int32), dim=-1)
    return occupied_cumsum == 0 | (is_occupied_each_ray & (occupied_cumsum == 1))


def get_visible_mask(
    K: Tensor,
    T_cam_to_voxel: Tensor,
    h: int,
    w: int,
    occupied_voxel: Tensor,
    voxel_size: float,
    device: torch.device,
):
    cam_point = make_cam_point(K, h, w, voxel_size, occupied_voxel.shape, device)
    voxel_point = (homogeneous(cam_point, dim=-1) @ T_cam_to_voxel.T)[..., :3]
    voxelized_point, valid_mask = voxelize_point(
        voxel_point, voxel_size, occupied_voxel.shape
    )  # [hw, p, 3], [hw, p]
    clamped_voxelized_point = torch.where(
        torch.stack([valid_mask] * 3, dim=-1), voxelized_point, 0
    )

    occupied_point = torch.where(
        valid_mask,
        (
            occupied_voxel[
                clamped_voxelized_point[..., 0],
                clamped_voxelized_point[..., 1],
                clamped_voxelized_point[..., 2],
            ]
        ),
        False,
    )
    visible_point_mask = reduce_ray(occupied_point).view(-1)
    visible_point = voxelized_point.view(-1, 3)[visible_point_mask]
    X, Y, Z = occupied_voxel.shape
    visible_valid_mask = (
        (visible_point[:, 0] >= 0)
        & (visible_point[:, 0] < X)
        & (visible_point[:, 1] >= 0)
        & (visible_point[:, 1] < Y)
        & (visible_point[:, 2] >= 0)
        & (visible_point[:, 2] < Z)
    )
    visible_valid_point = visible_point[visible_valid_mask]

    visible_mask = torch.zeros_like(occupied_voxel, dtype=torch.bool)
    visible_mask[visible_valid_point[:, 0], visible_valid_point[:, 1], visible_valid_point[:, 2]] = True
    return visible_mask


def main(root_dir: Path):
    assert root_dir.exists()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    calib = Kitti360Dataset._load_calibs("data/KITTI-360/", (0, -15))
    visualizer = Visualizer(
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

    test_dataset = Kitti360Dataset(
        data_path='data/KITTI-360',
        pose_path='data/KITTI-360/data_poses',
        split_path=os.path.join('datasets/kitti_360/splits/seg', "test_files.txt"),
        target_image_size=(192, 640),
        frame_count=1,
        return_stereo=False,
        return_fisheye=False,
        return_gt_depth=False,
        return_pseudo_depth=False,
        return_3d_bboxes=True,
        bboxes_semantic_labels=["car",],
        return_segmentation=False,
        return_voxel=True,
        keyframe_offset=0,
        fisheye_rotation=-15,
        fisheye_offset=10,
        dilation=1,
        is_preprocessed=True,
    )
    print("Dataset is initialized")

    height, width = calib["im_size"]

    for _i, data in enumerate(test_dataset):
        voxel_grid = data["voxel"]

        intrinsic_matrix = torch.tensor(
            visualizer.intrinsic, device=device, dtype=torch.float32
        )
        extrinsic_matrix = torch.tensor(
            np.linalg.inv(visualizer.cam_incl_adjust)
            @ calib["T_velo_to_cam"]["00"]
            @ visualizer.voxel2velo,
            device=device,
            dtype=torch.float32,
        )

        visible_mask = get_visible_mask(
            intrinsic_matrix,
            torch.inverse(extrinsic_matrix),
            height,
            width,
            torch.tensor(((voxel_grid != 0) & (voxel_grid != 255)), device=device),
            VOXEL_SIZE,
            device,
        )
        seq_dir = root_dir / str(data["seq"])
        seq_dir.mkdir(exist_ok=True)
        img_id = data["img_id"]
        output_path = seq_dir / f"{img_id:0>10}.npy"
        if output_path.exists():
            print(f"{output_path} already exists")
            continue
        np.save(output_path, visible_mask.cpu().numpy())
        print(f"Saved to {output_path}")

    # visualizer.render(voxel_grid)
    # visualizer.render((~visible_mask).cpu().numpy())

    return


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main(Path("./data/KITTI-360/visible_mask"))
