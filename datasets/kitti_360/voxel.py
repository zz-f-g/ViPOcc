import argparse
from pathlib import Path
import yaml

import numpy as np
from numpy.typing import NDArray
import open3d as o3d
from mayavi import mlab

VOXEL_ORIGIN = np.array([0, -25.6, -2])
VOXEL_SIZE = 0.2
VOXEL_RESOLUTION = (256, 256, 32)


def read_calib():
    P = np.array(
        [
            552.554261, 0.000000, 682.049453, 0.000000,
            0.000000, 552.554261, 238.769549, 0.000000,
            0.000000, 0.000000, 1.000000, 0.000000,
        ]
    ).reshape(3, 4)
    cam2velo = np.array(
        [
            0.04307104361, -0.08829286498, 0.995162929, 0.8043914418,
            -0.999004371, 0.007784614041, 0.04392796942, 0.2993489574,
            -0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824,
        ]
    ).reshape(3, 4)
    return {
        "intrinsic": P,
        "velo2cam": np.concatenate(
            [
                np.linalg.inv(
                    np.concatenate(
                        [cam2velo, np.array([0, 0, 0, 1]).reshape(1, 4)],
                        axis=0,
                    )
                )[:3, :4],
                np.array([0, 0, 0, 1]).reshape(1, 4),
            ],
            axis=0,
        ),
    }


def project_voxel(
    cam_K: NDArray[np.float32],
    cam_E: NDArray[np.float32],
    img_H: int | None = None,
    img_W: int | None = None,
):
    X, Y, Z = VOXEL_RESOLUTION
    xx, yy, zz = np.meshgrid(range(X), range(Y), range(Z), indexing="ij")
    vox_coords = np.concatenate(
        [
            xx.reshape(1, -1),
            yy.reshape(1, -1),
            zz.reshape(1, -1),
        ],
        axis=0,
    ).astype(
        np.int32
    )  # [3, XYZ]
    points_velo = (
        vox_coords + np.array([0.5, 0.5, 0.5]).reshape(3, 1)
    ) * VOXEL_SIZE + VOXEL_ORIGIN.reshape(3, 1)
    points_cam = cam_E @ np.concatenate(
        (
            points_velo,
            np.ones_like(points_velo[:1, :]),
        ),
        axis=0,
    )
    points_proj_ = (cam_K @ points_cam).T
    z = points_proj_[:, 2].reshape(X, Y, Z)
    points_proj = (points_proj_[:, :2] / points_proj_[:, 2:] + 1e-3).reshape(X, Y, Z, 2)
    if img_H is None or img_W is None:
        frustum_mask = (
            (points_proj > np.array([-1, -1])).all(axis=-1)
            & (points_proj < np.array([1, 1])).all(axis=-1)
            & (z > 1e-3)
        )

    else:
        frustum_mask = (
            (points_proj > 0).all(axis=-1)
            & (points_proj < np.array([img_W - 1, img_H - 1])).all(axis=-1)
            & (z > 1e-3)
        )
    return points_proj, z, frustum_mask


def voxel2pc(
    voxel: NDArray[np.uint8],
    learning_map_inv: NDArray[np.uint8],
    color_map: NDArray[np.uint8],
    frustum_mask: NDArray[np.bool_] | None,
):
    if frustum_mask is not None:
        points_in_frustum = np.where((voxel != 255) & (voxel != 0) & frustum_mask)
        points_out_frustum = np.where((voxel != 255) & (voxel != 0) & ~frustum_mask)
        return (
            np.concatenate(
                (
                    np.stack(points_in_frustum, axis=-1),
                    np.stack(points_out_frustum, axis=-1),
                ),
                axis=0,
            ),
            np.concatenate(
                (
                    color_map[learning_map_inv[voxel[points_in_frustum]]] / 255.0,
                    color_map[learning_map_inv[voxel[points_out_frustum]]] / 255.0 / 2,
                ),
                axis=0,
            ),
            (voxel == 255),
        )
    else:
        points = np.where((voxel != 255) & (voxel != 0))
        return (
            np.stack(points, axis=-1),
            color_map[learning_map_inv[voxel[points]]] / 255.0,
            (voxel == 255),
        )


def vis_voxel(voxel: NDArray[int], frustum_mask: NDArray[bool] | None = None):
    calib_info = read_calib()
    if frustum_mask is None:
        _points_proj, _z, frustum_mask = project_voxel(
            calib_info["intrinsic"], calib_info["velo2cam"], 376, 1408
        )
    with open("datasets/kitti_360/sscbench-kitti360.yaml", "r") as f:
        kitti360 = yaml.safe_load((f))
    learning_map_inv = np.zeros((19,), dtype=np.uint8)
    for label in kitti360["learning_map_inv"]:
        learning_map_inv[label] = kitti360["learning_map_inv"][label]
    color_map = np.zeros((260, 3), dtype=np.uint8)
    for label in kitti360["color_map"]:
        color_map[label] = np.array(kitti360["color_map"][label], dtype=np.uint8)

    # points, colors, _invalid = voxel2pc(voxel, learning_map_inv, color_map, frustum_mask)
    points, colors, _invalid = voxel2pc(voxel, learning_map_inv, color_map)

    # 可视化点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.flip(colors, axis=-1))  # bgr -> rgb
    o3d.visualization.draw_geometries([pcd])


def vis_voxel_grid(non_empty_indices: NDArray[int], semantic_values: NDArray[int] | None = None):
    with open("datasets/kitti_360/sscbench-kitti360.yaml", "r") as f:
        kitti360 = yaml.safe_load((f))
    learning_map_inv = np.zeros((19,), dtype=np.uint8)
    for label in kitti360["learning_map_inv"]:
        learning_map_inv[label] = kitti360["learning_map_inv"][label]
    color_map = np.zeros((260, 3), dtype=np.uint8)
    for label in kitti360["color_map"]:
        color_map[label] = np.array(kitti360["color_map"][label], dtype=np.uint8)
    color_lut = np.concatenate(
        (
            np.flip(color_map[learning_map_inv], axis=-1), # bgr -> rgb
            np.full((learning_map_inv.shape[0], 1), 255),
        ),
        axis=1,
    ).astype(np.uint8)

    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    plt_plot_fov = mlab.points3d(
        non_empty_indices[..., 0] - VOXEL_RESOLUTION[0] / 2,
        non_empty_indices[..., 1] - VOXEL_RESOLUTION[1] / 2,
        non_empty_indices[..., 2] - VOXEL_RESOLUTION[2] / 2,
        scale_factor=1.0,
        mode="cube",
        opacity=1,
        vmin=0,
        vmax=18,
    )
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.mlab_source.dataset.point_data.scalars = color_lut[
        (
            np.ones_like(non_empty_indices[..., 0], dtype=np.uint8)
            if semantic_values is None
            else semantic_values
        )
    ]
    # scene = figure.scene
    # scene.camera.position = [118.7195754824976, 118.70290907014409, 120.11124225247899]
    # scene.camera.focal_point = [0.008333206176757812, -0.008333206176757812, 1.399999976158142]
    # scene.camera.view_angle = 30.0
    # scene.camera.view_up = [0.0, 0.0, 1.0]
    # scene.camera.clipping_range = [114.42016931210819, 320.9039783052695]
    # scene.camera.compute_view_plane_normal()
    # scene.camera.orthogonalize_view_up()
    # scene.render()
    mlab.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="read in seq & idx")
    parser.add_argument("seq", type=int, help="seq {0:11} \ {01, 08}")
    parser.add_argument("idx", type=int, help="idx")
    args = parser.parse_args()
    root_path = Path("/mnt/disk/sscbench-kitti/preprocess/labels")
    voxels_dir = None
    ext_map = {"invalid": ".invalid", "voxel": ".label", "gt": "_1_1.npy"}
    get_path = lambda seq, idx, ext: (
        root_path / f"2013_05_28_drive_{seq:0>4}_sync" / voxels_dir / f"{idx:0>6}{ext}"
        if voxels_dir
        else root_path / f"2013_05_28_drive_{seq:0>4}_sync" / f"{idx:0>6}{ext}"
    )
    path = get_path(args.seq, args.idx, ext_map["gt"])
    assert path.exists(), path.as_posix() + " does not exist."
    voxel = np.load(path).astype(np.uint8)
    vis_voxel_grid(voxel, 255)
