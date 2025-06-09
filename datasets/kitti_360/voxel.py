import yaml

import numpy as np
from numpy.typing import NDArray
import open3d as o3d

from datasets.kitti_360.process_bbox3d import EDGES

EDGE_LINES = [list(edge) for group in EDGES.values() for edge in group]

VOXEL_ORIGIN = np.array([0, -25.6, -2])
VOXEL_SIZE = 0.2
VOXEL_RESOLUTION = (256, 256, 32)


def create_wireframe_box(vertices, color=[1, 0, 0]):
    line_colors = [color for _ in EDGE_LINES]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(EDGE_LINES),
    )
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    return line_set


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

def vis_voxel_bbox(non_empty_indices: NDArray[np.uint32], semantic_values: NDArray[np.uint32] | None = None, bboxes_verts: NDArray[np.float32] | None = None):
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
    ) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(non_empty_indices)
    pcd.colors = o3d.utility.Vector3dVector(
        color_lut[:, :3][
            (
                np.ones_like(non_empty_indices[..., 0], dtype=np.uint8)
                if semantic_values is None
                else semantic_values
            )
        ]
    )
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)

        )
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        return line_set
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(
        [voxel_grid, coord_frame]
        if bboxes_verts is None
        else [voxel_grid, coord_frame]
        + [create_wireframe_box(verts) for verts in bboxes_verts]
    )


def prepare_voxel(
    projs: NDArray[np.float32],
    voxellidar2c: NDArray[np.float32],
    voxel: NDArray[np.int32],
    vertices: NDArray[np.float32] | None,
):
    X, Y, Z = VOXEL_RESOLUTION
    vox_coords = np.stack(
        np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing="ij"),
        axis=0,
    ).reshape(3, -1)
    points_velo = (
        (vox_coords + 0.5) * VOXEL_SIZE + np.array(VOXEL_ORIGIN).reshape(3, 1)
    ).astype(np.float32)
    points_cam = voxellidar2c @ np.concatenate(
        [points_velo, np.ones_like(points_velo[:1, ...])],
        axis=0,
    )
    points_projected = projs @ points_cam[:3, :]
    points_uv = (points_projected[:2] / points_projected[2:]).transpose()
    in_frustum = (
        (points_uv > np.array([-1, -1])).all(-1)
        & (points_uv < np.array([1, 1])).all(-1)
        & (points_projected[2] > 1e-3)
    ).reshape(*VOXEL_RESOLUTION)
    vox = np.where((voxel == 0) | (~in_frustum), 255, voxel)
    non_empty_indices = np.stack(np.where(vox != 255), axis=0).T
    semantic_values = vox[vox != 255]
    if vertices is None:
        return non_empty_indices, semantic_values, None
    else:
        bboxes_verts_in_voxellidar = (
            np.concatenate(
                [vertices, np.ones(vertices[..., :1].shape)],
                axis=-1,
            ).reshape(-1, 4)
            @ np.linalg.inv(voxellidar2c).T
        )[..., :3].reshape(vertices.shape)

        bboxes_verts_in_voxel = (
            bboxes_verts_in_voxellidar - VOXEL_ORIGIN
        ) / VOXEL_SIZE - 0.5
        return non_empty_indices, semantic_values, bboxes_verts_in_voxel

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
