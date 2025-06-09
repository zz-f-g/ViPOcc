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


def vis_voxel_bbox(
    non_empty_indices: NDArray[np.uint32],
    semantic_values: NDArray[np.uint32] | None = None,
    bboxes_verts: NDArray[np.float32] | None = None,
):
    with open("datasets/kitti_360/sscbench-kitti360.yaml", "r") as f:
        kitti360 = yaml.safe_load((f))
    learning_map_inv = np.zeros((19,), dtype=np.uint8)
    for label in kitti360["learning_map_inv"]:
        learning_map_inv[label] = kitti360["learning_map_inv"][label]
    color_map = np.zeros((260, 3), dtype=np.uint8)
    for label in kitti360["color_map"]:
        color_map[label] = np.array(kitti360["color_map"][label], dtype=np.uint8)
    color_lut = (
        np.concatenate(
            (
                np.flip(color_map[learning_map_inv], axis=-1),  # bgr -> rgb
                np.full((learning_map_inv.shape[0], 1), 255),
            ),
            axis=1,
        )
        / 255.0
    )

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

    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=5.0, origin=[0, 0, 0]
    # )
    # o3d.visualization.draw_geometries(
    #     [voxel_grid, coord_frame]
    #     if bboxes_verts is None
    #     else [voxel_grid, coord_frame]
    #     + [create_wireframe_box(verts) for verts in bboxes_verts]
    # )
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=2560, height=1440)
    vis.add_geometry(voxel_grid)
    vis.add_geometry(axis)
    if bboxes_verts is not None:
        for verts in bboxes_verts:
            vis.add_geometry(create_wireframe_box(verts))
    vis.run()


            np.concatenate(
                (
                    np.flip(color_map[learning_map_inv], axis=-1),  # bgr -> rgb
                    np.full((learning_map_inv.shape[0], 1), 255),
                ),
                axis=1,
            )
            / 255.0
        )


