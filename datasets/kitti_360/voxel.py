import yaml

import numpy as np
from numpy.typing import NDArray
import open3d as o3d

from datasets.kitti_360.process_bbox3d import EDGES

EDGE_LINES = [list(edge) for group in EDGES.values() for edge in group]

VOXEL_ORIGIN = np.array([1, -25.6, -2.5])
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


class Visualizer:
    def __init__(
        self,
        hw: tuple[int, int],
        voxel_origin_in_velo: tuple[float, float, float],
        voxel_size: float,
        intrinsic_normalized: NDArray[np.float32],
        velo2cam: NDArray[np.float32],
        cam_incl_adjust: NDArray[np.float32],
    ):
        self.h: int = hw[0]
        self.w: int = hw[1]

        voxel2velo = np.eye(4)
        voxel2velo[:3, 3] = np.array(voxel_origin_in_velo)
        self.voxel2velo: NDArray[np.float32] = voxel2velo
        self.voxel_size: float = voxel_size

        K = np.copy(intrinsic_normalized)
        K[0, 0] = K[0, 0] / 2.0 * self.w
        K[1, 1] = K[1, 1] / 2.0 * self.h
        K[0, 2] = (K[0, 2] + 1) / 2.0 * self.w
        K[1, 2] = (K[1, 2] + 1) / 2.0 * self.h
        self.intrinsic: NDArray[np.float32] = K
        self.intrinsic_normalized = intrinsic_normalized

        self.velo2cam: NDArray[np.float32] = velo2cam
        self.cam_incl_adjust: NDArray[np.float32] = cam_incl_adjust

        with open("datasets/kitti_360/sscbench-kitti360.yaml", "r") as f:
            kitti360 = yaml.safe_load((f))
        learning_map_inv = np.zeros((19,), dtype=np.uint8)
        for label in kitti360["learning_map_inv"]:
            learning_map_inv[label] = kitti360["learning_map_inv"][label]
        color_map = np.zeros((260, 3), dtype=np.uint8)
        for label in kitti360["color_map"]:
            color_map[label] = np.array(kitti360["color_map"][label], dtype=np.uint8)
        self.color_lut: NDArray[np.float32] = (
            np.concatenate(
                (
                    np.flip(color_map[learning_map_inv], axis=-1),  # bgr -> rgb
                    np.full((learning_map_inv.shape[0], 1), 255),
                ),
                axis=1,
            )
            / 255.0
        )

    def render(
        self,
        voxel: NDArray[np.uint32],
        velo2cam: NDArray[np.float32] | None = None,
        bbox_vertices_in_cam: NDArray[np.float32] | None = None,
    ):
        extrinsic: NDArray[np.float32] = (
            np.linalg.inv(self.cam_incl_adjust)
            @ (velo2cam if velo2cam is not None else self.velo2cam)
            @ self.voxel2velo
        )
        X, Y, Z = voxel.shape
        vox_coords = np.stack(
            np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing="ij"),
            axis=0,
        ).reshape(3, -1)
        points_voxel = (vox_coords.astype(np.float32) + 0.5) * VOXEL_SIZE

        points_cam = extrinsic @ np.concatenate(
            [points_voxel, np.ones_like(points_voxel[:1, ...])],
            axis=0,
        )
        points_projected = self.intrinsic_normalized @ points_cam[:3, :]
        points_uv = (points_projected[:2] / points_projected[2:]).transpose()
        in_frustum = (
            (points_uv > np.array([-1, -1])).all(-1)
            & (points_uv < np.array([1, 1])).all(-1)
            & (points_projected[2] > 1e-3)
        ).reshape(*voxel.shape)

        vox = np.where((voxel == 0) | (~in_frustum), 255, voxel)
        non_empty_indices = np.stack(np.where(vox != 255), axis=0).T
        semantic_values = vox[vox != 255]
        pcd = o3d.geometry.PointCloud()

        non_empty_indices = (
            non_empty_indices.astype(np.float32) + 0.5
        ) * self.voxel_size
        pcd.points = o3d.utility.Vector3dVector(non_empty_indices)

        if semantic_values is not None:
            colors = self.color_lut[:, :3][semantic_values].astype(np.float32)
        else:
            colors = self.color_lut[:, :3][
                np.ones_like(non_empty_indices[..., 0], dtype=np.uint32)
            ]

        pcd.colors = o3d.utility.Vector3dVector(colors)

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, voxel_size=self.voxel_size
        )

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.w, height=self.h)

        vis.add_geometry(voxel_grid)
        if bbox_vertices_in_cam is not None:
            for vertices_in_cam in bbox_vertices_in_cam:
                vis.add_geometry(
                    create_wireframe_box(
                        (
                            np.concatenate(
                                [
                                    vertices_in_cam,
                                    np.ones_like(vertices_in_cam[..., :1]),
                                ],
                                axis=-1,
                            )
                            @ np.linalg.inv(
                                (velo2cam if velo2cam is not None else self.velo2cam)
                                @ self.voxel2velo
                            ).T
                        )[..., :3]
                    )
                )

        view_control = vis.get_view_control()
        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.intrinsic.intrinsic_matrix = self.intrinsic
        camera_params.extrinsic = extrinsic
        view_control.convert_from_pinhole_camera_parameters(
            camera_params, allow_arbitrary=True
        )

        vis.poll_events()
        vis.update_renderer()
        vis.run()
        vis.destroy_window()
