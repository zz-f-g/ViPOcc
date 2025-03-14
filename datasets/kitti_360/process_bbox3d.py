import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torch import Tensor

EDGES = {
    "length": (
        (0, 5),
        (1, 4),
        (3, 6),
        (2, 7),
    ),
    "width": (
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
    ),
    "height": (
        (0, 1),
        (2, 3),
        (7, 6),
        (5, 4),
    ),
}


def compute_edge(vertices: Tensor, direction: str):
    assert direction in EDGES
    edges = []
    for edge_idx in EDGES[direction]:
        edges.append(vertices[:, edge_idx[0], :] - vertices[:, edge_idx[1], :])
    edge = torch.stack(edges, dim=2).mean(dim=2)
    edge_norm = torch.norm(edge, dim=1)
    return edge_norm, edge / edge_norm[..., None]


def convert_vertices(vertices: Tensor):
    assert vertices.shape[1:] == (8, 3)
    length, e_l = compute_edge(vertices, "length")
    width, e_w = compute_edge(vertices, "width")
    height, e_h = compute_edge(vertices, "height")
    rotation_matrix = torch.stack([-e_w, -e_h, e_l], dim=2)
    # euler_angles = R.from_matrix(rotation_matrix).as_euler("yxz", degrees=False)
    return {
        "center": vertices.mean(axis=1),
        "whl": torch.stack((width, height, length), dim=1),
        "rotation": rotation_matrix,
    }
