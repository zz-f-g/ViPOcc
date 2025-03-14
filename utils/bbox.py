from typing import NamedTuple
import numpy as np
import torch
from torch import Tensor


class Bbox(NamedTuple):
    """
    We use camera coordinates system in
    rotation is from car to camera
    I_3 rotation for cars parallel to kitti car
    """

    center: Tensor  # [N, 3] N is the number of bboxes
    whl: Tensor  # [N, 3 (width, height, length)]
    rotation: Tensor  # [N, 3, 3]
    label: Tensor  # [N]


def point_in_which_bbox(point: Tensor, bbox: Bbox):
    """
    >>> bbox = Bbox(
    ...     center=torch.tensor(
    ...         [
    ...             [0.0, 0.0, 0.0],
    ...             [0.0, 0.0, 5.0],
    ...         ]
    ...     ),
    ...     whl=torch.tensor(
    ...         [
    ...             [2.0, 1.0, 3.0],
    ...             [2.0, 1.0, 3.0],
    ...         ]
    ...     ),
    ...     rotation=torch.tensor(
    ...         [
    ...             [
    ...                 [1.0, 0.0, 0.0],
    ...                 [0.0, 1.0, 0.0],
    ...                 [0.0, 0.0, 1.0],
    ...             ],
    ...             [
    ...                 [0.0, 0.0, 1.0],
    ...                 [0.0, 1.0, 0.0],
    ...                 [-1.0, 0.0, 0.0],
    ...             ],
    ...         ]
    ...     ),
    ...     label=torch.tensor([1, 1]),
    ... )
    >>> point = torch.tensor(
    ...     [
    ...         [0.0, 0.0, 0.0],
    ...         [0.6, 0.6, 0.6],
    ...         [0.9, 0.0, 0.9],
    ...         [0.9, 0.4, 1.4],
    ...         [1.1, 0.4, 0.6],
    ...         [1.4, 0.4, 4.1],
    ...         [1.6, 0.4, 0.6],
    ...     ]
    ... )
    >>> point_in_which_bbox(point, bbox)[1]
    tensor([[ True, False,  True,  True, False, False, False],
            [False, False, False, False, False,  True, False]])
    """
    n_pts, _ = point.shape
    n_bbox = bbox.label.shape[0]
    point_homogenous = torch.cat([point, point.new_ones((n_pts, 1))], dim=1)  # [NP, 4]
    t_b2c = torch.cat(
        (
            torch.cat((bbox.rotation, bbox.rotation.new_zeros((n_bbox, 1, 3))), dim=1),
            torch.cat((bbox.center, bbox.center.new_ones((n_bbox, 1))), dim=1)[
                ..., None
            ],
        ),
        dim=2,
    )  # [NB, 4, 4]
    point_in_bbox = (torch.inverse(t_b2c) @ point_homogenous.permute(1, 0))[
        :, :3, :
    ] / (
        bbox.whl[..., None] / 2
    )  # [NB, 3, NP]
    return point_in_bbox.permute(0, 2, 1), torch.all(
        (point_in_bbox > -1) & (point_in_bbox < 1), dim=1
    )  # [NB, NP, 3], [NB, NP]


def get_density_in_bbox(mlp, feature: Tensor, point: Tensor, bboxes: list[Bbox]):
    n, p, cf = feature.shape
    assert point.shape == (n, p, 3)
    assert len(bboxes) == n

    densities = []
    masks = []
    for feature_eatch_batch, point_eatch_batch, bbox in zip(feature, point, bboxes):
        if bbox is None:
            densities.append(point.new_zeros((p,))) # [NP]
            masks.append(point.new_zeros((p,)).to(torch.bool)) # [NP]
            continue
        n_bbox_this_batch = bbox.center.shape[0]
        point_in_bbox, mask = point_in_which_bbox(
            point_eatch_batch, bbox
        )  # [NB, NP, 3] [NB, NP]
        density_eatch_batch = mask.to(feature.dtype) * mlp(
            feature_eatch_batch[None, ...].expand(n_bbox_this_batch, -1, -1),
            point_in_bbox,
        )  # [NB, NP]
        densities.append(1 - (1 - density_eatch_batch).prod(dim=0))  # [NP]
        masks.append(mask.any(dim=0))  # [NP]

    return torch.stack(densities, dim=0), torch.stack(masks, dim=0)  # [B, NP] [B, NP]


def draw_3d_bbox(image, vertices_2d, thickness=1):
    """
    在 2D 图像上绘制 3D bounding box，并在特定顶点上绘制空心小圆圈

    参数：
        image: np.ndarray -> 输入的图像 (H, W, 3)
        vertices_2d: np.ndarray -> 8 个顶点的 2D 坐标 (8, 2)
        color: tuple -> 线条颜色
        thickness: int -> 线条粗细

    返回：
        带有 3D bounding box 和标记的图像
    """
    H, W, _ = image.shape
    vertices_2d *= 0.5
    vertices_2d += 0.5
    vertices_2d *= np.array([W - 1, H - 1])
    vertices_2d = vertices_2d.astype(np.int32)
    image = image.copy()

    # 定义 3D 盒子的边
    edges = [
        (2, 3),
        (5, 4),
        (7, 6),  # 上下边缘
        (2, 7),
        (1, 4),
        (3, 6),  # 竖直边
        (1, 3),
        (4, 6),
        (5, 7),  # 前后边缘
        (0, 1),
        (0, 5),
        (0, 2),
    ]

    some_edges = {
        (0, 1): (1.0, 0, 0),  # height
        (0, 2): (0, 1.0, 0),  # width
        (0, 5): (0, 0, 1.0),  # length
    }

    # 绘制 3D bounding box 的线条
    for edge in edges:
        cv2.line(
            image,
            tuple(vertices_2d[edge[0]]),
            tuple(vertices_2d[edge[1]]),
            some_edges[edge] if edge in some_edges else (1.0, 1.0, 1.0),
            thickness,
            lineType=cv2.LINE_AA,
        )

    return image


def project(verts: np.ndarray, proj: np.ndarray):
    verts = (proj @ verts.T).T
    verts[:, :2] /= np.abs(verts[:, 2:3])
    return verts[:, :2]


def draw_bbox_wrapper(img, bboxes_vertices, proj):
    img = (img * 0.5 + 0.5).permute(1, 2, 0).numpy()  # .astype(np.uint8) * 255
    # bboxes_faces = [bbox["faces"] for bbox in bboxes]
    image_with_bbox = img
    for bbox_vertices in bboxes_vertices:
        image_with_bbox = draw_3d_bbox(
            image_with_bbox,
            project(bbox_vertices, proj),
        )
    return image_with_bbox


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    # feature = torch.rand((2, 3, 4))
    # point = torch.rand((2, 3, 3))
    # bboxes = [
    #     Bbox(
    #         center=torch.zeros((1, 3)),
    #         whl=torch.tensor((2, 1, 3)).view(1, 3),
    #         rotation=torch.eye(3)[None, ...],
    #         label=torch.zeros((1,)),
    #     ),
    #     Bbox(
    #         center=torch.full((1, 3), 10),
    #         whl=torch.tensor((2, 1, 3)).view(1, 3),
    #         rotation=torch.eye(3)[None, ...],
    #         label=torch.zeros((1,)),
    #     ),
    # ]
    # sigma, mask = get_density_in_bbox(feature, point, bboxes)
