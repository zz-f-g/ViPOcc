import torch
from kornia.geometry.depth import depth_to_3d
import torch.nn.functional as F

def warp(img_r, depth_t, depth_r, pose_t2r, K):
    """
    warp a reference image to the target image.
    Args:
        img_r: [B, 3, H, W], the reference image (where to sample pixels)
        depth_t: [B, 1, H, W], target depth map
        depth_r: [B, 1, H, W], reference depth map
        pose_t2r: [B, 3, 4], relative pose from target image to reference image
        K: [B, 3, 3], camera intrinsic matrix

    Returns:

    """
    B, _, H, W = img_r.size()

    KT = torch.matmul(K, pose_t2r.float())  # [B, 3, 4]

    p_cam = depth_to_3d(depth_t, K)  # [B, 3, H, W], 3D points of target image
    p_cam = torch.cat([p_cam, torch.ones(B, 1, H, W).type_as(p_cam)], 1)  # [B, 4, H, W]
    p_ref = torch.matmul(KT, p_cam.view(B, 4, -1))  # =KTP, [B, 3, HxW]
    pix_coords = p_ref[:, :2, :] / (p_ref[:, 2, :].unsqueeze(1) + 1e-7)  # [B, 2, HxW]
    pix_coords = pix_coords.view(B, 2, H, W)
    pix_coords = pix_coords.permute(0, 2, 3, 1)  # [B, H, W, 2]

    pix_coords[..., 0] /= W - 1
    pix_coords[..., 1] /= H - 1
    pix_coords = (pix_coords - 0.5) * 2

    projected_img = F.grid_sample(img_r, pix_coords, align_corners=False)
    projected_depth = F.grid_sample(depth_r, pix_coords, align_corners=False)
    computed_depth = p_ref[:, 2, :].unsqueeze(1).view(B, 1, H, W)

    valid_points = pix_coords.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()

    return projected_img, projected_depth, computed_depth, valid_mask
