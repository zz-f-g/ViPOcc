import torch
import torch.nn.functional as F

from models.common.model.layers import ssim
from utils.warp import warp


def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 10000:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0.).type_as(diff)
    return mean_value


def compute_errors_l1ssim(img0, img1, mask=None):
    """
    Computes L1 SSIM between two images using L1 SSIM.
    """
    n, pc, h, w, nv, c = img0.shape
    img1 = img1.expand(img0.shape)
    img0 = img0.permute(0, 1, 4, 5, 2, 3).reshape(-1, c, h, w)
    img1 = img1.permute(0, 1, 4, 5, 2, 3).reshape(-1, c, h, w)
    errors = .85 * torch.mean(ssim(img0, img1, pad_reflection=False, gaussian_average=True, comp_mode=True),
                              dim=1) + .15 * torch.mean(torch.abs(img0 - img1), dim=1)
    errors = errors.view(n, pc, nv, h, w).permute(0, 1, 3, 4, 2).unsqueeze(-1)
    if mask is not None:
        return errors, mask
    else:
        return errors


def edge_aware_smoothness(gt_img, depth, mask=None):
    """
    Compute smoothness loss.
    """
    n, pc, h, w = depth.shape
    gt_img = gt_img.permute(0, 1, 4, 5, 2, 3).reshape(-1, 3, h, w)
    depth = 1 / depth.reshape(-1, 1, h, w).clamp(1e-3, 80)
    depth = depth / torch.mean(depth, dim=[2, 3], keepdim=True)

    gt_img = F.interpolate(gt_img, (h, w))

    d_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
    d_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

    i_dx = torch.mean(torch.abs(gt_img[:, :, :, :-1] - gt_img[:, :, :, 1:]), 1, keepdim=True)
    i_dy = torch.mean(torch.abs(gt_img[:, :, :-1, :] - gt_img[:, :, 1:, :]), 1, keepdim=True)

    d_dx *= torch.exp(-i_dx)
    d_dy *= torch.exp(-i_dy)

    errors = F.pad(d_dx, pad=(0, 1), mode='constant', value=0) + F.pad(d_dy, pad=(0, 0, 0, 1), mode='constant', value=0)
    errors = errors.view(n, pc, h, w)
    return errors


class ReconstructionLoss:
    def __init__(self, config, use_automasking=False):
        super().__init__()
        self.criterion_str = config.get("criterion", "l2")
        if self.criterion_str == "l2":
            self.rgb_coarse_crit = torch.nn.MSELoss(reduction="none")
            self.rgb_fine_crit = torch.nn.MSELoss(reduction="none")
        elif self.criterion_str == "l1":
            self.rgb_coarse_crit = torch.nn.L1Loss(reduction="none")
            self.rgb_fine_crit = torch.nn.L1Loss(reduction="none")
        elif self.criterion_str == "l1+ssim":
            self.rgb_coarse_crit = compute_errors_l1ssim
            self.rgb_fine_crit = compute_errors_l1ssim
        self.invalid_policy = config.get("invalid_policy", "strict")
        assert self.invalid_policy in ["strict", "weight_guided", "weight_guided_diverse", None, "none"]
        self.ignore_invalid = self.invalid_policy is not None and self.invalid_policy != "none"
        self.lambda_coarse = config.get("lambda_coarse", 1)
        self.lambda_fine = config.get("lambda_fine", 1)

        self.use_automasking = use_automasking

        self.median_thresholding = config.get("median_thresholding", False)

        self.alpha_reg_reduction = config.get("alpha_reg_reduction", "ray")
        self.alpha_reg_fraction = config.get("alpha_reg_fraction", 1 / 8)

        if self.alpha_reg_reduction not in ("ray", "slice"):
            raise ValueError(f"Unknown reduction for alpha regularization: {self.alpha_reg_reduction}")

        # ============================== my losses ================================================
        self.lambda_depth_supervision = config.get("lambda_depth_supervision", 0)
        self.lambda_depth_recon = config.get("lambda_depth_recon", 0)
        self.lambda_temporal_alignment = config.get("lambda_temporal_alignment", 0)

        self.depth_recon_version = config.get("depth_recon_version", 1)

    @staticmethod
    def get_loss_metric_names():
        return ["loss", "loss_rgb_coarse", "loss_rgb_fine", "loss_depth_recon", "loss_temp_align"]

    def __call__(self, data):
        loss_dict = {}

        loss_coarse_all = 0
        loss_fine_all = 0
        loss_depth_super = 0
        loss_depth_recon = 0
        loss_temp_align = 0
        loss = 0

        coarse_0 = data["coarse"][0]  # dict of {'rgb', 'depth', 'invalid', 'weights', 'alphas', 'rgb_samps'}
        fine_0 = data["fine"][0]
        invalid_coarse = coarse_0["invalid"]
        invalid_fine = fine_0["invalid"]

        weights_coarse = coarse_0["weights"]
        weights_fine = fine_0["weights"]

        if self.invalid_policy == "strict":
            # Consider all rays invalid where there is at least one invalidly sampled color
            invalid_coarse = torch.all(torch.any(invalid_coarse > .5, dim=-2), dim=-1).unsqueeze(-1)
            invalid_fine = torch.all(torch.any(invalid_fine > .5, dim=-2), dim=-1).unsqueeze(-1)
        elif self.invalid_policy == "weight_guided":  # √
            # Integrate invalid indicator function over the weights.
            # It is invalid if > 90% of the mass is invalid. (Arbitrary threshold)
            invalid_coarse = torch.all(
                (invalid_coarse.to(torch.float32) * weights_coarse.unsqueeze(-1)).sum(-2) > .9, dim=-1,
                keepdim=True)
            invalid_fine = torch.all((invalid_fine.to(torch.float32) * weights_fine.unsqueeze(-1)).sum(-2) > .9,
                                     dim=-1, keepdim=True)
        elif self.invalid_policy == "weight_guided_diverse":
            # We now also consider, whether there is enough variance in the ray colors to give a meaningful supervision signal.
            rgb_samps_c = coarse_0["rgb_samps"]
            rgb_samps_f = fine_0["rgb_samps"]
            ray_std_c = torch.std(rgb_samps_c, dim=-3).mean(-1)
            ray_std_f = torch.std(rgb_samps_f, dim=-3).mean(-1)

            # Integrate invalid indicator function over the weights. It is invalid if > 90% of the mass is invalid. (Arbitrary threshold)
            invalid_coarse = torch.all(
                ((invalid_coarse.to(torch.float32) * weights_coarse.unsqueeze(-1)).sum(-2) > .9) | (
                        ray_std_c < 0.01), dim=-1, keepdim=True)
            invalid_fine = torch.all(
                ((invalid_fine.to(torch.float32) * weights_fine.unsqueeze(-1)).sum(-2) > .9) | (ray_std_f < 0.01),
                dim=-1, keepdim=True)
        elif self.invalid_policy == "none":
            invalid_coarse = torch.zeros_like(
                torch.all(torch.any(invalid_coarse > .5, dim=-2), dim=-1).unsqueeze(-1), dtype=torch.bool)
            invalid_fine = torch.zeros_like(torch.all(torch.any(invalid_fine > .5, dim=-2), dim=-1).unsqueeze(-1),
                                            dtype=torch.bool)
        else:
            raise NotImplementedError

        coarse = data["coarse"][0]
        fine = data["fine"][0]

        rgb_coarse = coarse["rgb"]
        rgb_fine = fine["rgb"]
        rgb_gt = data["rgb_gt"]

        if self.use_automasking:  # false
            thresh_gt = rgb_gt[..., -1:]
            rgb_coarse = rgb_coarse[..., :-1]
            rgb_fine = rgb_fine[..., :-1]
            rgb_gt = rgb_gt[..., :-1]

        rgb_gt = rgb_gt.unsqueeze(-2)

        using_fine = len(fine) > 0

        b, pc, h, w, nv, c = rgb_coarse.shape

        # Take minimum across all reconstructed views
        rgb_loss = self.rgb_coarse_crit(rgb_coarse, rgb_gt)  # [16, 64, 8, 8, 4, 1]
        rgb_loss = rgb_loss.amin(-2)

        if self.use_automasking:
            rgb_loss = torch.min(rgb_loss, thresh_gt)

        if self.ignore_invalid:
            rgb_loss = rgb_loss * (1 - invalid_coarse.to(torch.float32))

        if self.median_thresholding:
            threshold = torch.median(rgb_loss.view(b, -1), dim=-1)[0].view(-1, 1, 1, 1, 1)
            rgb_loss = rgb_loss[rgb_loss <= threshold]

        rgb_loss = rgb_loss.mean()

        loss_coarse_all += rgb_loss.item() * self.lambda_coarse  # lambda_coarse=1
        if using_fine:
            fine_loss = self.rgb_fine_crit(rgb_fine, rgb_gt)
            fine_loss = fine_loss.amin(-2)

            if self.use_automasking:
                fine_loss = torch.min(fine_loss, thresh_gt)

            if self.ignore_invalid:
                fine_loss = fine_loss * (1 - invalid_fine.to(torch.float32))

            if self.median_thresholding:
                threshold = torch.median(fine_loss.view(b, -1), dim=-1)[0].view(-1, 1, 1, 1, 1)
                fine_loss = fine_loss[fine_loss <= threshold]

            fine_loss = fine_loss.mean()
            rgb_loss = rgb_loss * self.lambda_coarse + fine_loss * self.lambda_fine
            loss_fine_all += fine_loss.item() * self.lambda_fine
        else:
            loss_dict["loss_rgb_fine"] = 0

        loss += rgb_loss

        # my losses
        if self.lambda_depth_supervision > 0:
            # depth_gt = data['depths'][0]
            # pred_depth = data['predicted_depth'][0]
            depth_gt = torch.cat([data['depths'][0], data['depths'][1]], dim=0)
            pred_depth = torch.cat([data['predicted_depth'][0], data['predicted_depth'][1]], dim=0)

            mask = depth_gt > 0
            depth_gt = depth_gt[mask]
            pred_depth = pred_depth[mask]

            loss_depth_super = (depth_gt - pred_depth).abs().mean()
            loss += loss_depth_super * self.lambda_depth_supervision

        if self.lambda_depth_recon > 0:
            pred_depth = data['depth_reconstruction']['pred_depth']  # [b, patch_num, patch_size, patch_size]
            rendered_depth = data['depth_reconstruction']['rendered_depth']
            # version 1:
            if self.depth_recon_version == 1:
                loss_depth_recon = F.l1_loss(pred_depth, rendered_depth)
            # version 2: (weighted alignment)
            if self.depth_recon_version == 2:
                normed_pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
                weight_mask = 1 - normed_pred_depth
                loss_depth_recon = (weight_mask * (pred_depth - rendered_depth).abs()).mean()
            # version 3: (distribution alignment)
            if self.depth_recon_version == 3:
                normed_pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
                normed_rendered_depth = (rendered_depth - rendered_depth.min()) / (
                        rendered_depth.max() - rendered_depth.min())
                loss_depth_recon = F.l1_loss(normed_pred_depth, normed_rendered_depth)

            loss += loss_depth_recon * self.lambda_depth_recon

        if self.lambda_temporal_alignment > 0:
            # =============================== warping =============================================
            # using second frame (reference frame) to reconstruct the first frame (target frame)
            depth_t = data['predicted_depth'][0]
            depth_r = data['predicted_depth'][1]
            img_r = data["imgs"][1]
            img_t = data["imgs"][0]
            pose_t2r = torch.inverse(data["poses"][1]) @ data["poses"][0]
            pose_t2r = pose_t2r[:, :3, :]
            b, c, h, w = img_r.shape
            K = data["projs"][0].clone()
            K[:, 0, 2] += 1
            K[:, 1, 2] += 1
            K[:, 0, :] = K[:, 0, :] / 2. * w
            K[:, 1, :] = K[:, 1, :] / 2. * h
            warped_img_r2t, projected_depth, computed_depth, valid_mask = warp(img_r, depth_t, depth_r, pose_t2r, K)

            # photometric loss
            diff_img = (img_t - warped_img_r2t).abs().clamp(0, 1)
            diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)

            weight_mask = (1 - diff_depth)
            diff_img = diff_img * weight_mask
            temporal_loss = mean_on_mask(diff_img, valid_mask)
            geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)
            loss_temp_align = temporal_loss + geometry_consistency_loss

            loss += loss_temp_align * self.lambda_temporal_alignment

        loss_dict["loss_rgb_coarse"] = loss_coarse_all
        loss_dict["loss_rgb_fine"] = loss_fine_all
        loss_dict["loss_depth_super"] = loss_depth_super
        loss_dict["loss_depth_recon"] = loss_depth_recon
        loss_dict["loss_temp_align"] = loss_temp_align
        loss_dict["loss_invalid_ratio"] = invalid_coarse.float().mean().item()
        loss_dict["loss"] = loss.item()

        return loss, loss_dict
