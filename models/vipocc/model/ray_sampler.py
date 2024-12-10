import torch
from omegaconf import ListConfig

from models.common.util import util, unproj_map


class RaySampler:
    def sample(self, images, poses, projs):
        raise NotImplementedError

    def reconstruct(self, render_dict):
        raise NotImplementedError


class RandomRaySampler(RaySampler):
    def __init__(self, ray_batch_size, z_near, z_far, channels=3):
        self.ray_batch_size = ray_batch_size
        self.z_near = z_near
        self.z_far = z_far
        self.channels = channels

    def sample(self, images, poses, projs):
        n, v, c, h, w = images.shape

        all_rgb_gt = []
        all_rays = []

        for n_ in range(n):
            focals = projs[n_, :, [0, 1], [0, 1]]
            centers = projs[n_, :, [0, 1], [2, 2]]

            rays = util.gen_rays(poses[n_].view(-1, 4, 4), w, h, focal=focals, c=centers, z_near=self.z_near,
                                 z_far=self.z_far).view(-1, 8)

            rgb_gt = images[n_].view(-1, self.channels, h, w)
            rgb_gt = (rgb_gt.permute(0, 2, 3, 1).contiguous().reshape(-1, self.channels))

            pix_inds = torch.randint(0, v * h * w, (self.ray_batch_size,))

            rgb_gt = rgb_gt[pix_inds]
            rays = rays[pix_inds]

            all_rgb_gt.append(rgb_gt)
            all_rays.append(rays)

        all_rgb_gt = torch.stack(all_rgb_gt)
        all_rays = torch.stack(all_rays)

        return all_rays, all_rgb_gt

    def reconstruct(self, render_dict, channels=None):
        coarse = render_dict["coarse"]
        fine = render_dict["fine"]

        if channels is None:
            channels = self.channels

        c_rgb = coarse["rgb"]  # n, n_pts, v * 3
        c_weights = coarse["weights"]
        c_depth = coarse["depth"]
        c_invalid = coarse["invalid"]

        f_rgb = fine["rgb"]  # n, n_pts, v * 3
        f_weights = fine["weights"]
        f_depth = fine["depth"]
        f_invalid = fine["invalid"]

        rgb_gt = render_dict["rgb_gt"]

        n, n_pts, v_c = c_rgb.shape
        v = v_c // self.channels
        c_n_smps = c_weights.shape[-1]
        f_n_smps = f_weights.shape[-1]

        coarse["rgb"] = c_rgb.view(n, n_pts, v, channels)
        coarse["weights"] = c_weights.view(n, n_pts, c_n_smps)
        coarse["depth"] = c_depth.view(n, n_pts)
        coarse["invalid"] = c_invalid.view(n, n_pts, c_n_smps, v)

        fine["rgb"] = f_rgb.view(n, n_pts, v, channels)
        fine["weights"] = f_weights.view(n, n_pts, f_n_smps)
        fine["depth"] = f_depth.view(n, n_pts)
        fine["invalid"] = f_invalid.view(n, n_pts, f_n_smps, v)

        if "alphas" in coarse:
            c_alphas = coarse["alphas"]
            f_alphas = fine["alphas"]
            coarse["alphas"] = c_alphas.view(n, n_pts, c_n_smps)
            fine["alphas"] = f_alphas.view(n, n_pts, f_n_smps)

        if "z_samps" in coarse:
            c_z_samps = coarse["z_samps"]
            f_z_samps = fine["z_samps"]
            coarse["z_samps"] = c_z_samps.view(n, n_pts, c_n_smps)
            fine["z_samps"] = f_z_samps.view(n, n_pts, f_n_smps)

        if "rgb_samps" in coarse:
            c_rgb_samps = coarse["rgb_samps"]
            f_rgb_samps = fine["rgb_samps"]
            coarse["rgb_samps"] = c_rgb_samps.view(n, n_pts, c_n_smps, v, channels)
            fine["rgb_samps"] = f_rgb_samps.view(n, n_pts, f_n_smps, v, channels)

        render_dict["coarse"] = coarse
        render_dict["fine"] = fine
        render_dict["rgb_gt"] = rgb_gt.view(n, n_pts, channels)

        return render_dict


class PatchRaySampler(RaySampler):
    def __init__(self, ray_batch_size, z_near, z_far, patch_size, channels=3):
        self.ray_batch_size = ray_batch_size  # 4096
        self.z_near = z_near  # 3
        self.z_far = z_far  # 80
        if isinstance(patch_size, int):
            self.patch_size_x, self.patch_size_y = patch_size, patch_size
        elif isinstance(patch_size, tuple) or isinstance(patch_size, list) or isinstance(patch_size, ListConfig):
            self.patch_size_y = patch_size[0]
            self.patch_size_x = patch_size[1]
        else:
            raise ValueError(f"Invalid format for patch size")
        self.channels = channels
        assert (ray_batch_size % (self.patch_size_x * self.patch_size_y)) == 0
        self._patch_count = self.ray_batch_size // (self.patch_size_x * self.patch_size_y)  # 64

        # ========================================= init unprojection map =========================================
        self.batch_size = None
        self.v = None
        self.h = None
        self.w = None
        self.cam_unproj_map = None
        self.cam_nears = None
        self.cam_fars = None
        self.device = None

    def init_ray(self, images, projs):
        n, v, c, h, w = images.shape
        device = images.device
        # assert 相机focal center 恒定
        focal = projs[0, :, [0, 1], [0, 1]]  # [4,2]
        center = projs[0, :, [0, 1], [2, 2]]  # [4,2]

        self.batch_size = n
        self.v = v
        self.h = h
        self.w = w
        self.device = device
        # cam_unproj_map: [1,1,h,w,3,1]
        self.cam_unproj_map = unproj_map(w, h, focal, center, device=device, norm_dir=True).unsqueeze(-1).unsqueeze(0)
        self.cam_nears = torch.tensor(self.z_near, device=device).view(1, 1, 1, 1, 1).expand(n, v, h, w, 1)
        self.cam_fars = torch.tensor(self.z_far, device=device).view(1, 1, 1, 1, 1).expand(n, v, h, w, 1)

    def sample(self, images, poses, projs, patch_anchors=None):
        """
        Sample rays from a set of images (N_loss) by random patches. Each patch offers `patch_size*patch_size` rays. The
        `ray_batch_size` sets the total number of sampled rays.
        :param images: (N, 4, C, H, W)
        :param poses: (N, 4, 4, 4)
        :param projs: (N, 4, 3, 3)
        :param patch_anchors:

        :return all_rays: (N, 4096, 8), 8 for (3 cam_centers, 3 cam_raydir, 1 cam_nears, 1 cam_fars)
        :return all_rgb_gt: (N, 4096, 3)
        """
        if self.h is None:
            self.init_ray(images, projs)
        n, v, c, h, w = images.shape
        images = images.permute(0, 1, 3, 4, 2)  # [B,4,h,w,3]
        cam_centers = poses[:, :, None, None, :3, 3]  # [B,4,1,1,3]
        cam_centers = cam_centers.expand(-1, -1, h, w, -1)  # [B,4,h,w,3]
        # [B,4,1,1,3,3] x [1,1,h,w,3,1]
        cam_raydir = torch.matmul(poses[:, :, None, None, :3, :3], self.cam_unproj_map)[..., 0]  # [B,4,h,w,3]
        rays = torch.cat((cam_centers,
                          cam_raydir,
                          self.cam_nears,  # [b,4,h,w,1]
                          self.cam_fars), dim=-1)  # [b,4,h,w,8]

        if patch_anchors is None:
            sample_indices = torch.hstack([torch.randint(0, v, (self._patch_count, 1), device=self.device),
                                           torch.randint(0, h - self.patch_size_y, (self._patch_count, 1),
                                                         device=self.device),
                                           torch.randint(0, w - self.patch_size_x, (self._patch_count, 1),
                                                         device=self.device)]
                                          )
            sample_rgb_gt = [
                images[:, v, y:y + self.patch_size_y, x:x + self.patch_size_x, :] for v, y, x in sample_indices
            ]
            sample_rgb_gt = torch.stack(sample_rgb_gt, dim=1).view(n, -1, 3)  # -->[B, 64, 8, 8, 3]-->[B, 4096, 3]

            sample_rays = [
                rays[:, v, y:y + self.patch_size_y, x:x + self.patch_size_x, :] for v, y, x in sample_indices
            ]
            sample_rays = torch.stack(sample_rays, dim=1).view(n, -1, 8)  # -->[B, 64, 8, 8, 8]-->[B, 4096, 8]
        else:
            sample_indices = torch.stack([torch.randint(2, v, (n, self._patch_count // 2), device=self.device),
                                          torch.randint(0, h - self.patch_size_y, (n, self._patch_count // 2),
                                                        device=self.device),
                                          torch.randint(0, w - self.patch_size_x, (n, self._patch_count // 2),
                                                        device=self.device)], dim=-1)
            all_sample_indices = torch.cat([sample_indices, patch_anchors], dim=1)
            all_rgb_gt = []
            all_rays = []
            for n_ in range(n):  # Iterate over batch
                sample_indices = all_sample_indices[n_]
                sample_rgb_gt = [
                    images[n_][v, y:y + self.patch_size_y, x:x + self.patch_size_x, :].reshape(-1, self.channels)
                    for v, y, x in sample_indices
                ]
                sample_rgb_gt = torch.cat(sample_rgb_gt, dim=0)

                sample_rays = [
                    rays[v, y:y + self.patch_size_y, x:x + self.patch_size_x, :].reshape(-1, 8)
                    for v, y, x in sample_indices
                ]
                sample_rays = torch.cat(sample_rays, dim=0)

                all_rgb_gt.append(sample_rgb_gt)
                all_rays.append(sample_rays)
            sample_rgb_gt = [
                images[:, v, y:y + self.patch_size_y, x:x + self.patch_size_x, :] for v, y, x in sample_indices
            ]
            sample_rgb_gt = torch.stack(sample_rgb_gt, dim=1).view(n, -1, 3)  # -->[B, 64, 8, 8, 3]-->[B, 4096, 3]

            sample_rays = [
                rays[:, v, y:y + self.patch_size_y, x:x + self.patch_size_x, :] for v, y, x in sample_indices
            ]
            sample_rays = torch.stack(sample_rays, dim=1).view(n, -1, 8)  # -->[B, 64, 8, 8, 8]-->[B, 4096, 8]

        return sample_rays, sample_rgb_gt

    def reconstruct(self, render_dict, channels=None):
        coarse = render_dict["coarse"]
        fine = render_dict["fine"]

        if channels is None:
            channels = self.channels

        c_rgb = coarse["rgb"]  # n, n_pts, v * 3
        c_weights = coarse["weights"]
        c_depth = coarse["depth"]
        c_invalid = coarse["invalid"]

        f_rgb = fine["rgb"]  # n, n_pts, v * 3
        f_weights = fine["weights"]
        f_depth = fine["depth"]
        f_invalid = fine["invalid"]

        rgb_gt = render_dict["rgb_gt"]

        n, n_pts, v_c = c_rgb.shape
        v = v_c // channels
        c_n_smps = c_weights.shape[-1]
        f_n_smps = f_weights.shape[-1]
        # (This can be a different v from the sample method)

        coarse["rgb"] = c_rgb.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, v, channels)
        coarse["weights"] = c_weights.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, c_n_smps)
        coarse["depth"] = c_depth.view(n, self._patch_count, self.patch_size_y, self.patch_size_x)
        coarse["invalid"] = c_invalid.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, c_n_smps, v)

        fine["rgb"] = f_rgb.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, v, channels)
        fine["weights"] = f_weights.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, f_n_smps)
        fine["depth"] = f_depth.view(n, self._patch_count, self.patch_size_y, self.patch_size_x)
        fine["invalid"] = f_invalid.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, f_n_smps, v)

        if "alphas" in coarse:
            c_alphas = coarse["alphas"]
            f_alphas = fine["alphas"]
            coarse["alphas"] = c_alphas.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, c_n_smps)
            fine["alphas"] = f_alphas.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, f_n_smps)

        if "z_samps" in coarse:
            c_z_samps = coarse["z_samps"]
            f_z_samps = fine["z_samps"]
            coarse["z_samps"] = c_z_samps.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, c_n_smps)
            fine["z_samps"] = f_z_samps.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, f_n_smps)

        if "rgb_samps" in coarse:
            c_rgb_samps = coarse["rgb_samps"]
            f_rgb_samps = fine["rgb_samps"]
            coarse["rgb_samps"] = c_rgb_samps.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, c_n_smps,
                                                   v, channels)
            fine["rgb_samps"] = f_rgb_samps.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, f_n_smps,
                                                 v, channels)

        render_dict["coarse"] = coarse
        render_dict["fine"] = fine
        render_dict["rgb_gt"] = rgb_gt.view(n, self._patch_count, self.patch_size_y, self.patch_size_x, channels)

        return render_dict


class ImageRaySampler(RaySampler):
    def __init__(self, z_near, z_far, height=None, width=None, channels=3, norm_dir=True):
        self.z_near = z_near
        self.z_far = z_far
        self.height = height
        self.width = width
        self.channels = channels
        self.norm_dir = norm_dir

    def sample(self, images, poses, projs):
        n, v, _, _ = poses.shape

        if self.height is None:
            self.height, self.width = images.shape[-2:]

        all_rgb_gt = []
        all_rays = []

        for n_ in range(n):
            focals = projs[n_, :, [0, 1], [0, 1]]
            centers = projs[n_, :, [0, 1], [2, 2]]

            rays = util.gen_rays(poses[n_].view(-1, 4, 4), self.width, self.height, focal=focals, c=centers,
                                 z_near=self.z_near, z_far=self.z_far, norm_dir=self.norm_dir).view(-1, 8)
            all_rays.append(rays)

            if images is not None:
                rgb_gt = images[n_].view(-1, self.channels, self.height, self.width)
                rgb_gt = (rgb_gt.permute(0, 2, 3, 1).contiguous().reshape(-1, self.channels))
                all_rgb_gt.append(rgb_gt)

        all_rays = torch.stack(all_rays)
        if images is not None:
            all_rgb_gt = torch.stack(all_rgb_gt)
        else:
            all_rgb_gt = None

        return all_rays, all_rgb_gt

    def reconstruct(self, render_dict, channels=None):
        coarse = render_dict["coarse"]
        fine = render_dict["fine"]

        if channels is None:
            channels = self.channels

        c_rgb = coarse["rgb"]  # n, n_pts, v * 3
        c_weights = coarse["weights"]
        c_depth = coarse["depth"]
        c_invalid = coarse["invalid"]

        f_rgb = fine["rgb"]  # n, n_pts, v * 3
        f_weights = fine["weights"]
        f_depth = fine["depth"]
        f_invalid = fine["invalid"]

        n, n_pts, v_c = c_rgb.shape
        v_in = n_pts // (self.height * self.width)
        v_render = v_c // channels
        c_n_smps = c_weights.shape[-1]
        f_n_smps = f_weights.shape[-1]
        # (This can be a different v from the sample method)

        coarse["rgb"] = c_rgb.view(n, v_in, self.height, self.width, v_render, channels)
        coarse["weights"] = c_weights.view(n, v_in, self.height, self.width, c_n_smps)
        coarse["depth"] = c_depth.view(n, v_in, self.height, self.width)
        coarse["invalid"] = c_invalid.view(n, v_in, self.height, self.width, c_n_smps, v_render)

        fine["rgb"] = f_rgb.view(n, v_in, self.height, self.width, v_render, channels)
        fine["weights"] = f_weights.view(n, v_in, self.height, self.width, f_n_smps)
        fine["depth"] = f_depth.view(n, v_in, self.height, self.width)
        fine["invalid"] = f_invalid.view(n, v_in, self.height, self.width, f_n_smps, v_render)

        if "alphas" in coarse:
            c_alphas = coarse["alphas"]
            f_alphas = fine["alphas"]
            coarse["alphas"] = c_alphas.view(n, v_in, self.height, self.width, c_n_smps)
            fine["alphas"] = f_alphas.view(n, v_in, self.height, self.width, f_n_smps)

        if "z_samps" in coarse:
            c_z_samps = coarse["z_samps"]
            f_z_samps = fine["z_samps"]
            coarse["z_samps"] = c_z_samps.view(n, v_in, self.height, self.width, c_n_smps)
            fine["z_samps"] = f_z_samps.view(n, v_in, self.height, self.width, f_n_smps)

        if "rgb_samps" in coarse:
            c_rgb_samps = coarse["rgb_samps"]
            f_rgb_samps = fine["rgb_samps"]
            coarse["rgb_samps"] = c_rgb_samps.view(n, v_in, self.height, self.width, c_n_smps, v_render, channels)
            fine["rgb_samps"] = f_rgb_samps.view(n, v_in, self.height, self.width, f_n_smps, v_render, channels)

        render_dict["coarse"] = coarse
        render_dict["fine"] = fine

        if "rgb_gt" in render_dict:
            rgb_gt = render_dict["rgb_gt"]
            render_dict["rgb_gt"] = rgb_gt.view(n, v_in, self.height, self.width, channels)

        return render_dict


class DepthRaySampler(RaySampler):
    """
    Sample depths and rays.
    """

    def __init__(self, z_near, z_far, patch_size, ray_batch_size):
        self.patch_size = patch_size
        self.ray_batch_size = ray_batch_size
        assert (ray_batch_size % (patch_size ** 2)) == 0
        self._patch_count = ray_batch_size // (patch_size ** 2)

        self.z_near = z_near  # 3
        self.z_far = z_far  # 80
        self.channels = 1

        # ========================================= init unprojection map =========================================
        self.b = None
        self.v = None
        self.h = None
        self.w = None
        self.cam_unproj_map = None
        self.cam_nears = None
        self.cam_fars = None
        self.device = None

    def init_ray(self, depth, projs):
        n, _, h, w = depth.shape
        device = depth.device
        # assert 相机focal center 恒定
        focal = projs[0, :, [0, 1], [0, 1]]  # [4,2]
        center = projs[0, :, [0, 1], [2, 2]]  # [4,2]

        self.b = n
        self.h = h
        self.w = w
        self.device = device
        # cam_unproj_map: [1,1,h,w,3,1]
        self.cam_unproj_map = unproj_map(w, h, focal, center, device=device, norm_dir=True).unsqueeze(-1).unsqueeze(0)
        self.cam_nears = torch.tensor(self.z_near, device=device).view(1, 1, 1, 1, 1).expand(n, 1, h, w, 1)
        self.cam_fars = torch.tensor(self.z_far, device=device).view(1, 1, 1, 1, 1).expand(n, 1, h, w, 1)

        grid_x = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).expand(-1, -1, h, -1)
        grid_y = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).expand(-1, -1, -1, w)
        img_points = torch.cat((grid_x, grid_y, torch.ones_like(grid_x)), dim=1).expand(n, -1, -1, -1)
        self.img_points = img_points.permute(0, 2, 3, 1)  # [b,192,640,3]

    def sample(self, depth, poses, projs):
        """
        Sample rays from a set of images (N_loss) by random patches. Each patch offers `patch_size*patch_size` rays. The
        `ray_batch_size` sets the total number of sampled rays.
        :param depth: (N, 1, H, W)
        :param poses: (N, 4, 4, 4)
        :param projs: (N, 4, 3, 3)

        :return all_rays: (N, 4096, 8), 8 for (3 cam_centers, 3 cam_raydir, 1 cam_nears, 1 cam_fars)
        """
        if self.h is None:
            self.init_ray(depth, projs)
        n, _, h, w = depth.shape
        depth = depth.permute(0, 2, 3, 1)  # [B,h,w,1]

        sample_indices = torch.hstack(
            [torch.randint(0, h - self.patch_size, (self._patch_count, 1), device=self.device),
             torch.randint(0, w - self.patch_size, (self._patch_count, 1), device=self.device)])

        # no iteration version
        cam_centers = poses[:, :, None, None, :3, 3]  # [B,1,1,1,3]
        cam_centers = cam_centers.expand(-1, -1, h, w, -1)  # [B,1,h,w,3]
        # [B,4,1,1,3,3] x [1,1,h,w,3,1]
        cam_raydir = torch.matmul(poses[:, :, None, None, :3, :3], self.cam_unproj_map)[..., 0]  # [B,4,h,w,3]
        rays = torch.cat((cam_centers,
                          cam_raydir,
                          self.cam_nears,  # [b,4,h,w,1]
                          self.cam_fars), dim=-1)  # [b,4,h,w,8]

        sample_depth = [depth[:, y:y + self.patch_size, x:x + self.patch_size, :] for y, x in sample_indices]
        sample_depth = torch.stack(sample_depth, dim=1)

        sample_rays = [rays[:, 0, y:y + self.patch_size, x:x + self.patch_size, :] for y, x in sample_indices]
        sample_rays = torch.stack(sample_rays, dim=1).view(n, -1, 8)

        img_points = [self.img_points[:, y:y + self.patch_size, x:x + self.patch_size, :] for y, x in sample_indices]
        img_points = torch.stack(img_points, dim=1)  # [b,patch_num,patch_size,patch_size,3]

        inv_K = torch.inverse(projs[:, 0])  # [b,3,3]
        cam_points = (inv_K @ img_points.view(n, -1, 3).permute(0, 2, 1))  # [b,3,N]
        # cam_points = cam_points.permute(0, 2, 1).view(b,patch_num,patch_size,patch_size,3)
        factors = cam_points[:, 2, :] / torch.norm(cam_points, dim=1)  # [b,N]
        factors = factors.view(n, self._patch_count, self.patch_size, self.patch_size)

        return sample_rays, sample_depth, factors

    def reconstruct(self, rendered_depth):
        rendered_depth = rendered_depth.reshape(self.b, self._patch_count, self.patch_size, self.patch_size)
        return rendered_depth
