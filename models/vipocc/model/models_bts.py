import torch
import torch.nn.functional as F
from torch import nn

from models.common.backbones import make_backbone
from models.common.model.code import PositionalEncoding
from models.common.model.mlp_util import make_mlp

EPS = 1e-3


class BTSNet(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.d_min = conf.get("z_near")
        self.d_max = conf.get("z_far")

        self.learn_empty = conf.get("learn_empty", True)
        self.empty_empty = conf.get("empty_empty", False)
        self.inv_z = conf.get("inv_z", True)

        self.color_interpolation = conf.get("color_interpolation", "bilinear")
        self.code_mode = conf.get("code_mode", "z")
        if self.code_mode not in ["z", "distance"]:
            raise NotImplementedError(f"Unknown mode for positional encoding: {self.code_mode}")

        self.encoder = make_backbone(conf["backbone"])
        self.code_xyz = PositionalEncoding.from_conf(conf["code"], d_in=3)

        self.flip_augmentation = conf.get("flip_augmentation", False)

        self.return_sample_depth = conf.get("return_sample_depth", False)

        self.sample_color = conf.get("sample_color", True)

        d_in = self.encoder.latent_size + self.code_xyz.d_out
        d_out = 1 if self.sample_color else 4

        self._d_in = d_in
        self._d_out = d_out

        self.mlp_coarse = make_mlp(conf["mlp_coarse"], d_in, d_out=d_out)  # 103->64->1
        self.mlp_fine = make_mlp(conf["mlp_fine"], d_in, d_out=d_out, allow_empty=True)  # none

        if self.learn_empty:
            self.empty_feature = nn.Parameter(torch.randn((self.encoder.latent_size,), requires_grad=True))

    def encode(self, images, Ks, poses_c2w, ids_encoder=None, ids_render=None, images_alt=None):
        """
        Encode the input images using the `self.encoder` (resnet50 + modified mono2 dec), and save the results in
        `self.grid_f_features`.

        :param images: (B, V, 3, H, W), V = 8, values in [-1,1]
        :param Ks: (B, V, 3, 3)
        :param poses_c2w: (B, V, 4, 4)
        :param ids_encoder: [0]
        :param ids_render: list of 4, [1,2,5,6] or [0,3,4,7]
        :param images_alt: (B, V, 3, H, W), altered image, values in [0,1]

        """
        poses_w2c = torch.inverse(poses_c2w)

        if ids_encoder is None:
            images_encoder = images
            Ks_encoder = Ks
            poses_w2c_encoder = poses_w2c
            ids_encoder = list(range(len(images)))
        else:  # √
            images_encoder = images[:, ids_encoder]  # [B,1,3,192,640]
            Ks_encoder = Ks[:, ids_encoder]  # [B,1,3,3]
            poses_w2c_encoder = poses_w2c[:, ids_encoder]  # [B,1,4,4]

        if images_alt is not None:
            images = images_alt
        else:
            images = images * .5 + .5

        if ids_render is None:
            images_render = images
            Ks_render = Ks
            poses_w2c_render = poses_w2c
            ids_render = list(range(len(images)))
        else:  # √
            images_render = images[:, ids_render]  # (B, V/2, 3, H, W)
            Ks_render = Ks[:, ids_render]
            poses_w2c_render = poses_w2c[:, ids_render]

        n, nv, c, h, w = images_encoder.shape  # B,1,3,192,640
        c_l = self.encoder.latent_size  # 64

        if self.flip_augmentation and self.training:
            do_flip = (torch.rand(1) > .5).item()
        else:
            do_flip = False

        if do_flip:
            images_encoder = torch.flip(images_encoder, dims=(-1,))
        # image_latents_ms: [16,64,192,640]
        image_latents_ms = self.encoder(images_encoder.view(n * nv, c, h, w))

        if do_flip:
            image_latents_ms = torch.flip(image_latents_ms, dims=(-1,))

        _, _, h_, w_ = image_latents_ms.shape
        image_latents_ms = image_latents_ms.view(n, nv, c_l, h_, w_)  # [16, 1, 64, 192, 640]

        self.grid_f_features = image_latents_ms
        self.grid_f_Ks = Ks_encoder
        self.grid_f_poses_w2c = poses_w2c_encoder

        self.grid_c_imgs = images_render
        self.grid_c_Ks = Ks_render
        self.grid_c_poses_w2c = poses_w2c_render


    def sample_features(self, xyz, use_single_featuremap=True):
        n, n_pts, _ = xyz.shape
        n, nv, c, h, w = self.grid_f_features.shape

        xyz = xyz.unsqueeze(1)  # (n, 1, pts, 3)
        ones = torch.ones_like(xyz[..., :1])
        xyz = torch.cat((xyz, ones), dim=-1)  # (n, 1, pts, 4)
        xyz_projected = ((self.grid_f_poses_w2c[:, :nv, :3, :]) @ xyz.permute(0, 1, 3, 2))
        distance = torch.norm(xyz_projected, dim=-2).unsqueeze(-1)
        xyz_projected = (self.grid_f_Ks[:, :nv] @ xyz_projected).permute(0, 1, 3, 2)
        xy = xyz_projected[:, :, :, [0, 1]]
        z = xyz_projected[:, :, :, 2:3]

        xy = xy / z.clamp_min(EPS)
        invalid = (z <= EPS) | (xy[:, :, :, :1] < -1) | (xy[:, :, :, :1] > 1) | (xy[:, :, :, 1:2] < -1) | (
                xy[:, :, :, 1:2] > 1)

        if self.code_mode == "z":  # √
            # Get z into [-1, 1] range
            if self.inv_z:
                z = (1 / z.clamp_min(EPS) - 1 / self.d_max) / (1 / self.d_min - 1 / self.d_max)
            else:
                z = (z - self.d_min) / (self.d_max - self.d_min)
            z = 2 * z - 1
            xyz_projected = torch.cat((xy, z), dim=-1)
        elif self.code_mode == "distance":
            if self.inv_z:
                distance = (1 / distance.clamp_min(EPS) - 1 / self.d_max) / (1 / self.d_min - 1 / self.d_max)
            else:
                distance = (distance - self.d_min) / (self.d_max - self.d_min)
            distance = 2 * distance - 1
            xyz_projected = torch.cat((xy, distance), dim=-1)
        xyz_code = self.code_xyz(xyz_projected.view(n * nv * n_pts, -1)).view(n, nv, n_pts, -1)

        feature_map = self.grid_f_features[:, :nv]
        # These samples are from different scales
        if self.learn_empty:
            empty_feature_expanded = self.empty_feature.view(1, 1, 1, c).expand(n, nv, n_pts, c)

        sampled_features = F.grid_sample(feature_map.view(n * nv, c, h, w), xy.view(n * nv, 1, -1, 2), mode="bilinear",
                                         padding_mode="border", align_corners=False).view(n, nv, c, n_pts).permute(0, 1,
                                                                                                                   3, 2)

        if self.learn_empty:
            sampled_features[invalid.expand(-1, -1, -1, c)] = empty_feature_expanded[invalid.expand(-1, -1, -1, c)]

        sampled_features = torch.cat((sampled_features, xyz_code), dim=-1)  # [b,1,6250,64] + [b,1,6250,39]

        if use_single_featuremap:
            sampled_features = sampled_features.mean(dim=1)
            invalid = torch.any(invalid, dim=1)

        return sampled_features, invalid

    def sample_colors(self, xyz):
        """
        Sample colors from N_render images. N_render=4.

        :param xyz: [B, N, 3]
        :return: sampled colors [B, 4, N, 3]
        """
        # plt.hist(xyz[0, :, 1].detach().cpu().numpy(), bins=100)
        n, n_pts, _ = xyz.shape
        n, nv, c, h, w = self.grid_c_imgs.shape
        xyz = xyz.unsqueeze(1)  # (n, 1, pts, 3)
        ones = torch.ones_like(xyz[..., :1])
        xyz = torch.cat((xyz, ones), dim=-1)
        xyz_projected = self.grid_c_poses_w2c[:, :, :3, :] @ xyz.permute(0, 1, 3, 2)  # [b,v,3,6250]
        distance = torch.norm(xyz_projected, dim=-2).unsqueeze(-1)  # [b,v,6250,1]
        xyz_projected = (self.grid_c_Ks @ xyz_projected).permute(0, 1, 3, 2)  # =K[X,Y,Z]: [b,v,6250,3]
        xy = xyz_projected[:, :, :, [0, 1]]
        z = xyz_projected[:, :, :, 2:3]

        # This scales the x-axis into the right range.
        xy = xy / z.clamp_min(EPS)  # u,v  [B,4,6250,2]
        invalid = (z <= EPS) | (xy[:, :, :, :1] < -1) | (xy[:, :, :, :1] > 1) | (xy[:, :, :, 1:2] < -1) | (
                xy[:, :, :, 1:2] > 1)
        # sampled_colors: [B,4,6250,3]
        sampled_colors = F.grid_sample(self.grid_c_imgs.view(n * nv, c, h, w), xy.view(n * nv, 1, -1, 2),
                                       mode=self.color_interpolation, padding_mode="border", align_corners=False).view(
            n, nv, c, n_pts).permute(0, 1, 3, 2)
        assert not torch.any(torch.isnan(sampled_colors))

        if self.return_sample_depth:
            distance = distance.view(n, nv, n_pts, 1)
            sampled_colors = torch.cat((sampled_colors, distance), dim=-1)

        return sampled_colors, invalid

    def forward(self, xyz, coarse=True, viewdirs=None, far=False, only_density=False):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz: (B, 6250, 3)

        :return: rgb: (B, 6250, 4*3)
        :return: invalid: (B, 6250, 4)
        :return: sigma: (B, 6250, 1)
        """

        n, n_pts, _ = xyz.shape
        nv = self.grid_c_imgs.shape[1]

        # Sampled features all has shape: scales [n, n_pts, c + xyz_code]
        sampled_features, invalid_features = self.sample_features(xyz, use_single_featuremap=not only_density)
        sampled_features = sampled_features.reshape(n * n_pts, -1)  # [b,6250,103] --> [b*6250,103]

        mlp_input = sampled_features.view(n, n_pts, -1)

        # Run main NeRF network
        if coarse or self.mlp_fine is None:
            mlp_output = self.mlp_coarse(
                mlp_input,
                combine_inner_dims=(n_pts,),
            )  # [16,6250,1]
        else:
            mlp_output = self.mlp_fine(
                mlp_input,
                combine_inner_dims=(n_pts,),
            )

        # (n, pts, c) -> (n, n_pts, c)
        mlp_output = mlp_output.reshape(n, n_pts, self._d_out)

        if self.sample_color:
            sigma = mlp_output[..., :1]
            sigma = F.softplus(sigma)
            rgb, invalid_colors = self.sample_colors(xyz)  # (n, nv, pts, 3)
        else:
            sigma = mlp_output[..., :1]
            sigma = F.relu(sigma)
            rgb = mlp_output[..., 1:4].reshape(n, 1, n_pts, 3)
            rgb = F.sigmoid(rgb)
            invalid_colors = invalid_features.unsqueeze(-2)
            nv = 1

        if self.empty_empty:
            sigma[invalid_features[..., 0]] = 0

        if not only_density:
            _, _, _, c = rgb.shape
            rgb = rgb.permute(0, 2, 1, 3).reshape(n, n_pts, nv * c)  # (n, pts, nv * 3)
            invalid_colors = invalid_colors.permute(0, 2, 1, 3).reshape(n, n_pts, nv)

            invalid = invalid_colors | invalid_features  # Invalid features gets broadcasted to (n, n_pts, nv)
            invalid = invalid.to(rgb.dtype)
        else:
            rgb = torch.zeros((n, n_pts, nv * 3), device=sigma.device)
            invalid = invalid_features.to(sigma.dtype)
        return rgb, invalid, sigma  # rgb: (B, 6250, 4*3)
