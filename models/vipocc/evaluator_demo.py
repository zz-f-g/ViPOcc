from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.data_util import make_demo_dataset
from models.common.backbones.res_inv_depth import RID
from models.common.render import NeRFRenderer
from models.vipocc.model.ray_sampler import ImageRaySampler
from utils.inference_setup import get_pts_vox, build_voxels, render_profile
from utils.base_evaluator import base_evaluation
from utils.modules import DepthRefinement
from utils.plotting import color_tensor
from utils.projection_operations import distance_to_z
from plyfile import PlyData, PlyElement
from models.vipocc.model.models_bts import BTSNet

def save_plot(img, file_name):
    cv2.imwrite(file_name, cv2.cvtColor((img * 255).clip(max=255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def visualize_depth(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = (depth * 255.0).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
    return depth


class BTSWrapper(nn.Module):
    def __init__(self, renderer, config) -> None:
        super().__init__()

        self.renderer = renderer
        self.device = 'cuda'

        self.z_near = config["z_near"]
        self.z_far = config["z_far"]
        self.ray_batch_size = config["ray_batch_size"]
        self.sampler = ImageRaySampler(self.z_near, self.z_far)

        self.use_depth_branch = config.backbone.use_depth_branch
        if self.use_depth_branch:
            self.depth_refine = DepthRefinement()
            self.res_inv_depth = RID(resnet_layers=18)

        self.draw_occ = config.get("draw_occ", True)
        self.draw_rendered_depth = config.get("draw_rendered_depth", True)
        self.draw_pred_depth = config.get("draw_pred_depth", True)
        self.draw_pseudo_depth = config.get("draw_pseudo_depth", True)
        self.draw_bev = config.get("draw_bev", True)

        self.save_dir = Path(config["save_dir"])
        self.depth_scaling = config.get("depth_scaling", None)

        render_range_dict = config["render_range_dict"]
        self.X_RANGE = render_range_dict["x_range"]
        self.Y_RANGE = render_range_dict["y_range"]
        self.Z_RANGE = render_range_dict["z_range"]
        ppm = render_range_dict["ppm"]
        self.p_res_x = int(ppm * abs(self.X_RANGE[0] - self.X_RANGE[1]))
        self.p_res_z = int(ppm * abs(self.Z_RANGE[0] - self.Z_RANGE[1]))
        self.p_res_y = render_range_dict["p_res_y"]
        self.p_res = [self.p_res_z, self.p_res_x]

        y_steps = (1 - (torch.linspace(0, 1 - 1 / self.p_res_y, self.p_res_y) + 1 / (2 * self.p_res_y))).tolist()
        cmap = plt.cm.get_cmap("magma")
        self.y_to_color = (torch.tensor(list(map(cmap, y_steps)), device='cuda')[:, :3] * 255).to(torch.uint8)
        faces = [[0, 1, 2, 3], [0, 3, 7, 4], [2, 6, 7, 3], [1, 2, 6, 5], [0, 1, 5, 4], [4, 5, 6, 7]]
        self.faces_t = torch.tensor(faces, device='cuda')

        self.cam_incl_adjust = torch.tensor(
            [[1.0000000, 0.0000000, 0.0000000, 0],
             [0.0000000, 0.9961947, -0.0871557, 0],
             [0.0000000, 0.0871557, 0.9961947, 0],
             [0.0000000, 000000000, 0.0000000, 1]
             ],
            dtype=torch.float32).view(1, 4, 4)
        self.proj = torch.tensor([
            [0.7849, 0.0000, -0.0312, 0],
            [0.0000, 2.9391, 0.2701, 0],
            [0.0000, 0.0000, 1.0000, 0],
            [0.0000, 0.0000, 0.0000, 1],
        ], dtype=torch.float32).view(1, 4, 4)
        self.render_range_dict = config["render_range_dict"]

    @staticmethod
    def get_loss_metric_names():
        return ["loss", "loss_l2", "loss_mask", "loss_temporal"]

    def predict_depth(self, images, pseudo_depth=None):
        res_inv_depth = self.res_inv_depth(images)
        predicted_depth = self.depth_refine(pseudo_depth, res_inv_depth, vis=True)
        return predicted_depth

    @torch.no_grad()
    def forward(self, data):
        data = dict(data)
        images = torch.stack(data["imgs"], dim=1)[:, :1]  # n, 1, c, h, w
        poses = torch.eye(4).view(1, 1, 4, 4).to(self.device)
        projs = self.proj.view(1, 1, 4, 4).to(self.device)[:, :, :3, :3]

        # save rgb image
        self.save_dir.mkdir(exist_ok=True, parents=True)
        img = images[0, 0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
        save_rgb_path = str(self.save_dir / f"{data['seq'][0]}_{int(data['img_id'][0].cpu().numpy()):0>10d}_rgb.png")
        cv2.imwrite(save_rgb_path, cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        if self.draw_occ:
            self.renderer.net.encode(images, projs, poses, ids_encoder=[0], ids_render=[0])
            q_pts = get_pts_vox(self.X_RANGE, self.Y_RANGE, self.Z_RANGE, self.p_res[1], self.p_res_y, self.p_res[0])
            q_pts = q_pts.to(self.device).reshape(1, -1, 3)
            _, invalid, sigmas = self.renderer.net.forward(q_pts)

            alphas = sigmas
            alphas = alphas.reshape(1, 1, self.p_res[1], self.p_res_y, self.p_res[0])  # (x, y, z)
            alphas_mean = F.avg_pool3d(alphas, kernel_size=2, stride=1, padding=0)
            is_occupied = alphas_mean.squeeze() > .5
            verts, faces, colors = build_voxels(is_occupied.nonzero(), self.p_res[1], self.p_res_y, self.p_res[0],
                                                q_pts.squeeze(0).T,
                                                self.y_to_color, self.faces_t)

            verts_el, faces_el = self.parse_ply(verts, faces, colors)

            save_ply_path = save_rgb_path.replace("_rgb.png", "_voxel.ply")
            PlyData([verts_el, faces_el]).write(save_ply_path)

        if self.draw_rendered_depth:
            n, v, c, h, w = images.shape
            to_base_pose = torch.inverse(poses[:, :1, :, :])
            poses = to_base_pose.expand(-1, v, -1, -1) @ poses
            ids_encoder = [0]
            self.renderer.net.encode(images, projs, poses, ids_encoder=ids_encoder, ids_render=ids_encoder)
            all_rays, all_rgb_gt = self.sampler.sample(images * .5 + .5, poses, projs)
            rendered_depth = self.renderer(all_rays, depth_only=True)
            rendered_depth = rendered_depth.reshape(n, -1, h, w)  # [1,1,192,640]
            rendered_depth = distance_to_z(rendered_depth, projs)
            depth = rendered_depth

            save_disp_r_path = save_rgb_path.replace("_rgb.png", "_disp_r.png")

            # save disp images
            depth = depth[0, 0].cpu().numpy()
            disp = 1 / depth
            disp_save = visualize_depth(disp)
            cv2.imwrite(save_disp_r_path, disp_save)

        if self.draw_pred_depth:
            if not self.use_depth_branch:
                raise ValueError('Model has no direct depth predictions. Please select other models or set '
                                 'use_depth_branch = True. ')
            depth = self.predict_depth(images[:, 0], data["pseudo_depth"][0])
            save_disp_p_path = save_rgb_path.replace("_rgb.png", "_disp_p.png")

            # save disp images
            depth = depth[0, 0].cpu().numpy()
            disp = 1 / depth
            disp_save = visualize_depth(disp)
            cv2.imwrite(save_disp_p_path, disp_save)

        if self.draw_pseudo_depth:
            save_pseudo_disp_path = save_rgb_path.replace("_rgb.png", "_disp_pseudo.png")
            pseudo_depth_save = data["pseudo_depth"][0][0, 0].cpu().numpy()
            pseudo_disp_save = 1 / pseudo_depth_save
            pseudo_disp_save = visualize_depth(pseudo_disp_save)
            cv2.imwrite(save_pseudo_disp_path, pseudo_disp_save)

        if self.draw_bev:
            self.render_range_dict["y_range"] = [0, 0.75]
            profile = render_profile(self.renderer.net, self.cam_incl_adjust, render_range_dict=self.render_range_dict)

            save_bev_path = save_rgb_path.replace("_rgb.png", "_bev.png")
            save_plot(color_tensor(profile.cpu(), "magma", norm=True).numpy(), save_bev_path)

        return data

    def parse_ply(self, verts, faces, colors):
        verts = list(map(tuple, verts))
        verts_data = np.array(verts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        face_data = np.array(faces, dtype='i4')
        color_data = np.array(colors, dtype='u1')
        ply_faces = np.empty(len(faces),
                             dtype=[('vertex_indices', 'i4', (4,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

        ply_faces['vertex_indices'] = face_data
        ply_faces["red"] = color_data[:, 0]
        ply_faces["green"] = color_data[:, 1]
        ply_faces["blue"] = color_data[:, 2]

        verts_el = PlyElement.describe(verts_data, "vertex")
        faces_el = PlyElement.describe(ply_faces, "face")

        return verts_el, faces_el


def evaluation(local_rank, config):
    return base_evaluation(local_rank, config, get_dataflow, initialize, get_metrics)


def get_dataflow(config):
    test_dataset = make_demo_dataset(config["data"])
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=False)
    return test_loader


def get_metrics(config, device):
    return {}


def initialize(config: dict, logger=None):
    arch = config["model_conf"].get("arch", "BTSNet")
    net = globals()[arch](config["model_conf"])
    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()

    model = BTSWrapper(
        renderer,
        config["model_conf"]
    )

    return model
