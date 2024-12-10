import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.data_util import make_test_dataset
from models.common.backbones import Monodepth2
from models.common.backbones.res_depth import ResidualDepth
from models.common.backbones.res_inv_depth import RID
from models.common.render import NeRFRenderer
from models.vipocc.model.models_bts import BTSNet
from models.vipocc.model.ray_sampler import ImageRaySampler
from utils.base_evaluator import base_evaluation
from utils.metrics import MeanMetric
from utils.modules import DepthRefinement
from utils.projection_operations import distance_to_z

class BTSWrapper(nn.Module):
    def __init__(self, renderer, config) -> None:
        super().__init__()

        self.renderer = renderer

        self.z_near = config["z_near"]
        self.z_far = config["z_far"]
        self.ray_batch_size = config["ray_batch_size"]
        self.sampler = ImageRaySampler(self.z_near, self.z_far)

        self.use_depth_branch = config.backbone.use_depth_branch
        if self.use_depth_branch:
            # self.depth_sampler = DepthRaySampler(self.z_near, self.z_far, self.patch_size, self.ray_batch_size // 2)
            self.depth_refine = DepthRefinement()
            self.rid_ablation = config.backbone.get("rid_ablation", 'rid')
            if self.rid_ablation == 'res_depth':
                self.res_depth = ResidualDepth(resnet_layers=18)
            elif self.rid_ablation == 'depth':
                self.depth_model = Monodepth2(resnet_layers=18, d_out=1)
            else:
                self.res_inv_depth = RID(resnet_layers=18)

        self.depth_scaling = config.get("depth_scaling", None)
        self.eval_rendered_depth = config.get("eval_rendered_depth", True)

    @staticmethod
    def get_loss_metric_names():
        return ["loss", "loss_l2", "loss_mask", "loss_temporal"]

    def predict_depth(self, images, pseudo_depth=None):
        if self.rid_ablation == 'depth':
            predicted_depth, _ = self.depth_model(images)
        elif self.rid_ablation == 'res_depth':
            predicted_depth = self.res_depth(images) + pseudo_depth
        elif self.rid_ablation == 'rid':
            res_inv_depth = self.res_inv_depth(images)
            predicted_depth = self.depth_refine(pseudo_depth, res_inv_depth)
        else:
            raise NotImplementedError
        return predicted_depth

    def forward(self, data):
        data = dict(data)
        images = torch.stack(data["imgs"], dim=1)[:, :1]  # n, 1, c, h, w
        poses = torch.stack(data["poses"], dim=1)[:, :1]  # n, 1, 4, 4 w2c
        projs = torch.stack(data["projs"], dim=1)[:, :1]  # n, 1, 4, 4 (-1, 1)

        if self.eval_rendered_depth:
            n, v, c, h, w = images.shape
            to_base_pose = torch.inverse(poses[:, :1, :, :])
            poses = to_base_pose.expand(-1, v, -1, -1) @ poses
            ids_encoder = [0]
            self.renderer.net.encode(images, projs, poses, ids_encoder=ids_encoder, ids_render=ids_encoder)
            all_rays, all_rgb_gt = self.sampler.sample(images * .5 + .5, poses, projs)
            rendered_depth = self.renderer(all_rays, depth_only=True)
            rendered_depth = rendered_depth.reshape(n, -1, h, w)  # [1,1,192,640]
            rendered_depth = distance_to_z(rendered_depth, projs)
            data["predicted_depth"] = rendered_depth
        else:
            data["predicted_depth"] = self.predict_depth(images[:, 0], data["pseudo_depth"][0])

        data.update(self.compute_depth_metrics(data))
        return data

    def compute_depth_metrics(self, data):
        depth_gt = data["depths"][0]
        # depth_pred = data["fine"][0]["depth"][:, :1]
        depth_pred = data["predicted_depth"]

        # vis_tensor(1 / depth_pred[0, 0], 'disp')
        # vis_tensor(1 / data['pseudo_depth'][0], 'pseudo_disp')
        # vis_tensor(data['imgs'][0][0], 'rgb')
        depth_pred = F.interpolate(depth_pred, depth_gt.shape[-2:])
        if self.depth_scaling == "median":
            mask = depth_gt > 0
            scaling = torch.median(depth_gt[mask]) / torch.median(depth_pred[mask])
            depth_pred = scaling * depth_pred
        elif self.depth_scaling == "l2":
            mask = depth_gt > 0
            depth_pred = depth_pred
            depth_gt_ = depth_gt[mask]
            depth_pred_ = depth_pred[mask]
            depth_pred_ = torch.stack((depth_pred_, torch.ones_like(depth_pred_)), dim=-1)
            x = torch.linalg.lstsq(depth_pred_.to(torch.float32),
                                   depth_gt_.unsqueeze(-1).to(torch.float32)).solution.squeeze()
            depth_pred = depth_pred * x[0] + x[1]

        depth_pred = torch.clamp(depth_pred, 1e-3, 80)
        mask = depth_gt != 0

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]

        thresh = torch.maximum((depth_gt / depth_pred), (depth_pred / depth_gt))
        a1 = (thresh < 1.25).to(torch.float)
        a2 = (thresh < 1.25 ** 2).to(torch.float)
        a3 = (thresh < 1.25 ** 3).to(torch.float)
        a1 = a1.mean()
        a2 = a2.mean()
        a3 = a3.mean()

        rmse = (depth_gt - depth_pred) ** 2
        rmse = rmse.mean() ** .5

        rmse_log = (torch.log(depth_gt) - torch.log(depth_pred)) ** 2
        rmse_log = rmse_log.mean() ** .5

        abs_rel = torch.abs(depth_gt - depth_pred) / depth_gt
        abs_rel = abs_rel.mean()

        sq_rel = ((depth_gt - depth_pred) ** 2) / depth_gt
        sq_rel = sq_rel.mean()

        metrics_dict = {
            "abs_rel": abs_rel,
            "sq_rel": sq_rel,
            "rmse": rmse,
            "rmse_log": rmse_log,
            "a1": a1,
            "a2": a2,
            "a3": a3
        }
        return metrics_dict


def evaluation(local_rank, config):
    return base_evaluation(local_rank, config, get_dataflow, initialize, get_metrics)


def get_dataflow(config):
    test_dataset = make_test_dataset(config["data"])
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=config["num_workers"], shuffle=False,
                             drop_last=False)

    return test_loader


def get_metrics(config, device):
    names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    metrics = {name: MeanMetric((lambda n: lambda x: x["output"][n])(name), device) for name in names}
    return metrics


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


def visualize(engine: Engine, logger: TensorboardLogger, step: int, tag: str):
    pass
