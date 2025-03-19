from copy import copy

import ignite.distributed as idist
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from torch import nn
from torch import optim
from torch.nn import functional as F

from datasets.data_util import make_datasets
from datasets.kitti_raw.kitti_raw_dataset import KittiRawDataset
from models.common.backbones import Monodepth2
from models.common.backbones.res_depth import ResidualDepth
from models.common.backbones.res_inv_depth import RID
from models.common.model.scheduler import make_scheduler
from models.common.render import NeRFRenderer
from models.vipocc.model.image_processor import RGBProcessor
from models.vipocc.model.loss import ReconstructionLoss
from models.vipocc.model.models_bts import BTSNet
from models.vipocc.model.ray_sampler import *
from utils.base_trainer import base_training
from utils.metrics import MeanMetric
from utils.modules import DepthRefinement
from utils.projection_operations import distance_to_z
from utils.bbox import Bbox, bbox3d_collate_fn


class BTSWrapper(nn.Module):
    def __init__(self, renderer, config) -> None:
        super().__init__()

        self.renderer = renderer

        self.z_near = config["z_near"]  # 3
        self.z_far = config["z_far"]  # 80
        self.ray_batch_size = config["ray_batch_size"]  # 4096
        frames_render = config.get("n_frames_render", 2)  # 2
        self.frame_sample_mode = config.get("frame_sample_mode", "default")
        self.loss_from_single_img = config.get("loss_from_single_img", False)

        self.sample_mode = config.get("sample_mode", "random")  # patch
        self.patch_size = config.get("patch_size", 16)  # 8

        self.prediction_mode = config.get("prediction_mode", "multiscale")

        self.alternating_ratio = config.get("alternating_ratio", None)

        self.train_image_processor = RGBProcessor()
        self.val_image_processor = RGBProcessor()

        if type(frames_render) == int:
            self.frames_render = list(range(frames_render))
        else:
            self.frames_render = frames_render
        self.frames = self.frames_render

        if self.sample_mode == "random":
            self.train_sampler = RandomRaySampler(self.ray_batch_size, self.z_near, self.z_far,
                                                  channels=self.train_image_processor.channels)
        elif self.sample_mode == "patch":
            self.train_sampler = PatchRaySampler(self.ray_batch_size, self.z_near, self.z_far, self.patch_size,
                                                 channels=self.train_image_processor.channels)
        elif self.sample_mode == "image":
            self.train_sampler = ImageRaySampler(self.z_near, self.z_far, channels=self.train_image_processor.channels)
        else:
            raise NotImplementedError

        self.val_sampler = ImageRaySampler(self.z_near, self.z_far)

        self._counter = 0
        self.use_depth_branch = config.backbone.use_depth_branch
        if self.use_depth_branch:
            self.depth_sampler = DepthRaySampler(self.z_near, self.z_far, self.patch_size, self.ray_batch_size // 2)
            self.depth_refine = DepthRefinement()
            self.rid_ablation = config.backbone.get("rid_ablation", 'rid')
            if self.rid_ablation == 'res_depth':
                self.res_depth = ResidualDepth(resnet_layers=18)
            elif self.rid_ablation == 'depth':
                self.depth_model = Monodepth2(resnet_layers=18, d_out=1)
            else:
                self.res_inv_depth = RID(resnet_layers=18)

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
        data = dict(data)  # view = 8
        images = torch.stack(data["imgs"], dim=1)  # n, v, 3, h, w
        poses = torch.stack(data["poses"], dim=1)  # n, v, 4, 4 w2c
        projs = torch.stack(data["projs"], dim=1)  # n, v, 4, 4 (-1, 1)
        bboxes_3d = [
            (
                Bbox(
                    center=bbox_dict["center"],
                    whl=bbox_dict["whl"],
                    rotation=bbox_dict["rotation"],
                    label=bbox_dict["semanticId"],
                )
                if bbox_dict
                else None
            )
            for bbox_dict in data["3d_bboxes"]
        ]

        n, v, c, h, w = images.shape

        # Use first frame as keyframe
        to_base_pose = torch.inverse(poses[:, :1, :, :])
        poses = to_base_pose.expand(-1, v, -1, -1) @ poses

        if self.training:
            frame_perm = torch.randperm(v)
        else:
            frame_perm = torch.arange(v)

        ids_encoder = [0]
        # self.frames_render: [0,1]
        ids_render = torch.sort(frame_perm[[i for i in self.frames_render if i < v]]).values

        if self.training:
            if self.frame_sample_mode == "only":
                ids_loss = [0]
                ids_render = ids_render[ids_render != 0]
            elif self.frame_sample_mode == "not":
                frame_perm = torch.randperm(v - 1) + 1
                ids_loss = torch.sort(frame_perm[[i for i in self.frames_render if i < v - 1]]).values
                ids_render = [i for i in range(v) if i not in ids_loss]
            elif self.frame_sample_mode == "stereo":
                if frame_perm[0] < v // 2:
                    ids_loss = list(range(v // 2))
                    ids_render = list(range(v // 2, v))
                else:
                    ids_loss = list(range(v // 2, v))
                    ids_render = list(range(v // 2))
            elif self.frame_sample_mode == "mono":
                split_i = v // 2
                if frame_perm[0] < v // 2:
                    ids_loss = list(range(0, split_i, 2)) + list(range(split_i + 1, v, 2))
                    ids_render = list(range(1, split_i, 2)) + list(range(split_i, v, 2))
                else:
                    ids_loss = list(range(1, split_i, 2)) + list(range(split_i, v, 2))
                    ids_render = list(range(0, split_i, 2)) + list(range(split_i + 1, v, 2))
            elif self.frame_sample_mode == "kitti360-mono":  # √
                # optimized code
                if torch.rand(1) < 0.5:
                    ids_loss = torch.tensor([0, 3, 4, 7], dtype=torch.int, device=images.device)
                    ids_render = torch.tensor([1, 2, 5, 6], dtype=torch.int, device=images.device)
                else:
                    ids_loss = torch.tensor([1, 2, 5, 6], dtype=torch.int, device=images.device)
                    ids_render = torch.tensor([0, 3, 4, 7], dtype=torch.int, device=images.device)

            elif self.frame_sample_mode == "default":
                ids_loss = frame_perm[[i for i in range(v) if frame_perm[i] not in ids_render]]
            else:
                raise NotImplementedError
        else:
            ids_loss = torch.arange(v)
            ids_render = [0]

        if self.loss_from_single_img:
            ids_loss = ids_loss[:1]

        ip = self.train_image_processor if self.training else self.val_image_processor

        images_ip = ip(images)  # [-1,1] --> [0,1]

        # encode the first frame into grid_f_features: list of 4 [16, 1, 64, 192, 640]
        self.renderer.net.encode(images, projs, poses, ids_encoder=ids_encoder, ids_render=ids_render,
                                 images_alt=images_ip)

        # ================================ validation during training ================================
        if not self.training:
            if self.use_depth_branch:
                data["predicted_depth"] = self.predict_depth(images[:, 0], data["pseudo_depth"][0])
            else:
                sampler = self.val_sampler
                all_rays, _ = sampler.sample(images_ip[:, :1], poses[:, :1], projs[:, :1])
                rendered_depth = self.renderer(all_rays, depth_only=True).reshape(n, -1, h, w)  # [1,1,192,640]
                rendered_depth = distance_to_z(rendered_depth, projs[:, :1])
                data["predicted_depth"] = rendered_depth
            if len(data["depths"]) > 0:
                data.update(self.compute_depth_metrics(data))
            return data

        sampler = self.train_sampler  # if self.training else self.val_sampler
        anchors = data.get('samples', [])  # list of 4, [B, 32, 3]
        if len(anchors) != 0:
            anchors = torch.cat((anchors[ids_loss[0]], anchors[ids_loss[1]]), dim=1)  # b,32,3
            all_rays, all_rgb_gt = sampler.sample(images_ip[:, ids_loss], poses[:, ids_loss], projs[:, ids_loss],
                                                  anchors)
        else:
            # Sample rays from N_loss images by random patches. all_rays: [B, 4096, 8];   all_rgb_gt: [B, 4096, 3]
            all_rays, all_rgb_gt = sampler.sample(images_ip[:, ids_loss], poses[:, ids_loss], projs[:, ids_loss])
        render_dict = self.renderer(all_rays, bboxes_3d, want_weights=True, want_alphas=True, want_rgb_samps=True)

        data["fine"] = []
        data["coarse"] = []

        if "fine" not in render_dict:
            render_dict["fine"] = dict(render_dict["coarse"])

        render_dict["rgb_gt"] = all_rgb_gt
        render_dict["rays"] = all_rays

        render_dict = sampler.reconstruct(render_dict)

        data["fine"].append(render_dict["fine"])
        data["coarse"].append(render_dict["coarse"])
        data["rgb_gt"] = render_dict["rgb_gt"]
        data["rays"] = render_dict["rays"]

        data["z_near"] = torch.tensor(self.z_near, device=images.device)
        data["z_far"] = torch.tensor(self.z_far, device=images.device)

        if self.training and self.use_depth_branch:
            batch_size = images_ip.shape[0]
            images = torch.cat((images[:, 0], images[:, 1]), dim=0)
            pseudo_depth = torch.cat((data["pseudo_depth"][0], data["pseudo_depth"][1]), dim=0)
            predicted_depth = self.predict_depth(images, pseudo_depth)
            data.update({"predicted_depth": [predicted_depth[:batch_size], predicted_depth[batch_size:]]})

            # sampling depth
            all_rays, pred_depth, scaling_ratios = self.depth_sampler.sample(data['predicted_depth'][0], poses[:, :1],
                                                                             projs[:, :1])
            rendered_depth = self.renderer(all_rays, depth_only=True)
            rendered_depth = self.depth_sampler.reconstruct(rendered_depth)
            rendered_depth *= scaling_ratios

            render_dict = {}
            render_dict["pred_depth"] = pred_depth
            render_dict["rendered_depth"] = rendered_depth.unsqueeze(-1)

            data['depth_reconstruction'] = render_dict

        if self.training:
            self._counter += 1

        return data

    def compute_depth_metrics(self, data):
        depth_gt = data["depths"][0]
        depth_pred = data["predicted_depth"]

        depth_pred = F.interpolate(depth_pred, depth_gt.shape[-2:])
        depth_pred = torch.clamp(depth_pred, 1e-3, 80)
        mask = depth_gt != 0

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]

        thresh = torch.maximum((depth_gt / depth_pred), (depth_pred / depth_gt))
        a1 = (thresh < 1.25).to(torch.float).mean()
        a2 = (thresh < 1.25 ** 2).to(torch.float).mean()
        a3 = (thresh < 1.25 ** 3).to(torch.float).mean()

        rmse = (depth_gt - depth_pred) ** 2
        rmse = rmse.mean() ** .5

        rmse_log = (torch.log(depth_gt) - torch.log(depth_pred)) ** 2
        rmse_log = rmse_log.mean() ** .5

        abs_rel = torch.mean(torch.abs(depth_gt - depth_pred) / depth_gt)

        sq_rel = torch.mean(((depth_gt - depth_pred) ** 2) / depth_gt)

        metrics_dict = {
            "abs_rel": abs_rel.view(1),
            "sq_rel": sq_rel.view(1),
            "rmse": rmse.view(1),
            "rmse_log": rmse_log.view(1),
            "a1": a1.view(1),
            "a2": a2.view(1),
            "a3": a3.view(1)
        }
        return metrics_dict


def training(local_rank, config):
    return base_training(local_rank, config, get_dataflow, initialize, get_metrics, visualize)


def get_dataflow(config, logger=None):
    if idist.get_local_rank() > 0:
        idist.barrier()

    mode = config.get("mode", "depth")

    train_dataset, test_dataset = make_datasets(config["data"])
    vis_dataset = copy(test_dataset)

    # Change eval dataset to only use a single prediction and to return gt depth.
    test_dataset.frame_count = 1 if isinstance(train_dataset, KittiRawDataset) else 2
    test_dataset._left_offset = 0
    test_dataset.return_stereo = mode == "nvs"
    test_dataset.length = min(256, test_dataset.length) if isinstance(train_dataset,
                                                                      KittiRawDataset) else test_dataset.length

    # Change visualisation dataset
    vis_dataset.length = 1
    vis_dataset._skip = 12 if isinstance(train_dataset, KittiRawDataset) else 50
    # vis_dataset.return_depth = True

    if idist.get_local_rank() == 0:
        # Ensure that only local rank 0 download the dataset
        idist.barrier()

    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    train_loader = idist.auto_dataloader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        drop_last=True,
        collate_fn=bbox3d_collate_fn if config["data"]["return_3d_bboxes"] else None,
    )
    test_loader = idist.auto_dataloader(
        test_dataset,
        batch_size=1,
        num_workers=config["num_workers"],
        shuffle=False,
        collate_fn=bbox3d_collate_fn if config["data"]["return_3d_bboxes"] else None,
    )
    vis_loader = idist.auto_dataloader(
        vis_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        collate_fn=bbox3d_collate_fn if config["data"]["return_3d_bboxes"] else None,
    )

    return train_loader, test_loader, vis_loader


def get_metrics(config, device, train=False):
    if train:
        # names = ['loss', 'loss_rgb_fine', 'loss_depth_recon', 'loss_temp_align']
        # metrics = {name: RunningAverage(output_transform=lambda x: x["loss_dict"][name]) for name in names}
        # return metrics
        names = ['loss']
        metrics = {name: RunningAverage(output_transform=lambda x: x["loss_dict"][name]) for name in names}
        return metrics
    names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    metrics = {name: MeanMetric((lambda n: lambda x: x["output"][n])(name), device) for name in names}
    return metrics


def initialize(config: dict, logger=None):
    """
    Initialize model, optimizer, criterion and scheduler.

    Model: BTSWrapper()
    Optimizer: Adam()
    Criterion: ReconstructionLoss(), 'l1 + ssim'
    Scheduler: StepLR(), base_lr=1e-4, gamma=0.1, step_size=120000
    """
    net = BTSNet(config["model_conf"])

    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()

    model = BTSWrapper(
        renderer,
        config["model_conf"],
    )

    model = idist.auto_model(model)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    optimizer = idist.auto_optim(optimizer)

    lr_scheduler = make_scheduler(config.get("scheduler", {}), optimizer)

    criterion = ReconstructionLoss(config["loss"])

    return model, optimizer, criterion, lr_scheduler


def visualize(engine: Engine, logger: TensorboardLogger, step: int, tag: str):
    pass
