import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.data_util import make_test_dataset
from utils.base_evaluator import base_evaluation
from utils.metrics import MeanMetric


class BTSWrapper(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.depth_scaling = config["depth_scaling"]

    @staticmethod
    def get_loss_metric_names():
        return ["loss", "loss_l2", "loss_mask", "loss_temporal"]

    def forward(self, data):
        data = dict(data)
        data.update(self.compute_depth_metrics(data))
        return data

    def compute_depth_metrics(self, data):
        depth_gt = data["depths"][0]
        depth_pred = data['pseudo_depth'][0]
        # vis_tensor(depth_gt[0, 0])
        # vis_tensor(depth_pred[0, 0])
        if self.depth_scaling == "median":
            mask = depth_gt > 0
            scaling = torch.median(depth_gt[mask]) / torch.median(depth_pred[mask])
            depth_pred = scaling * depth_pred

        depth_pred = torch.clamp(depth_pred, 1e-3, 80)
        # vis_tensor(depth_gt[0, 0])
        # vis_tensor(depth_pred[0, 0])
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
    model = BTSWrapper(config["model_conf"])
    return model
