from torch import nn


class DepthRefinement(nn.Module):
    """
    Residual Inverse Depth Module for precise metric depth recovery.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(DepthRefinement, self).__init__(*args, **kwargs)
        self.min_depth = 0.1
        self.max_depth = 80

    def forward(self, pseudo_depth, res_inv_depth, vis=False):
        pseudo_depth = pseudo_depth.clamp(min=self.min_depth, max=self.max_depth)
        res_inv_depth /= 10
        inv_pseudo_depth = 1. / pseudo_depth
        if vis:
            weight_mask = 1 - pseudo_depth / self.max_depth
            res_inv_depth *= weight_mask
        depth = 1. / (inv_pseudo_depth + res_inv_depth + 1e-8)
        return depth
