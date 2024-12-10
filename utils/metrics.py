import torch
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class MeanMetric(Metric):
    def __init__(self, output_transform=lambda x: x["output"], device="cpu"):
        self._sum = None
        self._num_examples = None
        self.required_output_keys = ()
        super(MeanMetric, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum = torch.tensor(0, device=self._device, dtype=float)
        self._num_examples = 0
        super(MeanMetric, self).reset()

    @reinit__is_reduced
    def update(self, value):
        if torch.any(torch.isnan(torch.tensor(value))):
            return
        self._sum += value
        self._num_examples += 1

    @sync_all_reduce("_num_examples:SUM", "_sum:SUM")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return self._sum.item() / self._num_examples

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output)
        self.update(output)
