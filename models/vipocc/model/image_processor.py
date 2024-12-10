from torch import nn


class RGBProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = 3

    def forward(self, images):
        images = images * .5 + .5
        return images
