from models.common.model.layers import *

import numpy as np
import torch
import torch.nn as nn

import torchvision.models as models


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if pretrained:
            self.encoder = resnets[num_layers](weights='DEFAULT')
        else:
            self.encoder = resnets[num_layers]()

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


class Decoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), use_skips=True):
        super(Decoder, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(80., dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(0.5, dtype=torch.float))

        self.use_skips = use_skips
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = [16, 32, 64, 128, 256]

        # decoder
        self.upconvs0 = []
        self.upconvs1 = []
        self.dispconvs = []
        self.i_to_scaleIdx_conversion = {}

        for i in range(4, -1, -1):  # 4,3,2,1,0
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.upconvs0.append(ConvBlock(num_ch_in, num_ch_out))

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.upconvs1.append(ConvBlock(num_ch_in, num_ch_out))

        for cnt, s in enumerate(self.scales):
            self.dispconvs.append(Conv3x3(self.num_ch_dec[s], 1))

            if s in range(4, -1, -1):
                self.i_to_scaleIdx_conversion[s] = cnt

        self.upconvs0 = nn.ModuleList(self.upconvs0)
        self.upconvs1 = nn.ModuleList(self.upconvs1)
        self.dispconvs = nn.ModuleList(self.dispconvs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        """
            input:
                list 5, with resolution: hxw/2, hxw/4, hxw/8, hxw/16, hxw/32
            return:
                list 4, with resolution: hxw, hxw/2, hxw/4, hxw/8
        """
        outputs = []

        # decoder
        x = input_features[-1]

        for cnt, i in enumerate(range(4, -1, -1)):
            x = self.upconvs0[cnt](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.upconvs1[cnt](x)
            if i in self.scales:
                idx = self.i_to_scaleIdx_conversion[i]
                depth = self.alpha * self.sigmoid(self.dispconvs[idx](x)) + self.beta
                outputs.append(depth)

        outputs = outputs[::-1]
        return outputs[0]


class ResidualDepth(nn.Module):
    def __init__(
            self,
            resnet_layers=18,
            scales=range(4)
    ):
        super().__init__()
        self.encoder = ResnetEncoder(resnet_layers, True, 1)
        self.num_ch_enc = self.encoder.num_ch_enc

        self.upsample_mode = 'nearest'
        self.scales = scales

        # decoder
        self.decoder = Decoder(num_ch_enc=self.num_ch_enc)
        self.num_ch_dec = self.decoder.num_ch_dec

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x: image (B, C, H, W)
        """
        x = torch.cat([x * .5 + .5], dim=1)
        features = self.encoder(x)
        output = self.decoder(features)

        return output


if __name__ == '__main__':
    model = ResidualDepth().cuda()
    model.train()
    tgt_img = torch.randn(4, 3, 192, 640).cuda()
    tgt_depth = model(tgt_img)

    print(tgt_depth.size())
