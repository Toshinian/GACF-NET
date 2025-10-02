from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .ResnetEncoder import ResnetEncoder
from .det_heads import Bev_predictor
import matplotlib.pyplot as PLT


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class Encoder(nn.Module):
    """ Encodes the Image into low-dimensional feature representation

    Attributes
    ----------
    num_layers : int
        Number of layers to use in the ResNet
    img_ht : int
        Height of the input RGB image
    img_wt : int
        Width of the input RGB image
    pretrained : bool
        Whether to initialize ResNet with pretrained ImageNet parameters

    Methods
    -------
    forward(x, is_training):
        Processes input image tensors into output feature tensors
    """

    def __init__(self, num_layers, img_ht, img_wt, pretrained=True):
        super(Encoder, self).__init__()

        self.resnet_encoder = ResnetEncoder(num_layers, pretrained)
        num_ch_enc = self.resnet_encoder.num_ch_enc
        self.conv1 = Conv3x3(num_ch_enc[-1], 1024)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.FloatTensor
            Batch of Image tensors
            | Shape: (batch_size, 3, img_height, img_width)

        Returns
        -------
        x : torch.FloatTensor
            Batch of low-dimensional image representations
            | Shape: (batch_size, 128, img_height/128, img_width/128)
        """

        batch_size, c, h, w = x.shape
        x = self.resnet_encoder(x)[-1]
        x = self.pool(self.conv1(x))
        return x



class Decoder(nn.Module):
    def __init__(self, num_ch_enc, num_class=5, type=''):
        super(Decoder, self).__init__()
        self.num_output_channels = num_class
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([64, 128, 256, 512])  # 各层目标通道数
        

        self.conv_blocks = nn.ModuleList()
        

        for i in range(3, -1, -1):

            in_ch = 1024 if i == 3 else self.num_ch_dec[i + 1]
            out_ch = self.num_ch_dec[i]
            

            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.conv_blocks.append(block)
        
        self.topview = Conv3x3(64, self.num_output_channels)  # 输入通道应为64

    def forward(self, x, is_training=True):

        for i, block in enumerate(self.conv_blocks):
            x = block(x)
        

        seg = self.topview(x)
        if not is_training:
            seg = F.softmax(seg, dim=1)
            
        return seg, x
