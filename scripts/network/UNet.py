import os
import sys
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torchvision.transforms.functional as FF


class ConvBlock(nn.Module):
    """
    卷积模块：
    由两层 Conv + Norm + ReLU 组成，
    可选加入 dropout。
    """

    def __init__(in_channels, features, dims, name, dropout=False):
        super(ConvBlock, self).__init__()

    def _block(in_channels, features, dims, name, dropout=False):
        # 模块内部由两层卷积、归一化和激活函数组成

        # 根据维度选择对应的卷积与归一化层
        if dims == 3:
            from torch.nn import InstanceNorm3d as Norm
            from torch.nn import Conv3d as Conv
        elif dims == 2:
            from torch.nn import InstanceNorm2d as Norm
            from torch.nn import Conv2d as Conv

        layers = [(name + "conv_1", Conv(
                    in_channels=in_channels,
                    out_channels=features,
                    kernel_size=3,
                    padding=1,
                    bias=False)),

                (name + "Inorm_1", Norm(num_features=features)),
                (name + "relu_1", nn.ReLU(inplace=True)),

                (name + "conv_2", Conv(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=3,
                    padding=1,
                    bias=False)),

                (name + "Inorm_2", Norm(num_features=features)),
                (name + "relu_2", nn.ReLU(inplace=True)),
            ]
        if dropout:
            layers.append((name + 'DropOut', nn.Dropout(0.2)))  # dropout 概率

        return nn.Sequential(
            OrderedDict(layers))


class EncoderBranch(nn.Module):

    """
    编码器分支：
    in_channels = 网络输入通道数
    features = 第一层的特征图数量，默认 6
    depth = 网络深度，默认 4
    """

    def __init__(self, in_channels, features=6, ndims=3, net_depth=4, dropout=False):

        super(EncoderBranch, self).__init__()

        # 支持 2D 与 3D 模式
        assert ndims in [2, 3]
        if ndims == 3:
            from torch.nn import MaxPool3d as MaxPooling
            from torch.nn import ConvTranspose3d as ConvTrans
            from torch.nn import Conv3d as FConv
        elif ndims == 2:
            from torch.nn import MaxPool2d as MaxPooling
            from torch.nn import ConvTranspose2d as ConvTrans
            from torch.nn import Conv2d as FConv

        self.dims = ndims
        self.depth = net_depth

        # 编码器主体
        self.encoder1 = ConvBlock._block(in_channels, features, ndims, name="encoder_1", dropout=dropout)
        self.pool1 = MaxPooling(kernel_size=2, stride=2)

        self.encoder2 = ConvBlock._block(features, features * 2, ndims, name="encoder_2", dropout=dropout)
        self.pool2 = MaxPooling(kernel_size=2, stride=2)

        self.encoder3 = ConvBlock._block(features * 2, features * 4, ndims, name="encoder_3", dropout=dropout)
        self.pool3 = MaxPooling(kernel_size=2, stride=2)

        self.encoder4 = ConvBlock._block(features * 4, features * 8, ndims, name="encoder_4", dropout=dropout)

    def forward(self, x):

        # 编码路径的逐层输出
        enc1 = self.encoder1(x)

        if self.depth == 1:
            return enc1
        enc2 = self.encoder2(self.pool1(enc1))
        if self.depth == 2:
            return enc1, enc2
        enc3 = self.encoder3(self.pool2(enc2))
        if self.depth == 3:
            return enc1, enc2, enc3
        enc4 = self.encoder4(self.pool3(enc3))
        if self.depth == 4:
            return enc1, enc2, enc3, enc4


class BottleNeck(nn.Module):

    def __init__(self, features=6, ndims=3, net_depth=4, dropout=False):

        super(BottleNeck, self).__init__()
        if ndims == 3:
            from torch.nn import MaxPool3d as MaxPooling
            from torch.nn import ConvTranspose3d as ConvTrans
            from torch.nn import Conv3d as FConv
        elif ndims == 2:
            from torch.nn import MaxPool2d as MaxPooling
            from torch.nn import ConvTranspose2d as ConvTrans
            from torch.nn import Conv2d as FConv

        self.pool = MaxPooling(kernel_size=2, stride=2)
        f = (2 ** (net_depth - 1)) * features
        self.bottleneck = ConvBlock._block(f, f, ndims, name="Bottleneck", dropout=dropout)

    def forward(self, enc_out):
        """
        enc_out 为编码器最后一层的输出，即 enc_output[-1]。
        """
        return self.bottleneck(self.pool(enc_out))


class DecoderBranch(nn.Module):

    """
    解码器分支：
    dec_depth == 1 表示输出回到最高分辨率层，
    默认值为 1。
    """

    def __init__(self, features=6, ndims=3, net_depth=4, dec_depth=1, dropout=False):

        super(DecoderBranch, self).__init__()

        self.dec_depth = dec_depth
        self.net_depth = net_depth

        if ndims == 3:
            from torch.nn import MaxPool3d as MaxPooling
            from torch.nn import ConvTranspose3d as ConvTrans
            from torch.nn import Conv3d as FConv
        elif ndims == 2:
            from torch.nn import MaxPool2d as MaxPooling
            from torch.nn import ConvTranspose2d as ConvTrans
            from torch.nn import Conv2d as FConv

        # 确保解码输出层级不超过编码深度
        assert self.dec_depth <= self.net_depth

        self.upconv1 = ConvTrans(features * 8, features * 8, kernel_size=2, stride=2)  # 最底层分支
        self.decoder1 = ConvBlock._block((features * 8) * 2, features * 4, ndims, name="decoder_4", dropout=dropout)

        self.upconv2 = ConvTrans(features * 4, features * 4, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock._block((features * 4) * 2, features * 2, ndims, name="decoder_3", dropout=dropout)

        self.upconv3 = ConvTrans(features * 2, features * 2, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock._block((features * 2) * 2, features, ndims, name="decoder_3", dropout=dropout)

        self.upconv4 = ConvTrans(features, features, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock._block((features) * 2, features, ndims, name="decoder_1", dropout=dropout)

    def forward(self, bottleneck, enc_output):
        """
        参数:
            bottleneck: 网络瓶颈层输出
            enc_output: 编码器各层输出列表，例如 [enc1, enc2, enc3, ...]
        """

        # 网络深度为 4 时的解码路径
        if self.net_depth == 4:

            dec4 = self.upconv1(bottleneck)
            dec4 = torch.cat((dec4, enc_output[-1]), dim=1)  # 与跳跃连接特征拼接
            dec4 = self.decoder1(dec4)

            if self.dec_depth == 4:
                return dec4

            dec3 = self.upconv2(dec4)
            dec3 = torch.cat((dec3, enc_output[-2]), dim=1)
            dec3 = self.decoder2(dec3)
            if self.dec_depth == 3:
                return dec3

            dec2 = self.upconv3(dec3)
            dec2 = torch.cat((dec2, enc_output[-3]), dim=1)
            dec2 = self.decoder3(dec2)

            if self.dec_depth == 2:
                return dec2

            dec1 = self.upconv4(dec2)
            dec1 = torch.cat((dec1, enc_output[-4]), dim=1)
            dec1 = self.decoder4(dec1)

            if self.dec_depth == 1:
                return dec1

        elif self.net_depth == 3:
            dec3 = self.upconv2(bottleneck)
            dec3 = torch.cat((dec3, enc_output[-1]), dim=1)
            dec3 = self.decoder2(dec3)
            if self.dec_depth == 3:
                return dec3

            dec2 = self.upconv3(dec3)
            dec2 = torch.cat((dec2, enc_output[-2]), dim=1)
            dec2 = self.decoder3(dec2)

            if self.dec_depth == 2:
                return dec2

            dec1 = self.upconv4(dec2)
            dec1 = torch.cat((dec1, enc_output[-3]), dim=1)
            dec1 = self.decoder4(dec1)

            if self.dec_depth == 1:
                return dec1

        elif self.net_depth == 2:

            dec2 = self.upconv3(bottleneck)
            dec2 = torch.cat((dec2, enc_output[-1]), dim=1)
            dec2 = self.decoder3(dec2)

            if self.dec_depth == 2:
                return dec2

            dec1 = self.upconv4(dec2)
            dec1 = torch.cat((dec1, enc_output[-2]), dim=1)
            dec1 = self.decoder4(dec1)

            if self.dec_depth == 1:
                return dec1


class UNet_MW(nn.Module):
    """
    自定义 U-Net。

    参数说明：
    in_channels = 输入通道数
    out_channels = 输出通道数
    features = 初始特征图数量
    dec_depth = 输出来自哪一层解码器（1 表示最高分辨率层）
    net_depth = 网络总深度
    """

    def __init__(self, params):
        """
        初始化 U-Net。
        """

        super(UNet_MW, self).__init__()
        in_channels = params.network_params.in_chan
        out_channels = params.network_params.out_chan
        features = params.network_params.fi
        net_depth = params.network_params.net_depth
        dropout = params.network_params.dropout
        dec_depth = params.network.dec_depth
        ndims = params.dataset.ndims

        if ndims == 3:
            from torch.nn import Conv3d as FConv
        elif ndims == 2:
            from torch.nn import Conv2d as FConv

        # 1. 编码器
        self.enc = EncoderBranch(in_channels, features, ndims, net_depth, dropout)
        # 2. 瓶颈层
        self.bottleneck = BottleNeck(features, ndims, net_depth, dropout)
        # 3. 解码器
        self.dec = DecoderBranch(features, ndims, net_depth, dec_depth, dropout)

        self.conv = FConv(in_channels=features, out_channels=out_channels, kernel_size=1)  # 输出头

    def forward(self, x, _):

        enc_outputs = self.enc(x)
        BottleNeck = self.bottleneck(enc_outputs[-1])
        dec_output = self.dec(BottleNeck, enc_outputs)
        cnn_output = self.conv(dec_output)

        return torch.sigmoid(cnn_output), 1
