import os
import sys
import torch
import numpy as np
import numbers
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn.utils import spectral_norm
import torchvision.transforms.functional as FF
from network.UNet import ConvBlock, EncoderBranch, DecoderBranch, BottleNeck


class WholeDiffeoUnit(nn.Module):
    """
    形变模块，主要完成以下步骤：
    1. 计算解码器分支输出
    2. 生成与维度匹配的位移场
    3. 执行可微分积分
    4. 将位移场应用到先验形状上
    """

    def __init__(self, params, branch=1):
        super(WholeDiffeoUnit, self).__init__()

        # 从参数中读取配置
        self.out_channels = params.network_params.out_chan
        self.ndims = params.dataset.ndims
        self.viscous = params.network.guas_smooth
        self.act = params.network.act
        self.features = params.network_params.fi
        self.dropout = params.network_params.dropout
        self.net_depth = params.network_params.net_depth
        self.dec_depth = params.network.dec_depth[branch]
        self.inshape = params.dataset.inshape
        self.int_steps = params.network.diffeo_int
        self.Guas_kernel = params.network.Guas_kernel
        self.Guas_P = params.network.sigma
        self.mega_P = params.network.mega_P
        self.integrator_name = getattr(params.network, 'integrator', 'scaling_squaring')
        self.r2net_blocks = getattr(params.network, 'r2net_blocks', 7)

        # 构建解码器输出
        self.dec = DecoderBranch(features=self.features, ndims=self.ndims, net_depth=self.net_depth, dec_depth=self.dec_depth, dropout=self.dropout)

        # 初始位移场尺寸
        frac_size_change = [1, 2, 4, 8]  # 各解码层对应的缩放比例
        self.flow_field_size = [int(s / frac_size_change[self.dec_depth - 1]) for s in self.inshape]
        # 上采样后的位移场尺寸
        self.Mega_inshape = [s * self.mega_P for s in self.inshape]

        # 1. 生成位移场
        self.gen_field = GenDisField(self.dec_depth, self.features, self.ndims)

        # 2. 应用可微分积分设置
        self.flow_integrator = build_flow_integrator(
            integrator_name=self.integrator_name,
            flow_field_size=self.flow_field_size,
            mega_size=self.Mega_inshape,
            int_steps=self.int_steps,
            viscous=self.viscous,
            Guas_kernel=self.Guas_kernel,
            Guas_P=self.Guas_P,
            mega_P=self.mega_P,
            ndims=self.ndims,
            r2net_blocks=self.r2net_blocks,
        )
        # 3. 将位移应用到先验形状
        self.transformer = mw_SpatialTransformer(self.Mega_inshape)

    def forward(self, BottleNeck, enc_outputs, prior_shape):
        # 计算解码器输出
        dec_output = self.dec(BottleNeck, enc_outputs)
        flow_field = self.gen_field(dec_output)
        flow_upsamp = self.flow_integrator(flow_field, self.act, self.viscous, self.ndims)
        sampled = WarpPriorShape(self, prior_shape, flow_upsamp)

        return flow_field, flow_upsamp, sampled


def build_flow_integrator(
    integrator_name,
    flow_field_size,
    mega_size,
    int_steps,
    viscous,
    Guas_kernel,
    Guas_P,
    mega_P,
    ndims,
    r2net_blocks,
):
    """根据配置选择积分器实现。"""

    if integrator_name == 'scaling_squaring':
        return DiffeoUnit(
            flow_field_size,
            mega_size,
            int_steps=int_steps,
            viscous=viscous,
            Guas_kernel=Guas_kernel,
            Guas_P=Guas_P,
            mega_P=mega_P,
        )
    if integrator_name == 'r2net_lc_resnet':
        return R2NetFlowIntegrator(
            flow_field_size,
            mega_size,
            ndims=ndims,
            n_blocks=r2net_blocks,
        )
    raise ValueError(f'不支持的积分器: {integrator_name}')


class LC_ResNet_Block(nn.Module):
    """R2Net 的 LC-ResNet 残差块。"""

    def __init__(self, ndims):
        super().__init__()

        if ndims == 2:
            Conv = nn.Conv2d
        elif ndims == 3:
            Conv = nn.Conv3d
        else:
            raise RuntimeError(f'仅支持 2D/3D，收到 ndims={ndims}')

        self.conv1 = spectral_norm(Conv(ndims, ndims, kernel_size=3, padding=1, stride=1, bias=False))
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = spectral_norm(Conv(ndims, ndims, kernel_size=1, padding=0, stride=1, bias=False))
        self.tanh = nn.Tanh()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.tanh(residual) / 2.0
        return x + residual


class R2NetFlowIntegrator(nn.Module):
    """使用 LC-ResNet 残差块替代 scaling and squaring。"""

    def __init__(self, flow_field_size, mega_size, ndims, n_blocks=7):
        super().__init__()
        self.Mega_inshape = mega_size
        modes = {2: 'bilinear', 3: 'trilinear'}
        self.blocks = nn.ModuleList([LC_ResNet_Block(ndims) for _ in range(n_blocks)])
        self.upsample = nn.Upsample(
            size=self.Mega_inshape,
            mode=modes[len(flow_field_size)],
            align_corners=False,
        )

    def forward(self, flow_field, *_):
        velocity = flow_field / 2.0
        for block in self.blocks:
            velocity = block(velocity)
        return self.upsample(velocity)


class GenDisField(nn.Module):
    """
    根据 U-Net 输出生成合适尺寸的位移场。

    输入：解码器输出 [batch, feature_maps, ...]
    输出：位移场 [batch, ndims, ...]
    """
    def __init__(self, layer_nb, features, ndims):
        super().__init__()

        if ndims == 3:
            from torch.nn import Conv3d as ConvD
        elif ndims == 2:
            from torch.nn import Conv2d as ConvD

        dec_features = [1, 1, 2, 4]  # 每个解码层对应的特征倍率
        self.flow_field = ConvD(dec_features[layer_nb - 1] * features, out_channels=ndims, kernel_size=1)  # 输出通道数等于空间维度
        self.flow_field.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_field.weight.shape))
        self.flow_field.bias = nn.Parameter(torch.zeros(self.flow_field.bias.shape))

    def forward(self, CNN_output):
        return self.flow_field(CNN_output)


class DiffeoUnit(nn.Module):
    """
    输入初始位移场，输出最终上采样后的位移场。

    输入为 [ndim, M, N] 形式的位移场，主要包含三步：
    1. 激活函数约束
    2. 多步积分放大
    3. 超分辨率上采样
    """

    def __init__(self, flow_field_size, mega_size, int_steps=7, viscous=1, Guas_kernel=5, Guas_P=2, mega_P=2):
        super(DiffeoUnit, self).__init__()

        # 1. 积分层
        self.flow_field_size = flow_field_size
        self.integrate_layer = mw_DiffeoLayer(flow_field_size, int_steps, Guas_kernel, Guas_P=Guas_P)

        # 2. 最终上采样
        self.Mega_inshape = mega_size
        modes = {2: 'bilinear', 3: 'trilinear'}
        self.MEGAsmoothing_upsample = nn.Upsample(self.Mega_inshape, mode=modes[len(flow_field_size)], align_corners=False)

    def forward(self, flow_field, act, viscous, ndims):

        # 1. 使用激活函数限制初始位移幅度
        if act:
            flow_field = DiffeoActivat(flow_field, self.flow_field_size)

        # 2. 通过积分得到最终位移场
        amplified_flow_field = self.integrate_layer(flow_field, viscous)

        # 3. 上采样到目标尺寸
        flow_Upsamp = self.MEGAsmoothing_upsample(amplified_flow_field)

        return flow_Upsamp


class mw_DiffeoLayer(nn.Module):
    """
    使用 scaling and squaring 对向量场进行积分。
    参考实现改编自：https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, inshape, nsteps, kernel=3, Guas_P=2):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        # 构建用于积分的空间变换器
        self.transformer = mw_SpatialTransformer(inshape)

        # ------------------------------
        # 平滑核配置
        # ------------------------------
        ndims = len(inshape)
        self.sigma = Guas_P
        self.SmthKernel = GaussianSmoothing(channels=ndims, kernel_size=kernel, sigma=Guas_P, dim=ndims)
        # ------------------------------
        # ------------------------------

    def forward(self, vec, viscous=1):

        for n in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
            if viscous:
                # 若启用黏性平滑，则在每次复合后执行一次平滑
                vec = self.SmthKernel(vec)

        return vec


class mw_SpatialTransformer(nn.Module):
    """
    PyTorch 版空间变换器。

    PyTorch 的 grid_sample 需要位于 -1 到 1 之间的坐标网格。
    src 可以是先验形状或位移场，常见形状如 [2, 3, x, x, x]。
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        # 生成采样网格（PyTorch 坐标格式）
        vectors = [torch.linspace(-1, 1, s) for s in size]
        grids = torch.meshgrid(*vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)  # 不参与训练，但随模型一起保存

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # 采样前需要按 PyTorch 约定调整坐标轴顺序
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True)


class GaussianSmoothing(nn.Module):
    """
    Adrian Sahlman 的高斯平滑实现：
    https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/7

    对 1D、2D 或 3D 张量执行高斯平滑。
    过滤操作会按输入通道独立进行，即使用 depthwise convolution。

    参数:
        channels (int, sequence): 输入张量的通道数，输出通道数相同。
        kernel_size (int, sequence): 高斯核大小。
        sigma (float, sequence): 高斯核标准差；如果小于 0，则 sigma 可学习。
        dim (int, optional): 数据维度，默认 2。
    """

    def __init__(self, channels, kernel_size=5, sigma=2, dim=2):
        super(GaussianSmoothing, self).__init__()
        # 默认初始化 sigma 为 2
        self.og_sigma = sigma

        kernel_dic = {3: 1, 5: 2}
        self.pad = kernel_dic[kernel_size]

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # 高斯核由各维高斯函数相乘得到
        kernel = 1
        meshgrids = torch.meshgrid(
            *[torch.arange(size, dtype=torch.float32) for size in kernel_size],
            indexing='ij',
        )

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                    torch.exp((-((mgrid - mean) / std) ** 2) / 2)

        # 保证高斯核元素和为 1
        kernel = kernel / torch.sum(kernel)

        # 调整为 depthwise convolution 所需的权重形状
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        if self.og_sigma < 0:
            # --- 可学习 sigma 的分支
            sigma = 2
            self.learnable = 1  # 标记当前为可学习卷积

            if dim == 1:
                self.conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=self.pad)
            elif dim == 2:
                self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=self.pad)
            elif dim == 3:
                self.conv = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=self.pad)
            else:
                raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

            # 使用高斯核初始化权重
            self.conv.weight = nn.Parameter(torch.cat((kernel, kernel), dim=1))
            self.conv.bias = nn.Parameter(torch.zeros(self.conv.bias.shape))

        else:
            # --- 固定高斯核分支
            self.learnable = 0

            self.register_buffer('weight', kernel)

            self.groups = channels

            if dim == 1:
                self.conv = F.conv1d
            elif dim == 2:
                self.conv = F.conv2d
            elif dim == 3:
                self.conv = F.conv3d
            else:
                raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

    def forward(self, input):
        """
        对输入执行高斯平滑。

        参数:
            input (torch.Tensor): 待平滑的输入张量。

        返回:
            filtered (torch.Tensor): 平滑后的输出张量。
        """
        # 根据当前模式选择固定卷积或可学习卷积
        if self.learnable == 1:
            return self.conv(input)
        else:
            return self.conv(input, weight=self.weight, groups=self.groups, padding=self.pad)


def DiffeoActivat(flow_field, size):
    """激活函数。

    参数:
        flow_field (tensor): 每个方向上的位移场张量。
        size (list): 位移场对应尺寸，用于限制初始位移幅度。

    返回:
        tensor: 经过激活函数约束后的位移场。
    """

    # 仅支持 2D 或 3D 位移场
    assert flow_field.size()[1] in [2, 3]
    assert len(size) in [2, 3]

    if len(size) == 3:
        flow_1 = torch.tanh(flow_field[:, 0, :, :, :]) * (1 / size[0])
        flow_2 = torch.tanh(flow_field[:, 1, :, :, :]) * (1 / size[1])
        flow_3 = torch.tanh(flow_field[:, 2, :, :, :]) * (1 / size[2])
        flow_field = torch.stack((flow_1, flow_2, flow_3), dim=1)
    elif len(size) == 2:
        flow_1 = torch.tanh(flow_field[:, 0, :, :]) * (1 / size[0])
        flow_2 = torch.tanh(flow_field[:, 1, :, :]) * (1 / size[1])
        flow_field = torch.stack((flow_1, flow_2), dim=1)

    return flow_field


def WarpPriorShape(self, prior_shape, disp_field):
    """
    对一组先验形状施加位移场变换。
    """
    # 将位移场应用到先验形状上
    disp_prior_shape = self.transformer(prior_shape, disp_field)

    return disp_prior_shape
