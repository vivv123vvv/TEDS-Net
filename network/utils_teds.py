import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn.utils import spectral_norm

from network.UNet import DecoderBranch
from parameters.acdc_parameters import normalize_integrator_name


class LC_ResNet_Block(nn.Module):
    """
    R2Net residual block:
    ConvSN(3x3) -> LeakyReLU -> ConvSN(1x1) -> Tanh -> Scale -> Add
    """

    def __init__(self, channels):
        super().__init__()
        ndims = channels
        conv = nn.Conv2d if ndims == 2 else nn.Conv3d

        self.conv1 = spectral_norm(
            conv(channels, channels, kernel_size=3, padding=1, stride=1, bias=False)
        )
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = spectral_norm(
            conv(channels, channels, kernel_size=1, padding=0, stride=1, bias=False)
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.tanh(residual)
        residual = residual / 2.0
        return x + residual


class R2Net_Integrator(nn.Module):
    """
    Learned integrator used by the R2Net branch.
    """

    def __init__(self, channels, n_blocks=7):
        super().__init__()
        self.blocks = nn.ModuleList([LC_ResNet_Block(channels) for _ in range(n_blocks)])

    def forward(self, flow0):
        velocity = flow0 / 2.0
        for block in self.blocks:
            velocity = block(velocity)
        return velocity


class WholeDiffeoUnit(nn.Module):
    """
    Diffeomorphic block with switchable integrators.
    """

    def __init__(self, params, branch=1):
        super().__init__()

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
        self.integrator = normalize_integrator_name(getattr(params.network, "integrator", "r2net"))

        self.dec = DecoderBranch(
            features=self.features,
            ndims=self.ndims,
            net_depth=self.net_depth,
            dec_depth=self.dec_depth,
            dropout=self.dropout,
        )

        frac_size_change = [1, 2, 4, 8]
        self.flow_field_size = [int(size / frac_size_change[self.dec_depth - 1]) for size in self.inshape]
        self.Mega_inshape = [size * self.mega_P for size in self.inshape]

        self.gen_field = GenDisField(self.dec_depth, self.features, self.ndims)

        if self.integrator == "original_teds":
            self.diffeo_field = DiffeoUnit(
                self.flow_field_size,
                self.Mega_inshape,
                self.int_steps,
                self.viscous,
                self.Guas_kernel,
                self.Guas_P,
                self.mega_P,
            )
        elif self.integrator == "r2net":
            self.r2net_integrator = R2Net_Integrator(
                channels=self.ndims,
                n_blocks=params.network.r2net_blocks,
            )
        else:
            raise ValueError(f"Unsupported integrator '{self.integrator}'")

        self.transformer = mw_SpatialTransformer(self.Mega_inshape)

    def forward(self, BottleNeck, enc_outputs, prior_shape):
        dec_output = self.dec(BottleNeck, enc_outputs)
        raw_velocity = self.gen_field(dec_output)

        if self.integrator == "original_teds":
            flow_field = raw_velocity
            flow_upsamp = self.diffeo_field(raw_velocity, self.act, self.viscous, self.ndims)
        elif self.integrator == "r2net":
            flow_field = self.r2net_integrator(raw_velocity)
            mode = "bilinear" if self.ndims == 2 else "trilinear"
            flow_upsamp = F.interpolate(
                flow_field,
                size=tuple(self.Mega_inshape),
                mode=mode,
                align_corners=True,
            )
        else:
            raise ValueError(f"Unsupported integrator '{self.integrator}'")

        sampled = WarpPriorShape(self, prior_shape, flow_upsamp)
        return flow_field, flow_upsamp, sampled


class GenDisField(nn.Module):
    """
    Generate the displacement field from decoder features.
    """

    def __init__(self, layer_nb, features, ndims):
        super().__init__()

        if ndims == 3:
            from torch.nn import Conv3d as ConvD
        elif ndims == 2:
            from torch.nn import Conv2d as ConvD
        else:
            raise ValueError(f"Unsupported ndims '{ndims}'")

        dec_features = [1, 1, 2, 4]
        self.flow_field = ConvD(
            dec_features[layer_nb - 1] * features,
            out_channels=ndims,
            kernel_size=1,
        )
        self.flow_field.weight = nn.Parameter(
            Normal(0, 1e-5).sample(self.flow_field.weight.shape)
        )
        self.flow_field.bias = nn.Parameter(torch.zeros(self.flow_field.bias.shape))

    def forward(self, cnn_output):
        return self.flow_field(cnn_output)


class DiffeoUnit(nn.Module):
    """
    Original TEDS scaling-and-squaring integrator plus upsampling.
    """

    def __init__(
        self,
        flow_field_size,
        mega_size,
        int_steps=7,
        viscous=1,
        Guas_kernel=5,
        Guas_P=2,
        mega_P=2,
    ):
        super().__init__()

        self.flow_field_size = flow_field_size
        self.integrate_layer = mw_DiffeoLayer(
            flow_field_size,
            int_steps,
            Guas_kernel,
            Guas_P=Guas_P,
        )

        self.Mega_inshape = mega_size
        modes = {2: "bilinear", 3: "trilinear"}
        self.MEGAsmoothing_upsample = nn.Upsample(
            self.Mega_inshape,
            mode=modes[len(flow_field_size)],
            align_corners=False,
        )

    def forward(self, flow_field, act, viscous, ndims):
        if act:
            flow_field = DiffeoActivat(flow_field, self.flow_field_size)

        amplified_flow_field = self.integrate_layer(flow_field, viscous)
        flow_upsamp = self.MEGAsmoothing_upsample(amplified_flow_field)
        return flow_upsamp


class mw_DiffeoLayer(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    Adapted from: https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, inshape, nsteps, kernel=3, Guas_P=2):
        super().__init__()

        assert nsteps >= 0, f"nsteps should be >= 0, found: {nsteps}"
        self.nsteps = nsteps
        self.transformer = mw_SpatialTransformer(inshape)

        ndims = len(inshape)
        self.sigma = Guas_P
        self.SmthKernel = GaussianSmoothing(
            channels=ndims,
            kernel_size=kernel,
            sigma=Guas_P,
            dim=ndims,
        )

    def forward(self, vec, viscous=1):
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
            if viscous:
                vec = self.SmthKernel(vec)

        return vec


class mw_SpatialTransformer(nn.Module):
    """
    PyTorch spatial transformer operating on normalized grids.
    """

    def __init__(self, size, mode="bilinear"):
        super().__init__()

        self.mode = mode
        vectors = [torch.linspace(-1, 1, size_dim) for size_dim in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True)


class GaussianSmoothing(nn.Module):
    """
    Gaussian smoothing on 1D, 2D or 3D tensors.
    """

    def __init__(self, channels, kernel_size=5, sigma=2, dim=2):
        super().__init__()
        self.og_sigma = sigma

        kernel_dic = {3: 1, 5: 2}
        self.pad = kernel_dic[kernel_size]

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size],
            indexing="ij",
        )

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(
                (-((mgrid - mean) / std) ** 2) / 2
            )

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        if self.og_sigma < 0:
            self.learnable = 1

            if dim == 1:
                self.conv = nn.Conv1d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=kernel_size,
                    padding=self.pad,
                )
            elif dim == 2:
                self.conv = nn.Conv2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=kernel_size,
                    padding=self.pad,
                )
            elif dim == 3:
                self.conv = nn.Conv3d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=kernel_size,
                    padding=self.pad,
                )
            else:
                raise RuntimeError(f"Only 1, 2 and 3 dimensions are supported. Received {dim}.")

            self.conv.weight = nn.Parameter(torch.cat((kernel, kernel), dim=1))
            self.conv.bias = nn.Parameter(torch.zeros(self.conv.bias.shape))
        else:
            self.learnable = 0
            self.register_buffer("weight", kernel)
            self.groups = channels

            if dim == 1:
                self.conv = F.conv1d
            elif dim == 2:
                self.conv = F.conv2d
            elif dim == 3:
                self.conv = F.conv3d
            else:
                raise RuntimeError(f"Only 1, 2 and 3 dimensions are supported. Received {dim}.")

    def forward(self, input_tensor):
        if self.learnable == 1:
            return self.conv(input_tensor)
        return self.conv(input_tensor, weight=self.weight, groups=self.groups, padding=self.pad)


def DiffeoActivat(flow_field, size):
    """
    Apply the original TEDS bounded activation to the flow field.
    """

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
    Warp the prior shape using the displacement field.
    """

    disp_prior_shape = self.transformer(prior_shape, disp_field)
    return disp_prior_shape
