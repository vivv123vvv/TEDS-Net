import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn.utils import spectral_norm

from network.UNet import DecoderBranch


def _normalized_grid(flow):
    height, width = flow.shape[2:]
    y_axis = torch.linspace(-1.0, 1.0, height, device=flow.device, dtype=flow.dtype)
    x_axis = torch.linspace(-1.0, 1.0, width, device=flow.device, dtype=flow.dtype)
    grid_y, grid_x = torch.meshgrid(y_axis, x_axis, indexing="ij")
    return torch.stack((grid_y, grid_x), dim=0).unsqueeze(0).repeat(flow.shape[0], 1, 1, 1)


def compose_backward_flow_torch(first_flow, second_flow):
    """组合 backward warping flow。"""

    if first_flow.shape != second_flow.shape:
        raise ValueError(
            f"待组合 flow 形状不一致: first={first_flow.shape}, second={second_flow.shape}"
        )

    grid = _normalized_grid(second_flow)
    new_locations = grid + second_flow
    sampling_grid = new_locations.permute(0, 2, 3, 1)[..., [1, 0]]
    sampled_first = F.grid_sample(
        first_flow,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return second_flow + sampled_first


class WholeDiffeoUnit(nn.Module):
    """形变模块。"""

    def __init__(self, params, branch=0):
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
        self.integrator_name = getattr(params.network, "integrator", "scaling_squaring")
        self.r2net_blocks = getattr(params.network, "r2net_blocks", 7)
        self.sdf_temperature = float(getattr(params.dataset, "sdf_temperature", 1.5))
        self.pose_enabled = bool(getattr(params.network, "pose_enabled", False))
        self.base_outer_radius = float(params.dataset.ps_meas[0])
        self.base_inner_radius = max(float(params.dataset.ps_meas[0] - params.dataset.ps_meas[1]), 1.0)
        self.pose_margin = float(getattr(params.dataset, "topology_margin", 3.0))

        self.dec = DecoderBranch(
            features=self.features,
            ndims=self.ndims,
            net_depth=self.net_depth,
            dec_depth=self.dec_depth,
            dropout=self.dropout,
        )

        frac_size_change = [1, 2, 4, 8]
        self.flow_field_size = [int(s / frac_size_change[self.dec_depth - 1]) for s in self.inshape]
        self.Mega_inshape = [s * self.mega_P for s in self.inshape]
        self.gen_field = GenDisField(self.dec_depth, self.features, self.ndims)

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
            step_alpha=getattr(params.network, "step_alpha", 0.08),
        )
        self.transformer = mw_SpatialTransformer(self.Mega_inshape)
        self.prior_generator = AnnulusPriorGenerator(
            size=self.Mega_inshape,
            base_outer_radius=self.base_outer_radius * self.mega_P,
            base_inner_radius=self.base_inner_radius * self.mega_P,
            margin=self.pose_margin * self.mega_P,
        )
        bottleneck_channels = (2 ** (self.net_depth - 1)) * self.features
        self.pose_head = PoseHead(
            in_channels=bottleneck_channels,
            translation_limit=getattr(params.dataset, "pose_translation_limit", 0.35),
            scale_limit=getattr(params.dataset, "pose_scale_limit", 0.35),
        )

    def forward(self, bottleneck, enc_outputs, prior_shape):
        dec_output = self.dec(bottleneck, enc_outputs)
        flow_field = self.gen_field(dec_output)

        if self.integrator_name != "lc_resnet_constrained":
            flow_upsamp = self.flow_integrator(flow_field, self.act, self.viscous, self.ndims)
            sampled = WarpPriorShape(self.transformer, prior_shape, flow_upsamp)
            return flow_field, flow_upsamp, sampled

        pose_params = self.pose_head(bottleneck) if self.pose_enabled else self.pose_head.identity(
            batch_size=bottleneck.shape[0],
            device=bottleneck.device,
            dtype=bottleneck.dtype,
        )
        posed_prior_sdf = self.prior_generator(pose_params).to(bottleneck.dtype)
        _, flow_upsamp, per_step_flows = self.flow_integrator(flow_field)
        warped_sdf_mega = WarpPriorShape(self.transformer, posed_prior_sdf, flow_upsamp)

        warped_sdf = F.interpolate(
            warped_sdf_mega,
            size=tuple(self.inshape),
            mode="bilinear",
            align_corners=False,
        )
        posed_prior_sdf = F.interpolate(
            posed_prior_sdf,
            size=tuple(self.inshape),
            mode="bilinear",
            align_corners=False,
        )
        final_mask = torch.sigmoid(-warped_sdf / self.sdf_temperature)
        posed_prior_mask = torch.sigmoid(-posed_prior_sdf / self.sdf_temperature)

        return {
            "final_mask": final_mask,
            "final_sdf": warped_sdf,
            "posed_prior_mask": posed_prior_mask,
            "posed_prior_sdf": posed_prior_sdf,
            "composed_flow": flow_upsamp,
            "per_step_flows": per_step_flows,
            "pose_params": pose_params,
        }


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
    step_alpha,
):
    """根据配置选择积分器实现。"""

    if integrator_name == "scaling_squaring":
        return DiffeoUnit(
            flow_field_size,
            mega_size,
            int_steps=int_steps,
            viscous=viscous,
            Guas_kernel=Guas_kernel,
            Guas_P=Guas_P,
            mega_P=mega_P,
        )
    if integrator_name == "lc_resnet_constrained":
        return LCResNetConstrainedComposer(
            flow_field_size=flow_field_size,
            mega_size=mega_size,
            ndims=ndims,
            n_blocks=r2net_blocks,
            step_alpha=step_alpha,
        )
    raise ValueError(f"不支持的积分器: {integrator_name}")


class PoseHead(nn.Module):
    """预测全局平移与内外半径缩放。"""

    def __init__(self, in_channels, translation_limit=0.35, scale_limit=0.35):
        super().__init__()
        self.translation_limit = float(translation_limit)
        self.scale_limit = float(scale_limit)
        hidden = max(32, in_channels // 2)
        self.fc1 = nn.Linear(in_channels, hidden)
        self.fc2 = nn.Linear(hidden, 4)

    def identity(self, batch_size, device, dtype):
        pose = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        pose[:, 2:] = 1.0
        return pose

    def forward(self, bottleneck):
        pooled = F.adaptive_avg_pool2d(bottleneck, output_size=1).flatten(1)
        features = F.relu(self.fc1(pooled))
        raw = self.fc2(features)

        tx = torch.tanh(raw[:, 0]) * self.translation_limit
        ty = torch.tanh(raw[:, 1]) * self.translation_limit
        outer_scale = 1.0 + torch.tanh(raw[:, 2]) * self.scale_limit
        inner_scale = 1.0 + torch.tanh(raw[:, 3]) * self.scale_limit
        return torch.stack((tx, ty, outer_scale, inner_scale), dim=1)


class AnnulusPriorGenerator(nn.Module):
    """根据 pose 参数在高分辨率网格上生成 annulus SDF。"""

    def __init__(self, size, base_outer_radius, base_inner_radius, margin):
        super().__init__()
        self.size = size
        self.base_outer_radius = float(base_outer_radius)
        self.base_inner_radius = float(base_inner_radius)
        self.margin = float(margin)

        yy, xx = torch.meshgrid(
            torch.arange(size[0], dtype=torch.float32),
            torch.arange(size[1], dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("yy", yy)
        self.register_buffer("xx", xx)

    def forward(self, pose_params):
        batch_size = pose_params.shape[0]
        height, width = self.size
        centre_x = ((pose_params[:, 0] + 1.0) * 0.5) * (width - 1)
        centre_y = ((pose_params[:, 1] + 1.0) * 0.5) * (height - 1)

        outer_radius = self.base_outer_radius * pose_params[:, 2].clamp(min=0.5, max=1.5)
        inner_radius = self.base_inner_radius * pose_params[:, 3].clamp(min=0.4, max=1.5)
        inner_radius = torch.minimum(inner_radius, outer_radius - self.margin)
        outer_radius = torch.maximum(outer_radius, inner_radius + self.margin)

        distance = torch.sqrt(
            (self.yy.unsqueeze(0) - centre_y.view(batch_size, 1, 1)) ** 2
            + (self.xx.unsqueeze(0) - centre_x.view(batch_size, 1, 1)) ** 2
        )
        outer_sdf = distance - outer_radius.view(batch_size, 1, 1)
        inner_sdf = inner_radius.view(batch_size, 1, 1) - distance
        annulus_sdf = torch.maximum(outer_sdf, inner_sdf)
        return annulus_sdf.unsqueeze(1)


class LCResNetStepBlock(nn.Module):
    """单步残差位移预测块。"""

    def __init__(self, ndims, hidden_channels=16):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(ndims * 2, hidden_channels, kernel_size=3, padding=1, bias=False))
        self.norm1 = nn.InstanceNorm2d(hidden_channels)
        self.conv2 = spectral_norm(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False))
        self.norm2 = nn.InstanceNorm2d(hidden_channels)
        self.conv3 = spectral_norm(nn.Conv2d(hidden_channels, ndims, kernel_size=1, padding=0, bias=False))

    def forward(self, current_flow, base_flow):
        x = torch.cat((current_flow, base_flow), dim=1)
        x = F.leaky_relu(self.norm1(self.conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.norm2(self.conv2(x)), negative_slope=0.2)
        return self.conv3(x)


class LCResNetConstrainedComposer(nn.Module):
    """用多个小步残差位移组合成最终 flow。"""

    def __init__(self, flow_field_size, mega_size, ndims, n_blocks=7, step_alpha=0.08):
        super().__init__()
        if ndims != 2:
            raise RuntimeError("当前 LC-ResNet 约束组合器仅支持 2D。")

        self.Mega_inshape = mega_size
        self.step_alpha = float(step_alpha)
        self.n_blocks = max(1, int(n_blocks))
        hidden = max(16, ndims * 8)
        self.blocks = nn.ModuleList([LCResNetStepBlock(ndims=ndims, hidden_channels=hidden) for _ in range(self.n_blocks - 1)])
        self.upsample = nn.Upsample(size=self.Mega_inshape, mode="bilinear", align_corners=False)

    def forward(self, flow_field, *_):
        delta = self.step_alpha * torch.tanh(flow_field)
        composed = delta
        steps = [delta]

        for block in self.blocks:
            residual = self.step_alpha * torch.tanh(block(composed, flow_field))
            composed = compose_backward_flow_torch(composed, residual)
            steps.append(residual)

        per_step_flows = torch.stack(steps, dim=1)
        return composed, self.upsample(composed), per_step_flows


class GenDisField(nn.Module):
    """根据 U-Net 输出生成位移场。"""

    def __init__(self, layer_nb, features, ndims):
        super().__init__()

        if ndims == 3:
            from torch.nn import Conv3d as ConvD
        elif ndims == 2:
            from torch.nn import Conv2d as ConvD
        else:
            raise RuntimeError(f"仅支持 2D/3D，收到 ndims={ndims}")

        dec_features = [1, 1, 2, 4]
        self.flow_field = ConvD(
            dec_features[layer_nb - 1] * features,
            out_channels=ndims,
            kernel_size=1,
        )
        self.flow_field.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_field.weight.shape))
        self.flow_field.bias = nn.Parameter(torch.zeros(self.flow_field.bias.shape))

    def forward(self, cnn_output):
        return self.flow_field(cnn_output)


class DiffeoUnit(nn.Module):
    """旧版 scaling and squaring 积分器。"""

    def __init__(self, flow_field_size, mega_size, int_steps=7, viscous=1, Guas_kernel=5, Guas_P=2, mega_P=2):
        super().__init__()
        self.flow_field_size = flow_field_size
        self.integrate_layer = mw_DiffeoLayer(flow_field_size, int_steps, Guas_kernel, Guas_P=Guas_P)
        self.Mega_inshape = mega_size
        modes = {2: "bilinear", 3: "trilinear"}
        self.MEGAsmoothing_upsample = nn.Upsample(self.Mega_inshape, mode=modes[len(flow_field_size)], align_corners=False)

    def forward(self, flow_field, act, viscous, ndims):
        if act:
            flow_field = DiffeoActivat(flow_field, self.flow_field_size)
        amplified_flow_field = self.integrate_layer(flow_field, viscous)
        return self.MEGAsmoothing_upsample(amplified_flow_field)


class mw_DiffeoLayer(nn.Module):
    """使用 scaling and squaring 对向量场积分。"""

    def __init__(self, inshape, nsteps, kernel=3, Guas_P=2):
        super().__init__()
        assert nsteps >= 0, f"nsteps should be >= 0, found: {nsteps}"
        self.nsteps = nsteps
        self.transformer = mw_SpatialTransformer(inshape)
        ndims = len(inshape)
        self.sigma = Guas_P
        self.SmthKernel = GaussianSmoothing(channels=ndims, kernel_size=kernel, sigma=Guas_P, dim=ndims)

    def forward(self, vec, viscous=1):
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
            if viscous:
                vec = self.SmthKernel(vec)
        return vec


class mw_SpatialTransformer(nn.Module):
    """PyTorch 版空间变换器。"""

    def __init__(self, size, mode="bilinear"):
        super().__init__()
        self.mode = mode
        vectors = [torch.linspace(-1, 1, s) for s in size]
        grids = torch.meshgrid(*vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0).type(torch.FloatTensor)
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
    """高斯平滑模块。"""

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
            *[torch.arange(size, dtype=torch.float32) for size in kernel_size],
            indexing="ij",
        )

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp((-((mgrid - mean) / std) ** 2) / 2)

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        if self.og_sigma < 0:
            self.learnable = 1
            if dim == 1:
                self.conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=self.pad)
            elif dim == 2:
                self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=self.pad)
            elif dim == 3:
                self.conv = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=self.pad)
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
    """旧版激活函数。"""

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


def WarpPriorShape(transformer, prior_shape, disp_field):
    """对一组先验形状施加位移场变换。"""

    return transformer(prior_shape, disp_field)
