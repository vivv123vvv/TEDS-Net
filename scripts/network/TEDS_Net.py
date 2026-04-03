import torch.nn as nn

from network.UNet import BottleNeck, EncoderBranch
from network.utils_teds import WholeDiffeoUnit, compose_backward_flow_torch


class TEDS_Net(nn.Module):
    """TEDS-Net 主体网络。"""

    def __init__(self, params):
        super().__init__()

        in_channels = params.network_params.in_chan
        features = params.network_params.fi
        net_depth = params.network_params.net_depth
        dropout = params.network_params.dropout
        dec_depth = params.network.dec_depth
        self.no_branches = len(dec_depth)
        self.integrator_name = getattr(params.network, "integrator", "scaling_squaring")

        ndims = params.dataset.ndims
        self.enc = EncoderBranch(in_channels, features, ndims, net_depth, dropout)
        self.bottleneck = BottleNeck(features, ndims, net_depth, dropout)

        if self.integrator_name == "lc_resnet_constrained" and self.no_branches != 1:
            raise RuntimeError("LC-ResNet 约束替换版本当前仅支持单分支 dec_depth 配置。")

        if self.no_branches == 1:
            self.STN = WholeDiffeoUnit(params, branch=0)
        elif self.no_branches == 2:
            self.STN_bulk = WholeDiffeoUnit(params, branch=0)
            self.STN_ft = WholeDiffeoUnit(params, branch=1)
        else:
            raise RuntimeError(f"不支持的分支数量: {self.no_branches}")

    def forward(self, x, prior_shape):
        enc_outputs = self.enc(x)
        bottleneck = self.bottleneck(enc_outputs[-1])

        if self.no_branches == 1:
            return self.STN(bottleneck, enc_outputs, prior_shape)

        flow_bulk_field, flow_bulk_upsamp, bulk_sampled = self.STN_bulk(bottleneck, enc_outputs, prior_shape)
        flow_ft_field, flow_ft_upsamp, ft_sampled = self.STN_ft(bottleneck, enc_outputs, bulk_sampled)
        composed_flow = compose_backward_flow_torch(flow_bulk_upsamp, flow_ft_upsamp)

        return {
            "final_mask": ft_sampled,
            "final_sdf": ft_sampled,
            "posed_prior_mask": bulk_sampled,
            "posed_prior_sdf": bulk_sampled,
            "composed_flow": composed_flow,
            "per_step_flows": None,
            "pose_params": None,
            "compatibility_outputs": (flow_bulk_field, flow_ft_field),
        }
