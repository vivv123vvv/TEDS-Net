import numpy as np
import torch
import torch.nn.functional as F


class dice_loss:
    """Dice 损失函数。"""

    def loss(self, y_true, y_pred, loss_mult=None):
        smooth = 1.0
        iflat = y_pred.reshape(-1)
        tflat = y_true.reshape(-1)

        intersection = (iflat * tflat).sum()
        dice = 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

        if loss_mult is not None:
            dice *= loss_mult

        return dice

    def np_loss(self, y_true, y_pred, loss_mult=None):
        return self.loss(y_true, y_pred, loss_mult).item()


class grad_loss:
    """兼容旧版的梯度正则损失。"""

    def __init__(self, params, penalty="l2"):
        self.penalty = penalty
        self.ndims = params.dataset.ndims

    def loss(self, _, y_pred, loss_mult=None):
        size = np.shape(y_pred)[2:]
        device = y_pred.device
        dtype = y_pred.dtype
        vectors = [torch.linspace(-1, 1, s, device=device, dtype=dtype) for s in size]
        grids = torch.meshgrid(*vectors, indexing="ij")
        grid = torch.stack(grids)
        flow_field = y_pred + grid.unsqueeze(0)

        dy = torch.abs(flow_field[:, :, 1:, :] - flow_field[:, :, :-1, :])
        dx = torch.abs(flow_field[:, :, :, 1:] - flow_field[:, :, :, :-1])

        if self.penalty == "l2":
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if loss_mult is not None:
            grad *= loss_mult

        return grad

    def np_loss(self, _, y_pred, loss_mult=None):
        return self.loss(_, y_pred, loss_mult).item()


class BoundarySDFLoss:
    """基于 SDF 的边界损失。"""

    def __init__(self, scale=5.0):
        self.scale = float(scale)

    def loss(self, target_sdf, pred_sdf, loss_mult=None):
        pred = torch.tanh(pred_sdf / self.scale)
        target = torch.tanh(target_sdf / self.scale)
        boundary = F.smooth_l1_loss(pred, target)
        if loss_mult is not None:
            boundary *= loss_mult
        return boundary


class BendingEnergyLoss:
    """二阶 bending energy，用于抑制剧烈折叠。"""

    def loss(self, flow, loss_mult=None):
        dxx = flow[:, :, :, 2:] - 2.0 * flow[:, :, :, 1:-1] + flow[:, :, :, :-2]
        dyy = flow[:, :, 2:, :] - 2.0 * flow[:, :, 1:-1, :] + flow[:, :, :-2, :]
        dxy = (
            flow[:, :, 1:, 1:]
            - flow[:, :, 1:, :-1]
            - flow[:, :, :-1, 1:]
            + flow[:, :, :-1, :-1]
        )
        bending = dxx.square().mean() + dyy.square().mean() + dxy.square().mean()
        if loss_mult is not None:
            bending *= loss_mult
        return bending


class JacobianBarrierLoss:
    """Jacobian barrier，惩罚 detJ 低于阈值。"""

    def __init__(self, epsilon=0.01):
        self.epsilon = float(epsilon)

    def _determinant(self, flow):
        height, width = flow.shape[2:]
        spacing_y = 2.0 / max(height - 1, 1)
        spacing_x = 2.0 / max(width - 1, 1)

        du_dy = (flow[:, 0, 1:, :] - flow[:, 0, :-1, :]) / spacing_y
        du_dx = (flow[:, 0, :, 1:] - flow[:, 0, :, :-1]) / spacing_x
        dv_dy = (flow[:, 1, 1:, :] - flow[:, 1, :-1, :]) / spacing_y
        dv_dx = (flow[:, 1, :, 1:] - flow[:, 1, :, :-1]) / spacing_x

        du_dy = du_dy[:, :, :-1]
        dv_dy = dv_dy[:, :, :-1]
        du_dx = du_dx[:, :-1, :]
        dv_dx = dv_dx[:, :-1, :]
        return (1.0 + du_dy) * (1.0 + dv_dx) - du_dx * dv_dy

    def loss(self, flow, loss_mult=None):
        determinant = self._determinant(flow)
        barrier = F.relu(self.epsilon - determinant).square().mean()
        if loss_mult is not None:
            barrier *= loss_mult
        return barrier


class PoseRegressionLoss:
    """pose head 的回归损失。"""

    def loss(self, target_pose, pred_pose, loss_mult=None):
        pose = F.smooth_l1_loss(pred_pose, target_pose)
        if loss_mult is not None:
            pose *= loss_mult
        return pose


class RingValidityLoss:
    """在极坐标下约束环形结构仍为单环。"""

    def __init__(self, params):
        self.num_angles = int(params.dataset.projection_angles)
        self.margin = float(params.dataset.topology_margin)
        self.base_outer_radius = float(params.dataset.ps_meas[0])
        self.base_inner_radius = max(float(params.dataset.ps_meas[0] - params.dataset.ps_meas[1]), 1.0)

    def _build_polar_grid(self, mask_prob, pose_params):
        batch, _, height, width = mask_prob.shape
        device = mask_prob.device
        dtype = mask_prob.dtype

        radii = torch.linspace(0.0, float(min(height, width) // 2 - 1), min(height, width) // 2, device=device, dtype=dtype)
        angles = torch.linspace(-np.pi, np.pi, self.num_angles, device=device, dtype=dtype)

        centre_x = ((pose_params[:, 0] + 1.0) * 0.5) * (width - 1)
        centre_y = ((pose_params[:, 1] + 1.0) * 0.5) * (height - 1)

        radii_grid = radii.view(1, 1, -1).expand(batch, self.num_angles, -1)
        angles_grid = angles.view(1, -1, 1).expand(batch, -1, radii.shape[0])
        x = centre_x.view(batch, 1, 1) + radii_grid * torch.cos(angles_grid)
        y = centre_y.view(batch, 1, 1) + radii_grid * torch.sin(angles_grid)

        x = (2.0 * x / max(width - 1, 1)) - 1.0
        y = (2.0 * y / max(height - 1, 1)) - 1.0
        grid = torch.stack((x, y), dim=-1)
        return grid, radii

    def loss(self, mask_prob, pose_params, loss_mult=None):
        grid, radii = self._build_polar_grid(mask_prob, pose_params)
        polar = F.grid_sample(
            mask_prob,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(1)

        outer_radius = self.base_outer_radius * pose_params[:, 2].clamp(min=0.5, max=1.5)
        inner_radius = self.base_inner_radius * pose_params[:, 3].clamp(min=0.4, max=1.5)
        inner_radius = torch.minimum(inner_radius, outer_radius - self.margin)

        radii = radii.view(1, 1, -1)
        sharpness = 1.5
        centre_weight = torch.sigmoid((inner_radius.view(-1, 1, 1) - self.margin * 0.5 - radii) / sharpness)
        ring_left = torch.sigmoid((radii - (inner_radius.view(-1, 1, 1) + self.margin * 0.5)) / sharpness)
        ring_right = torch.sigmoid(((outer_radius.view(-1, 1, 1) - self.margin * 0.5) - radii) / sharpness)
        ring_weight = ring_left * ring_right
        outside_weight = torch.sigmoid((radii - (outer_radius.view(-1, 1, 1) + self.margin * 0.5)) / sharpness)

        centre_loss = (polar * centre_weight).sum() / centre_weight.sum().clamp_min(1e-6)
        ring_loss = ((1.0 - polar) * ring_weight).sum() / ring_weight.sum().clamp_min(1e-6)
        outside_loss = (polar * outside_weight).sum() / outside_weight.sum().clamp_min(1e-6)

        transitions = torch.abs(polar[:, :, 1:] - polar[:, :, :-1]).sum(dim=-1)
        transition_loss = torch.abs(transitions - 2.0).mean()

        total = centre_loss + ring_loss + outside_loss + 0.5 * transition_loss
        if loss_mult is not None:
            total *= loss_mult
        return total
