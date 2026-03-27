import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage import measure


def dice_coefficient(pred_mask, target_mask):
    """计算二值分割 Dice。"""

    pred_mask = pred_mask.astype(bool)
    target_mask = target_mask.astype(bool)
    intersection = float(np.logical_and(pred_mask, target_mask).sum())
    denominator = float(pred_mask.sum() + target_mask.sum())
    if denominator == 0:
        return 1.0
    return (2.0 * intersection) / denominator


def calculate_betti_numbers(binary_mask):
    """计算二维二值掩码的 Betti 数。"""

    binary_mask = (binary_mask > 0).astype(np.uint8)
    if np.sum(binary_mask) == 0:
        return 0, 0

    _, component_count = measure.label(binary_mask, return_num=True, connectivity=1)
    euler_number = measure.euler_number(binary_mask, connectivity=1)
    hole_count = component_count - euler_number
    return int(component_count), int(hole_count)


def failure_distance(mask_shape, spacing):
    """当预测或标注为空时，使用切片对角线作为失败距离。"""

    height, width = mask_shape
    return float(math.sqrt((height * spacing[0]) ** 2 + (width * spacing[1]) ** 2))


def _surface_mask(binary_mask):
    structure = np.ones((3, 3), dtype=bool)
    eroded = ndimage.binary_erosion(binary_mask, structure=structure, border_value=0)
    return np.logical_xor(binary_mask, eroded)


def compute_hd_hd95_and_assd(pred_mask, target_mask, spacing):
    """同时计算 HD、HD95 与 ASSD。"""

    pred_mask = pred_mask.astype(bool)
    target_mask = target_mask.astype(bool)

    if not pred_mask.any() and not target_mask.any():
        return 0.0, 0.0, 0.0

    if not pred_mask.any() or not target_mask.any():
        fallback = failure_distance(pred_mask.shape, spacing)
        return fallback, fallback, fallback

    pred_surface = _surface_mask(pred_mask)
    target_surface = _surface_mask(target_mask)

    target_distance = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    pred_distance = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)

    pred_to_target = target_distance[pred_surface]
    target_to_pred = pred_distance[target_surface]
    distances = np.concatenate([pred_to_target, target_to_pred])

    hd = float(np.max(distances))
    hd95 = float(np.percentile(distances, 95))
    assd = float((pred_to_target.mean() + target_to_pred.mean()) / 2.0)
    return hd, hd95, assd


def jacobian_determinant_2d(flow):
    """计算二维归一化位移场的 Jacobian determinant。"""

    if flow.shape[0] != 2:
        raise ValueError(f"仅支持二维 flow，收到形状：{flow.shape}")

    height, width = flow.shape[1:]
    spacing_y = 2.0 / max(height - 1, 1)
    spacing_x = 2.0 / max(width - 1, 1)

    du_dy, du_dx = np.gradient(flow[0], spacing_y, spacing_x, edge_order=1)
    dv_dy, dv_dx = np.gradient(flow[1], spacing_y, spacing_x, edge_order=1)
    determinant = (1.0 + du_dy) * (1.0 + dv_dx) - (du_dx * dv_dy)
    return determinant


def folding_ratio(flow):
    """统计 Jacobian <= 0 的像素占比。"""

    determinant = jacobian_determinant_2d(flow)
    return float(np.mean(determinant <= 0.0))


def model_parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _normalized_grid(flow):
    height, width = flow.shape[2:]
    y_axis = torch.linspace(-1.0, 1.0, height, device=flow.device, dtype=flow.dtype)
    x_axis = torch.linspace(-1.0, 1.0, width, device=flow.device, dtype=flow.dtype)
    grid_y, grid_x = torch.meshgrid(y_axis, x_axis, indexing="ij")
    grid = torch.stack((grid_y, grid_x), dim=0).unsqueeze(0).repeat(flow.shape[0], 1, 1, 1)
    return grid


def compose_backward_flows(first_flow, second_flow):
    """复合 backward warping flow，使其等价于先 `first_flow` 后 `second_flow`。"""

    if first_flow.shape != second_flow.shape:
        raise ValueError(
            f"待复合 flow 形状不一致：first={first_flow.shape}, second={second_flow.shape}"
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
