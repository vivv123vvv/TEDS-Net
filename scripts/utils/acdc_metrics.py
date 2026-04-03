import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage import measure


def dice_coefficient(pred_mask, target_mask):
    """计算二值 Dice。"""

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
    """当某一侧为空时使用切片对角线作为失败距离。"""

    height, width = mask_shape
    return float(math.sqrt((height * spacing[0]) ** 2 + (width * spacing[1]) ** 2))


def _surface_mask(binary_mask):
    structure = np.ones((3, 3), dtype=bool)
    eroded = ndimage.binary_erosion(binary_mask, structure=structure, border_value=0)
    return np.logical_xor(binary_mask, eroded)


def compute_distance_metrics(pred_mask, target_mask, spacing):
    """计算 HD、HD95 与 ASSD。"""

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
        raise ValueError(f"仅支持二维 flow，收到形状 {flow.shape}")

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


def _normalized_grid(flow):
    height, width = flow.shape[2:]
    y_axis = torch.linspace(-1.0, 1.0, height, device=flow.device, dtype=flow.dtype)
    x_axis = torch.linspace(-1.0, 1.0, width, device=flow.device, dtype=flow.dtype)
    grid_y, grid_x = torch.meshgrid(y_axis, x_axis, indexing="ij")
    grid = torch.stack((grid_y, grid_x), dim=0).unsqueeze(0).repeat(flow.shape[0], 1, 1, 1)
    return grid


def compose_backward_flows(first_flow, second_flow):
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


def project_probability_to_annulus(
    probability,
    pose_params,
    base_outer_radius,
    base_inner_radius,
    margin,
    threshold=0.5,
    num_angles=128,
):
    """将概率图投影到最近的单环形状。"""

    probability = np.asarray(probability, dtype=np.float32)
    height, width = probability.shape

    tx, ty, outer_scale, inner_scale = [float(v) for v in pose_params]
    centre_x = ((tx + 1.0) * 0.5) * (width - 1)
    centre_y = ((ty + 1.0) * 0.5) * (height - 1)

    predicted_outer = np.clip(base_outer_radius * outer_scale, margin + 2.0, min(height, width) / 2 - 1.0)
    predicted_inner = np.clip(base_inner_radius * inner_scale, 1.0, predicted_outer - margin)

    max_radius = max(8, min(height, width) // 2 - 1)
    radii = np.arange(max_radius, dtype=np.float32)
    angles = np.linspace(-math.pi, math.pi, num_angles, endpoint=False, dtype=np.float32)

    x = centre_x + np.cos(angles)[:, None] * radii[None, :]
    y = centre_y + np.sin(angles)[:, None] * radii[None, :]
    polar_prob = ndimage.map_coordinates(
        probability,
        [y, x],
        order=1,
        mode="nearest",
    )

    inner_radii = np.full(num_angles, predicted_inner, dtype=np.float32)
    outer_radii = np.full(num_angles, predicted_outer, dtype=np.float32)

    for angle_index in range(num_angles):
        active = np.flatnonzero(polar_prob[angle_index] >= threshold)
        if active.size == 0:
            continue
        inner_radii[angle_index] = float(active[0])
        outer_radii[angle_index] = float(active[-1])

    inner_radii = ndimage.gaussian_filter1d(inner_radii, sigma=2.0, mode="wrap")
    outer_radii = ndimage.gaussian_filter1d(outer_radii, sigma=2.0, mode="wrap")
    inner_radii = np.clip(inner_radii, 1.0, max_radius - margin - 1.0)
    outer_radii = np.maximum(outer_radii, inner_radii + margin)
    outer_radii = np.clip(outer_radii, inner_radii + margin, max_radius - 1.0)

    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    distance = np.sqrt((yy - centre_y) ** 2 + (xx - centre_x) ** 2)
    theta = np.arctan2(yy - centre_y, xx - centre_x)

    extended_angles = np.concatenate([angles - 2.0 * math.pi, angles, angles + 2.0 * math.pi])
    extended_inner = np.tile(inner_radii, 3)
    extended_outer = np.tile(outer_radii, 3)
    inner_map = np.interp(theta.reshape(-1), extended_angles, extended_inner).reshape(height, width)
    outer_map = np.interp(theta.reshape(-1), extended_angles, extended_outer).reshape(height, width)

    projected = np.logical_and(distance >= inner_map, distance <= outer_map).astype(np.float32)
    return projected
