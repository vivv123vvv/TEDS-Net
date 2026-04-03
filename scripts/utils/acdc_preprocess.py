import json
import math
import os
from glob import glob

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage import measure


MANIFEST_NAME = "manifest.json"


def calculate_betti_numbers(binary_mask):
    """计算二维二值掩码的 Betti 数。"""

    binary_mask = (binary_mask > 0).astype(np.uint8)
    if np.sum(binary_mask) == 0:
        return 0, 0

    _, component_count = measure.label(binary_mask, return_num=True, connectivity=1)
    euler_number = measure.euler_number(binary_mask, connectivity=1)
    hole_count = component_count - euler_number
    return int(component_count), int(hole_count)


def binary_mask_to_sdf(binary_mask):
    """将二值环形掩码转换为 signed distance field。"""

    mask = binary_mask.astype(bool)
    if not mask.any():
        return np.ones_like(binary_mask, dtype=np.float32)

    inside_distance = ndimage.distance_transform_edt(mask)
    outside_distance = ndimage.distance_transform_edt(~mask)
    sdf = outside_distance - inside_distance
    return sdf.astype(np.float32)


def resolve_acdc_root(raw_data_path):
    """兼容传入 Resources 根目录或 database 根目录。"""

    candidates = [
        raw_data_path,
        os.path.join(raw_data_path, "database"),
    ]
    for candidate in candidates:
        train_dir = os.path.join(candidate, "training")
        test_dir = os.path.join(candidate, "testing")
        if os.path.isdir(train_dir) and os.path.isdir(test_dir):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        f"未找到 ACDC 原始数据目录。期望 `{raw_data_path}` 或其下 `database/` 包含 training/testing。"
    )


def resolve_manifest_path(processed_data_path):
    """返回预处理 manifest 路径。"""

    return os.path.join(os.path.abspath(processed_data_path), MANIFEST_NAME)


def load_manifest(processed_data_path):
    """读取预处理 manifest。"""

    manifest_path = resolve_manifest_path(processed_data_path)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"未找到预处理 manifest: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["manifest_path"] = manifest_path
    return manifest


def _resize_slice(array_2d, target_size, is_label):
    tensor = torch.from_numpy(array_2d).unsqueeze(0).unsqueeze(0).float()
    mode = "nearest" if is_label else "bilinear"
    kwargs = {} if is_label else {"align_corners": False}
    resized = F.interpolate(tensor, size=tuple(target_size), mode=mode, **kwargs)
    resized = resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    if is_label:
        return (resized > 0.5).astype(np.float32)
    return resized


def _normalize_image(array_2d):
    min_value = float(array_2d.min())
    max_value = float(array_2d.max())
    if max_value > min_value:
        return ((array_2d - min_value) / (max_value - min_value)).astype(np.float32)
    return np.zeros_like(array_2d, dtype=np.float32)


def _annulus_sdf_from_geometry(shape, centre_y, centre_x, outer_radius, inner_radius):
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((yy - centre_y) ** 2 + (xx - centre_x) ** 2)
    outer_sdf = distance - float(outer_radius)
    inner_sdf = float(inner_radius) - distance
    annulus_sdf = np.maximum(outer_sdf, inner_sdf)
    annulus_mask = (annulus_sdf <= 0.0).astype(np.float32)
    return annulus_mask.astype(np.float32), annulus_sdf.astype(np.float32)


def _make_ring_prior(shape, radius, thickness):
    height, width = shape
    centre_y = (height - 1) / 2.0
    centre_x = (width - 1) / 2.0
    inner_radius = max(radius - thickness, 1.0)
    return _annulus_sdf_from_geometry(shape, centre_y, centre_x, radius, inner_radius)


def _estimate_annulus_pose(mask, base_outer_radius, base_inner_radius):
    mask = (mask > 0).astype(np.uint8)
    if not mask.any():
        return np.zeros(4, dtype=np.float32)

    filled_mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    hole_mask = np.logical_and(filled_mask == 1, mask == 0)
    coords = np.argwhere(mask > 0)
    centre_y, centre_x = coords.mean(axis=0).astype(np.float32)
    height, width = mask.shape

    tx = (2.0 * centre_x / max(width - 1, 1)) - 1.0
    ty = (2.0 * centre_y / max(height - 1, 1)) - 1.0

    outer_radius = math.sqrt(max(float(filled_mask.sum()), 1.0) / math.pi)
    inner_radius = math.sqrt(max(float(hole_mask.sum()), 1.0) / math.pi)

    outer_scale = outer_radius / max(base_outer_radius, 1e-6)
    inner_scale = inner_radius / max(base_inner_radius, 1e-6)
    pose_target = np.array([tx, ty, outer_scale, inner_scale], dtype=np.float32)
    return pose_target


def _empty_subset_stats():
    return {
        "patients": 0,
        "frames": 0,
        "total_slices": 0,
        "empty_slices": 0,
        "non_ring_slices": 0,
        "kept_slices": 0,
    }


def preprocess_acdc_dataset(
    raw_data_path,
    processed_data_path,
    target_size,
    prior_radius,
    prior_thickness,
    force=False,
    limit_patients=0,
):
    """将官方 ACDC NIfTI 预处理成训练/评估复用的 2D 缓存。"""

    raw_root = resolve_acdc_root(raw_data_path)
    processed_root = os.path.abspath(processed_data_path)
    manifest_path = resolve_manifest_path(processed_root)

    if os.path.isfile(manifest_path) and not force:
        return load_manifest(processed_root)

    os.makedirs(processed_root, exist_ok=True)
    prior_mask, prior_sdf = _make_ring_prior(target_size, prior_radius, prior_thickness)
    base_inner_radius = max(float(prior_radius - prior_thickness), 1.0)

    records = []
    summary = {
        "target_size": list(target_size),
        "prior_radius": int(prior_radius),
        "prior_thickness": int(prior_thickness),
        "subsets": {
            "training": _empty_subset_stats(),
            "testing": _empty_subset_stats(),
        },
    }

    for source_subset in ("training", "testing"):
        subset_root = os.path.join(raw_root, source_subset)
        patient_dirs = sorted(
            name for name in os.listdir(subset_root)
            if os.path.isdir(os.path.join(subset_root, name))
        )
        if limit_patients:
            patient_dirs = patient_dirs[:limit_patients]

        subset_stats = summary["subsets"][source_subset]
        subset_stats["patients"] = len(patient_dirs)
        subset_dir = os.path.join(processed_root, source_subset)
        os.makedirs(subset_dir, exist_ok=True)

        for patient_id in patient_dirs:
            patient_dir = os.path.join(subset_root, patient_id)
            gt_paths = sorted(glob(os.path.join(patient_dir, "*_gt.nii.gz")))
            if not gt_paths:
                continue

            subset_stats["frames"] += len(gt_paths)

            for gt_path in gt_paths:
                image_path = gt_path.replace("_gt.nii.gz", ".nii.gz")
                if not os.path.isfile(image_path):
                    continue

                frame_id = os.path.basename(image_path).split("_")[1].split(".")[0]
                image_nifti = nib.load(image_path)
                label_nifti = nib.load(gt_path)
                image_volume = image_nifti.get_fdata().astype(np.float32)
                label_volume = label_nifti.get_fdata().astype(np.int16)
                spacing = [
                    float(image_nifti.header.get_zooms()[0]),
                    float(image_nifti.header.get_zooms()[1]),
                ]

                if image_volume.ndim != 3 or label_volume.ndim != 3:
                    raise RuntimeError(f"仅支持 3D ACDC 体数据，收到: {image_path}")

                for slice_index in range(image_volume.shape[-1]):
                    subset_stats["total_slices"] += 1
                    image_slice = image_volume[:, :, slice_index]
                    label_slice = (label_volume[:, :, slice_index] == 2).astype(np.uint8)

                    if not np.any(label_slice):
                        subset_stats["empty_slices"] += 1
                        continue

                    betti = calculate_betti_numbers(label_slice)
                    if betti != (1, 1):
                        subset_stats["non_ring_slices"] += 1
                        continue

                    normalized_original_image = _normalize_image(image_slice)
                    resized_image = _resize_slice(normalized_original_image, target_size, is_label=False)
                    resized_label = _resize_slice(label_slice.astype(np.float32), target_size, is_label=True)
                    label_sdf = binary_mask_to_sdf(resized_label)
                    pose_target = _estimate_annulus_pose(
                        resized_label,
                        base_outer_radius=float(prior_radius),
                        base_inner_radius=base_inner_radius,
                    )

                    slice_name = f"{patient_id}_{frame_id}_slice{slice_index:03d}.npz"
                    relative_path = os.path.join(source_subset, slice_name)
                    output_path = os.path.join(processed_root, relative_path)
                    np.savez_compressed(
                        output_path,
                        image=resized_image.astype(np.float32),
                        label=resized_label.astype(np.float32),
                        label_sdf=label_sdf.astype(np.float32),
                        prior=prior_mask.astype(np.float32),
                        prior_sdf=prior_sdf.astype(np.float32),
                        pose_target=pose_target.astype(np.float32),
                        original_image=normalized_original_image.astype(np.float32),
                        original_label=label_slice.astype(np.float32),
                    )

                    records.append(
                        {
                            "relative_path": relative_path.replace("\\", "/"),
                            "source_subset": source_subset,
                            "patient_id": patient_id,
                            "frame_id": frame_id,
                            "slice_index": int(slice_index),
                            "original_shape": [int(image_slice.shape[0]), int(image_slice.shape[1])],
                            "spacing": spacing,
                            "betti": [int(betti[0]), int(betti[1])],
                            "pose_target": [float(v) for v in pose_target],
                        }
                    )
                    subset_stats["kept_slices"] += 1

    manifest = {
        "version": 2,
        "raw_data_root": raw_root,
        "processed_data_root": processed_root,
        "target_size": list(target_size),
        "prior": {
            "radius": int(prior_radius),
            "thickness": int(prior_thickness),
            "inner_radius": int(base_inner_radius),
        },
        "records": records,
        "summary": summary,
    }

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    manifest["manifest_path"] = manifest_path
    return manifest
