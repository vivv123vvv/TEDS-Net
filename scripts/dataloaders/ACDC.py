import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from utils.acdc_preprocess import binary_mask_to_sdf, load_manifest


class ACDC_dataclass(Dataset):
    """读取预处理后的 ACDC 2D 缓存。"""

    def __init__(self, params, records, subset, return_metadata=False):
        self.params = params
        self.records = list(records)
        self.subset = subset
        self.return_metadata = return_metadata
        self.processed_root = os.path.abspath(params.dataset.processed_data_path)
        self.augmentation = params.dataset.augmentation
        self.training = subset.lower() == "train"

        if not self.records:
            raise RuntimeError(f"ACDC {subset} 子集为空，请先检查预处理缓存与划分逻辑。")

    @staticmethod
    def get_manifest(processed_data_path):
        return load_manifest(processed_data_path)

    def __len__(self):
        return len(self.records)

    def _maybe_augment(self, image, label, pose_target):
        if not self.training or not self.augmentation.enabled:
            label_sdf = binary_mask_to_sdf(label.squeeze(0).numpy())
            return image, label, torch.from_numpy(label_sdf).float(), pose_target

        angle = random.uniform(-self.augmentation.rotate_deg, self.augmentation.rotate_deg)
        translate = [
            int(random.uniform(-self.augmentation.translate_px, self.augmentation.translate_px)),
            int(random.uniform(-self.augmentation.translate_px, self.augmentation.translate_px)),
        ]
        scale = random.uniform(self.augmentation.scale_min, self.augmentation.scale_max)

        image = TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
        )
        label = TF.affine(
            label,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
        )

        image = TF.adjust_brightness(
            image,
            1.0 + random.uniform(-self.augmentation.brightness, self.augmentation.brightness),
        )
        image = TF.adjust_contrast(
            image,
            1.0 + random.uniform(-self.augmentation.contrast, self.augmentation.contrast),
        )
        image = TF.adjust_gamma(
            image,
            gamma=max(0.7, 1.0 + random.uniform(-self.augmentation.gamma, self.augmentation.gamma)),
        )
        image = torch.clamp(
            image + torch.randn_like(image) * self.augmentation.noise_std,
            min=0.0,
            max=1.0,
        )

        label = (label > 0.5).float()
        label_np = label.squeeze(0).cpu().numpy().astype(np.float32)
        if not np.any(label_np):
            label_sdf = binary_mask_to_sdf(label_np)
            return image, label, torch.from_numpy(label_sdf).float(), pose_target

        betti = load_pose = None
        try:
            from utils.acdc_preprocess import calculate_betti_numbers, _estimate_annulus_pose

            betti = calculate_betti_numbers(label_np)
            load_pose = _estimate_annulus_pose(
                label_np,
                base_outer_radius=float(self.params.dataset.ps_meas[0]),
                base_inner_radius=max(float(self.params.dataset.ps_meas[0] - self.params.dataset.ps_meas[1]), 1.0),
            )
        except Exception:
            betti = None

        if betti != (1, 1):
            label_sdf = binary_mask_to_sdf(label.squeeze(0).cpu().numpy().astype(np.float32))
            return image, label, torch.from_numpy(label_sdf).float(), pose_target

        label_sdf = binary_mask_to_sdf(label_np)
        pose_target = torch.from_numpy(load_pose).float()
        return image, label, torch.from_numpy(label_sdf).float(), pose_target

    def __getitem__(self, idx):
        record = self.records[idx]
        sample_path = os.path.join(self.processed_root, record["relative_path"])
        sample = np.load(sample_path)

        image = torch.from_numpy(sample["image"].astype(np.float32)).unsqueeze(0)
        prior_shape = torch.from_numpy(sample["prior_sdf"].astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(sample["label"].astype(np.float32)).unsqueeze(0)
        pose_target = torch.from_numpy(sample["pose_target"].astype(np.float32))

        image, label, label_sdf_single, pose_target = self._maybe_augment(
            image=image,
            label=label,
            pose_target=pose_target,
        )
        target = {
            "mask": label.float(),
            "sdf": label_sdf_single.unsqueeze(0) if label_sdf_single.ndim == 2 else label_sdf_single.float(),
            "pose": pose_target.float(),
        }

        if not self.return_metadata:
            return image.float(), prior_shape.float(), target

        metadata = {
            "relative_path": record["relative_path"],
            "patient_id": record["patient_id"],
            "frame_id": record["frame_id"],
            "slice_index": int(record["slice_index"]),
            "source_subset": record["source_subset"],
            "spacing": tuple(float(v) for v in record["spacing"]),
            "original_shape": tuple(int(v) for v in record["original_shape"]),
            "betti": tuple(int(v) for v in record["betti"]),
            "original_image": sample["original_image"].astype(np.float32),
            "original_label": sample["original_label"].astype(np.float32),
            "pose_target": tuple(float(v) for v in sample["pose_target"]),
        }
        return image.float(), prior_shape.float(), target, metadata
