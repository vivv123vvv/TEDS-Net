import os
from glob import glob

import nibabel as nib
import numpy as np
import raster_geometry as rg
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ACDC_dataclass(Dataset):
    """ACDC 2D 切片数据集。

    数据目录默认读取官方 ACDC 结构：
    `database/training/patientXXX` 与 `database/testing/patientXXX`。

    当前任务仅使用 myocardium 单类分割，对应标签值为 `2`。
    为了兼容当前 2D TEDS-Net 实现，这里会把 3D 体数据拆成 2D 切片。
    """

    def __init__(self, params, patient_ids, subset, aug=False):
        self.params = params
        self.aug = aug
        self.patient_ids = list(patient_ids)
        self.subset = subset
        self.data_root = self._resolve_data_root(params.data_path)

        assert subset in ['Train', 'Validation', 'Test']

        # 生成环形先验形状
        rad, thick = params.dataset.ps_meas
        height, width = params.dataset.inshape
        self.prior = rg.circle((height, width), radius=rad).astype(int) - rg.circle(
            (height, width),
            radius=(rad - thick),
        ).astype(int)

        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(f"ACDC {subset} 子集为空，请检查数据路径与切分逻辑。")
        print(f"ACDC {subset} 切片数: {len(self.samples)}，病人数: {len(self.patient_ids)}")

    @staticmethod
    def _resolve_data_root(data_path):
        """兼容传入 Resources 根目录或 database 根目录两种写法。"""

        candidates = [
            data_path,
            os.path.join(data_path, 'database'),
        ]
        for candidate in candidates:
            train_dir = os.path.join(candidate, 'training')
            test_dir = os.path.join(candidate, 'testing')
            if os.path.isdir(train_dir) and os.path.isdir(test_dir):
                return candidate

        raise FileNotFoundError(
            f"未找到 ACDC 数据目录。期望在 `{data_path}` 或其下的 `database/` 中包含 training 与 testing 子目录。"
        )

    def _build_samples(self):
        samples = []
        for patient_id in self.patient_ids:
            patient_dir = self._find_patient_dir(patient_id)
            gt_paths = sorted(glob(os.path.join(patient_dir, '*_gt.nii.gz')))
            if not gt_paths:
                continue

            for gt_path in gt_paths:
                image_path = gt_path.replace('_gt.nii.gz', '.nii.gz')
                if not os.path.isfile(image_path):
                    continue

                image_volume = nib.load(image_path).get_fdata().astype(np.float32)
                label_volume = nib.load(gt_path).get_fdata().astype(np.int16)

                if image_volume.ndim != 3 or label_volume.ndim != 3:
                    raise RuntimeError(f"仅支持 3D ACDC 体数据，收到文件: {image_path}")

                for slice_index in range(image_volume.shape[-1]):
                    image_slice = image_volume[:, :, slice_index]
                    label_slice = (label_volume[:, :, slice_index] == 2).astype(np.float32)

                    # 仅保留含 myocardium 的切片，避免绝大多数空白层稀释训练信号。
                    if not np.any(label_slice):
                        continue

                    image_slice = self._resize_slice(image_slice, is_label=False)
                    label_slice = self._resize_slice(label_slice, is_label=True)

                    samples.append((image_slice, label_slice, patient_id, slice_index))

        return samples

    def _find_patient_dir(self, patient_id):
        for subset_name in ('training', 'testing'):
            candidate = os.path.join(self.data_root, subset_name, patient_id)
            if os.path.isdir(candidate):
                return candidate

        raise FileNotFoundError(f"未找到病人目录: {patient_id}")

    def _resize_slice(self, array_2d, is_label):
        target_size = tuple(self.params.dataset.inshape)
        tensor = torch.from_numpy(array_2d).unsqueeze(0).unsqueeze(0)
        mode = 'nearest' if is_label else 'bilinear'
        kwargs = {} if is_label else {'align_corners': False}
        tensor = F.interpolate(tensor, size=target_size, mode=mode, **kwargs)
        array_2d = tensor.squeeze(0).squeeze(0).numpy().astype(np.float32)

        if is_label:
            return (array_2d > 0.5).astype(np.float32)

        min_value = float(array_2d.min())
        max_value = float(array_2d.max())
        if max_value > min_value:
            array_2d = (array_2d - min_value) / (max_value - min_value)
        else:
            array_2d = np.zeros_like(array_2d, dtype=np.float32)

        return array_2d

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_slice, label_slice, _, _ = self.samples[idx]

        image = np.expand_dims(image_slice, axis=0)
        label = np.expand_dims(label_slice, axis=0)
        prior_shape = np.expand_dims(self.prior, axis=0)

        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        prior_shape = torch.from_numpy(prior_shape.astype(np.float32))

        return image, prior_shape, label
