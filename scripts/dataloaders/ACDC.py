import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.acdc_preprocess import load_manifest


class ACDC_dataclass(Dataset):
    """读取预处理后的 ACDC 2D 缓存。"""

    def __init__(self, params, records, subset, return_metadata=False):
        self.params = params
        self.records = list(records)
        self.subset = subset
        self.return_metadata = return_metadata
        self.processed_root = os.path.abspath(params.dataset.processed_data_path)

        if not self.records:
            raise RuntimeError(f"ACDC {subset} 子集为空，请先检查预处理缓存与划分逻辑。")

    @staticmethod
    def get_manifest(processed_data_path):
        return load_manifest(processed_data_path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        sample_path = os.path.join(self.processed_root, record["relative_path"])
        sample = np.load(sample_path)

        image = torch.from_numpy(sample["image"].astype(np.float32)).unsqueeze(0)
        prior_shape = torch.from_numpy(sample["prior"].astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(sample["label"].astype(np.float32)).unsqueeze(0)

        if not self.return_metadata:
            return image, prior_shape, label

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
        }
        return image, prior_shape, label, metadata
