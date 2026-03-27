import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.acdc_preprocess import load_manifest


class ACDCNpzDataset(Dataset):
    """读取预处理后的 ACDC `.npz` 缓存。"""

    def __init__(self, processed_data_path, records=None, return_metadata=False):
        self.processed_root = os.path.abspath(processed_data_path)
        self.return_metadata = return_metadata

        if records is None:
            manifest = load_manifest(self.processed_root)
            records = manifest["records"]

        self.records = list(records)
        if not self.records:
            raise RuntimeError(f"未在 `{self.processed_root}` 中找到可用的 ACDC 样本记录。")

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
        prior = torch.from_numpy(sample["prior"].astype(np.float32)).unsqueeze(0)
        label_array = sample["label"].astype(np.float32)
        if np.max(label_array) > 1.0:
            label_array = (label_array == 2).astype(np.float32)
        else:
            label_array = (label_array > 0.5).astype(np.float32)
        label = torch.from_numpy(label_array).unsqueeze(0)

        if not self.return_metadata:
            return image, prior, label

        metadata = {
            "relative_path": record["relative_path"],
            "patient_id": record["patient_id"],
            "frame_id": record["frame_id"],
            "slice_index": int(record["slice_index"]),
            "source_subset": record["source_subset"],
            "spacing": tuple(float(value) for value in record["spacing"]),
            "original_shape": tuple(int(value) for value in record["original_shape"]),
            "betti": tuple(int(value) for value in record["betti"]),
            "original_image": sample["original_image"].astype(np.float32),
            "original_label": sample["original_label"].astype(np.float32),
        }
        return image, prior, label, metadata
