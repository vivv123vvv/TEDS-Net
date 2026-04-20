from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.acdc_benchmark import case_id_from_sample_id, sample_id_from_path


class ACDCNpzDataset(Dataset):
    def __init__(self, data_dir, file_list=None, include_metadata=False, mode=None):
        self.data_dir = Path(data_dir)
        self.include_metadata = include_metadata
        self.mode = mode

        if file_list is None:
            files = sorted(self.data_dir.glob("*.npz"))
        else:
            files = []
            for file_name in file_list:
                file_path = Path(file_name)
                if not file_path.is_absolute():
                    file_path = self.data_dir / file_path
                files.append(file_path)

        self.file_list = [Path(file_path) for file_path in files]
        if not self.file_list:
            print(f"Warning: no .npz files found under {self.data_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(file_path)

        image = data["image"]
        prior = data["prior"]
        label = data["label"]

        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        prior_tensor = torch.from_numpy(prior).float().unsqueeze(0)
        label_tensor = torch.from_numpy(label).float().unsqueeze(0)
        label_myo = (label_tensor == 2).float()

        if not self.include_metadata:
            return image_tensor, prior_tensor, label_myo

        sample_id = sample_id_from_path(file_path)
        case_id = case_id_from_sample_id(sample_id)
        return image_tensor, prior_tensor, label_myo, sample_id, case_id
