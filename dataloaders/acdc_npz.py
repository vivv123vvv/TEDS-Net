import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.acdc_benchmark import sample_id_from_path


class ACDCNpzDataset(Dataset):
    def __init__(
        self,
        data_dir,
        file_list=None,
        include_metadata=False,
        mode=None,
        dataset_id="acdc",
        image_key="image",
        prior_key="prior",
        label_key="label",
        target_rule="label_equals_value",
        label_value=2,
        case_id_pattern=r"^(patient\d+_frame\d+)",
    ):
        self.data_dir = Path(data_dir)
        self.include_metadata = include_metadata
        self.mode = mode
        self.dataset_id = dataset_id
        self.image_key = image_key
        self.prior_key = prior_key
        self.label_key = label_key
        self.target_rule = target_rule
        self.label_value = label_value
        self.case_id_pattern = re.compile(case_id_pattern) if case_id_pattern else None

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

    def _target_mask(self, label_tensor):
        if self.target_rule == "identity":
            return label_tensor.float()
        if self.target_rule == "nonzero":
            return (label_tensor > 0).float()
        if self.target_rule != "label_equals_value":
            raise ValueError(
                "Unsupported target_rule '{rule}' for dataset '{dataset}'".format(
                    rule=self.target_rule,
                    dataset=self.dataset_id,
                )
            )
        if self.label_value is None:
            return (label_tensor > 0).float()
        return (label_tensor == float(self.label_value)).float()

    def _case_id_from_sample_id(self, sample_id):
        if not self.case_id_pattern:
            return sample_id
        match = self.case_id_pattern.match(sample_id)
        return match.group(1) if match else sample_id

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        with np.load(file_path) as data:
            image = data[self.image_key]
            prior = data[self.prior_key]
            label = data[self.label_key]

        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        prior_tensor = torch.from_numpy(prior).float().unsqueeze(0)
        label_tensor = torch.from_numpy(label).float().unsqueeze(0)
        label_target = self._target_mask(label_tensor)

        if not self.include_metadata:
            return image_tensor, prior_tensor, label_target

        sample_id = sample_id_from_path(file_path)
        case_id = self._case_id_from_sample_id(sample_id)
        return image_tensor, prior_tensor, label_target, sample_id, case_id
