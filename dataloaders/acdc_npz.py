import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os


class ACDCNpzDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.file_list = glob.glob(os.path.join(data_dir, "*.npz"))
        if len(self.file_list) == 0:
            print(f"警告: 在 {data_dir} 未找到 .npz 文件")
        self.mode = mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])

        # 读取数据
        image = data['image']  # [H, W]
        prior = data['prior']  # [H, W]
        label = data['label']  # [H, W]

        # 转 Tensor 并增加 Channel 维度 -> [1, H, W]
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        prior_tensor = torch.from_numpy(prior).float().unsqueeze(0)

        # 处理标签：TEDS-Net 主要是二分类 (心肌 vs 背景)
        # 将 label=2 (心肌) 转为 1，其他为 0
        label_tensor = torch.from_numpy(label).float().unsqueeze(0)
        label_myo = (label_tensor == 2).float()

        return image_tensor, prior_tensor, label_myo