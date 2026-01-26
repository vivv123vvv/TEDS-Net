import os
from torch.utils.data import DataLoader, random_split
from dataloaders.acdc_npz import ACDCNpzDataset


def setup_acdc_dataloader(params, stages=['train', 'validation', 'test']):
    # 这里的 datapath 应该指向存放 .npz 文件的目录
    # 我们假设 params.dataset.datapath 会在 acdc_parameters.py 里被正确设置
    # 或者我们在这里强制指定预处理后的路径

    # 【注意】请确认这个路径是你存放 .npz 文件的路径
    processed_data_path = "./Resources/database/processed_2d"

    # 实例化完整数据集
    full_dataset = ACDCNpzDataset(processed_data_path)

    # 简单的划分数据集 (70% 训练, 20% 验证, 10% 测试)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    dataloader_dic = {}

    if 'train' in stages:
        dataloader_dic['train'] = DataLoader(
            train_set,
            batch_size=params.batch,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

    if 'validation' in stages:
        dataloader_dic['validation'] = DataLoader(
            val_set,
            batch_size=params.batch,
            shuffle=False,
            num_workers=0,
            drop_last=True  # 避免只有1个样本导致 BatchNorm 报错
        )

    if 'test' in stages:
        dataloader_dic['test'] = DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

    return dataloader_dic