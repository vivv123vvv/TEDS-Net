import os

import torch
from torch.utils.data import DataLoader


def _build_loader_kwargs(params, shuffle):
    """根据参数构建 DataLoader 关键字参数。"""

    worker_count = max(0, getattr(params, 'num_workers', 0))
    kwargs = {
        'batch_size': params.batch,
        'shuffle': shuffle,
        'num_workers': worker_count,
        'pin_memory': torch.cuda.is_available(),
    }
    if worker_count > 0:
        kwargs['persistent_workers'] = True

    return kwargs


def setup_mnist_dataloader(params, subset_list):

    from dataloaders.mnist import MNIST_dataclass as MyDataset

    params_train = _build_loader_kwargs(params, shuffle=True)
    params_val = _build_loader_kwargs(params, shuffle=False)
    params_test = _build_loader_kwargs(params, shuffle=False)

    dataloader_dict = {}
    if 'train' in subset_list:
        training_set = MyDataset(params, subset='Train')
        dataloader_dict['train'] = DataLoader(training_set, **params_train)
    if 'validation' in subset_list:
        val_set = MyDataset(params, subset='Validation')
        dataloader_dict['validation'] = DataLoader(val_set, **params_val)
    if 'test' in subset_list:
        test_set = MyDataset(params, subset='Test')
        dataloader_dict['test'] = DataLoader(test_set, **params_test)

    return dataloader_dict


def _resolve_acdc_root(data_path):
    """兼容传入 Resources 根目录或 database 根目录。"""

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


def _split_acdc_patients(data_path, validation_ratio=0.1):
    """按病人级别切分 train / validation / test，避免切片级泄漏。"""

    acdc_root = _resolve_acdc_root(data_path)
    train_root = os.path.join(acdc_root, 'training')
    test_root = os.path.join(acdc_root, 'testing')

    train_patients = sorted(
        name for name in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, name))
    )
    test_patients = sorted(
        name for name in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, name))
    )

    if len(train_patients) < 2:
        raise RuntimeError("ACDC training 病人数不足，无法继续切分 train / validation。")
    if not test_patients:
        raise RuntimeError("ACDC testing 目录为空，无法构建测试集。")

    validation_count = max(1, int(len(train_patients) * validation_ratio))
    validation_patients = train_patients[:validation_count]
    training_patients = train_patients[validation_count:]

    print(
        "ACDC 病人划分: "
        f"train={len(training_patients)}, "
        f"validation={len(validation_patients)}, "
        f"test={len(test_patients)}"
    )

    return {
        'train': training_patients,
        'val': validation_patients,
        'test': test_patients,
    }


def setup_acdc_dataloader(params, subset_list):

    from dataloaders.ACDC import ACDC_dataclass as MyDataset

    params_train = _build_loader_kwargs(params, shuffle=True)
    params_val = _build_loader_kwargs(params, shuffle=False)
    params_test = _build_loader_kwargs(params, shuffle=False)

    dataset_dict = _split_acdc_patients(params.data_path)

    # 构建 PyTorch DataLoader
    dataloader_dict = {}
    if 'train' in subset_list:
        training_set = MyDataset(params, dataset_dict['train'], subset='Train', aug=True)
        dataloader_dict['train'] = DataLoader(training_set, **params_train)

    if 'validation' in subset_list:
        val_set = MyDataset(params, dataset_dict['val'], subset='Validation')
        dataloader_dict['validation'] = DataLoader(val_set, **params_val)
    if 'test' in subset_list:
        test_set = MyDataset(params, dataset_dict['test'], subset='Test')
        dataloader_dict['test'] = DataLoader(test_set, **params_test)

    return dataloader_dict
