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


def setup_acdc_dataloader(params, subset_list):

    from dataloaders.ACDC import ACDC_dataclass as MyDataset

    params_train = _build_loader_kwargs(params, shuffle=True)
    params_val = _build_loader_kwargs(params, shuffle=False)
    params_test = _build_loader_kwargs(params, shuffle=False)

    # 此处需要替换成实际的数据划分字典
    dataset_dict = "<AMEND WITH YOUR DATALOADER DICTIONARY>"
    if isinstance(dataset_dict, str):
        raise NotImplementedError("ACDC 数据划分字典尚未配置，请先在 dataloaders/setup.py 中补全 dataset_dict。")

    # 构建 PyTorch DataLoader
    dataloader_dict = {}
    if 'train' in subset_list:
        training_set = MyDataset(params, dataset_dict['train'], subset='Train', aug=True)
        dataloader_dict['train'] = DataLoader(training_set, **params_train)

    if 'validation' in subset_list:
        val_set = MyDataset(params, dataset_dict['val'], subset='Train')
        dataloader_dict['validation'] = DataLoader(val_set, **params_val)
    if 'test' in subset_list:
        test_set = MyDataset(params, dataset_dict['test'], subset='Test')
        dataloader_dict['test'] = DataLoader(test_set, **params_test)

    return dataloader_dict
