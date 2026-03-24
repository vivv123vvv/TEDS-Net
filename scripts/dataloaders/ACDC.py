import os
import numpy as np
import raster_geometry as rg
import torch
from torch.utils.data import Dataset


class ACDC_dataclass(Dataset):
    """ACDC 数据集包装类。

    使用的 ACDC 数据集可从
    https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html 获取。
    当前任务仅使用心肌分割标签（label 2）。

    新使用者需要完成以下配置：
    1) 添加各数据子集对应的 ID 列表
    2) 检查参数文件中的数据路径配置，例如 `params.data_path`
    """

    def __init__(self,
                 params,
                 list_ids,
                 subset,
                 aug=False,

                 ):
        self.params = params
        self.aug = aug

        assert subset in ['Train', 'Test']

        self.list_IDS = list_ids

        # 生成环形先验形状
        rad, thick = params.dataset.ps_meas
        M, N = params.dataset.inshape
        self.prior = rg.circle((M, N), radius=rad).astype(int) - rg.circle((M, N), radius=(rad - thick)).astype(int)

    def __len__(self):
        # 返回当前子集中的体数据数量
        return len(self.list_IDS)

    def __getitem__(self, idx):

        ID = self.list_IDS[idx]

        # 读取体数据与分割标签
        x = np.load(os.path.join(self.params.data_path, f"Vol/{ID}"))
        y_seg = np.load(os.path.join(self.params.data_path, f"Seg/{ID}"))

        # 转换为 PyTorch 使用的张量格式
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x.astype(np.float32))
        y_seg = np.expand_dims(y_seg, axis=0)
        y_seg = torch.from_numpy(y_seg.astype(np.float32))
        prior_shape = np.expand_dims(self.prior, 0)
        prior_shape = torch.from_numpy(prior_shape.astype(np.float32))

        return x, prior_shape, y_seg
