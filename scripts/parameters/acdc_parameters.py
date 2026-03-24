from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List, Type
from enforce_typing import enforce_types


@enforce_types
@dataclass_json
@dataclass
class ACDC_dataset:
    """
    ACDC 数据集默认参数。
    """
    ndims: int = 2
    inshape: List = field(default_factory=lambda: [144, 208])
    ps_meas: List = field(default_factory=lambda: [35, 7])  # 先验形状尺寸参数
    datapath: str = "<< PATH TO ACDC DATASET >>"
    betti: List = field(default_factory=lambda: [1, 1, 0, 0])


@enforce_types
@dataclass_json
@dataclass
class TEDS_Arch:
    """
    TEDS-Net 结构默认参数。
    """

    # TEDS-Net 相关参数
    act: int = 1  # 是否启用激活函数
    diffeo_int: int = 8  # 网络内部的积分步数

    # 高斯平滑相关参数
    guas_smooth: int = 1  # 是否在复合过程中加入高斯平滑
    Guas_kernel: int = 5  # 平滑核大小
    sigma: float = 2.0  # 高斯核 sigma

    # 上采样参数
    mega_P: int = 2  # 位移场上采样倍数

    # 网络分支参数
    dec_depth: List = field(default_factory=lambda: [4, 2])  # bulk 分支输出于第 4 层，fine-tune 分支输出于第 2 层


@enforce_types
@dataclass_json
@dataclass
class GeneralNet:
    """
    通用网络结构默认参数。
    """

    dropout: int = 1  # 是否启用 dropout
    fi: int = 12  # 初始特征图数量
    net_depth: int = 4  # 网络深度
    in_chan: int = 1  # 输入通道数
    out_chan: int = 1  # 输出通道数


@enforce_types
@dataclass(frozen=True)
class LossParams:
    loss: List = field(default_factory=lambda: ["dice", "grad", "grad"])
    weight: List = field(default_factory=lambda: [1, 10000, 10000])


@enforce_types
@dataclass_json
@dataclass
class Parameters:

    # 训练参数
    epoch: int = 200
    lr: float = 0.0001
    lr_sch: bool = False
    batch: int = 5
    num_workers: int = 0
    checkpoint_freq: int = 50
    threshold: float = 0.3
    plot_predictions: bool = True
    max_train_batches: int = 0
    max_validation_batches: int = 0
    max_test_batches: int = 0

    # 损失函数参数
    loss_params: LossParams = field(default_factory=LossParams)

    # 网络超参数
    network_params: GeneralNet = field(default_factory=GeneralNet)

    # 数据集与网络配置
    data_path: str = "<< PATH TO ACDC DATASET >>"
    data: str = 'ACDC'
    dataset: ACDC_dataset = field(default_factory=ACDC_dataset)
    net: str = 'teds'
    network: TEDS_Arch = field(default_factory=TEDS_Arch)
