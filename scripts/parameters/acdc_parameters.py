from dataclasses import dataclass, field
from typing import List

from dataclasses_json import dataclass_json
from enforce_typing import enforce_types


@enforce_types
@dataclass_json
@dataclass
class ACDC_dataset:
    """ACDC 数据与预处理参数。"""

    ndims: int = 2
    inshape: List[int] = field(default_factory=lambda: [144, 208])
    ps_meas: List[int] = field(default_factory=lambda: [35, 7])
    raw_data_path: str = "<< PATH TO RAW ACDC DATASET >>"
    processed_data_path: str = "results/preprocessed/acdc_ring_144x208"
    validation_ratio: float = 0.1
    betti: List[int] = field(default_factory=lambda: [1, 1, 0, 0])


@enforce_types
@dataclass_json
@dataclass
class TEDS_Arch:
    """TEDS-Net 结构默认参数。"""

    act: int = 1
    diffeo_int: int = 8
    guas_smooth: int = 1
    Guas_kernel: int = 5
    sigma: float = 2.0
    mega_P: int = 2
    dec_depth: List[int] = field(default_factory=lambda: [4, 2])
    integrator: str = "r2net_lc_resnet"
    r2net_blocks: int = 7


@enforce_types
@dataclass_json
@dataclass
class GeneralNet:
    """通用网络结构默认参数。"""

    dropout: int = 1
    fi: int = 12
    net_depth: int = 4
    in_chan: int = 1
    out_chan: int = 1


@enforce_types
@dataclass_json
@dataclass
class LossParams:
    loss: List[str] = field(default_factory=lambda: ["dice", "grad", "grad"])
    weight: List[float] = field(default_factory=lambda: [1, 10000, 10000])


@enforce_types
@dataclass_json
@dataclass
class Parameters:
    """ACDC 训练与评估参数。"""

    epoch: int = 200
    lr: float = 0.0001
    lr_sch: bool = False
    batch: int = 5
    eval_batch: int = 1
    num_workers: int = 0
    checkpoint_freq: int = 50
    threshold: float = 0.3
    plot_predictions: bool = True
    max_train_batches: int = 0
    max_validation_batches: int = 0
    max_test_batches: int = 0
    seed: int = 42
    data_path: str = "results/preprocessed/acdc_ring_144x208"
    run_name: str = "acdc_batch200"
    output_root: str = "results/acdc"
    evaluate_only: bool = False

    loss_params: LossParams = field(default_factory=LossParams)
    network_params: GeneralNet = field(default_factory=GeneralNet)
    data: str = "ACDC"
    dataset: ACDC_dataset = field(default_factory=ACDC_dataset)
    net: str = "teds"
    network: TEDS_Arch = field(default_factory=TEDS_Arch)
