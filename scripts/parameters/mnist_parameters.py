from dataclasses import dataclass, field
from typing import List

from dataclasses_json import dataclass_json
from enforce_typing import enforce_types


@enforce_types
@dataclass_json
@dataclass
class MNIST_dataset:
    """MNIST 数据集默认参数。"""

    ndims: int = 2
    inshape: List[int] = field(default_factory=lambda: [28, 28])
    line_thick: int = 3


@enforce_types
@dataclass_json
@dataclass
class TEDS_Arch:
    """TEDS-Net 结构默认参数。"""

    act: int = 1
    diffeo_int: int = 8
    guas_smooth: int = 1
    Guas_kernel: int = 3
    sigma: float = 2.0
    mega_P: int = 2
    dec_depth: List[int] = field(default_factory=lambda: [1])
    integrator: str = "scaling_squaring"
    r2net_blocks: int = 7


@enforce_types
@dataclass_json
@dataclass
class GeneralNet:
    """通用网络结构默认参数。"""

    dropout: int = 1
    fi: int = 12
    net_depth: int = 2
    in_chan: int = 1
    out_chan: int = 1


@enforce_types
@dataclass_json
@dataclass
class LossParams:
    loss: List[str] = field(default_factory=lambda: ["dice", "grad"])
    weight: List[float] = field(default_factory=lambda: [1, 150])


@enforce_types
@dataclass_json
@dataclass
class Parameters:
    """MNIST 训练参数。"""

    epoch: int = 20
    lr: float = 0.0001
    batch: int = 200
    eval_batch: int = 1
    num_workers: int = 0
    threshold: float = 0.3
    plot_predictions: bool = True
    max_train_batches: int = 0
    max_validation_batches: int = 0
    max_test_batches: int = 0
    data_path: str = "tmp"
    run_name: str = "mnist_smoke"
    output_root: str = "results/mnist"
    evaluate_only: bool = False

    loss_params: LossParams = field(default_factory=LossParams)
    network_params: GeneralNet = field(default_factory=GeneralNet)
    net: str = "teds"
    network: TEDS_Arch = field(default_factory=TEDS_Arch)
    data: str = "mnist"
    dataset: MNIST_dataset = field(default_factory=MNIST_dataset)
