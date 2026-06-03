from dataclasses import dataclass, field
from typing import List

from dataclasses_json import dataclass_json
from enforce_typing import enforce_types


SUPPORTED_INTEGRATORS = ("original_teds", "r2net")


def normalize_integrator_name(name):
    normalized = str(name).strip().lower()
    if normalized not in SUPPORTED_INTEGRATORS:
        supported = ", ".join(SUPPORTED_INTEGRATORS)
        raise ValueError(f"Unsupported integrator '{name}'. Expected one of: {supported}")
    return normalized


@enforce_types
@dataclass_json
@dataclass
class ACDC_dataset:
    """
    Default Arguments for ACDC dataset
    """

    ndims: int = 2
    inshape: List = field(default_factory=lambda: [144, 208])
    ps_meas: List = field(default_factory=lambda: [35, 7])
    datapath: str = "<< PATH TO ACDC DATASET >>"
    betti: List = field(default_factory=lambda: [1, 1, 0, 0])


@enforce_types
@dataclass_json
@dataclass
class TEDS_Arch:
    """
    Default Parameters for TEDS-Net architecture
    """

    act: int = 1
    diffeo_int: int = 8
    guas_smooth: int = 1
    Guas_kernel: int = 5
    sigma: float = 2.0
    mega_P: int = 2
    integrator: str = "r2net"
    r2net_blocks: int = 7
    dec_depth: List = field(default_factory=lambda: [4, 2])

    def __post_init__(self):
        self.integrator = normalize_integrator_name(self.integrator)


@enforce_types
@dataclass_json
@dataclass
class GeneralNet:
    """
    Default Parameters for General Network Architecture
    """

    dropout: int = 1
    fi: int = 12
    net_depth: int = 4
    in_chan: int = 1
    out_chan: int = 1


@enforce_types
@dataclass(frozen=True)
class LossParams:
    loss: List = field(default_factory=lambda: ["dice", "grad", "grad"])
    weight: List = field(default_factory=lambda: [1, 10000, 10000])


@enforce_types
@dataclass_json
@dataclass
class Parameters:
    epoch: int = 200
    lr: float = 0.0001
    lr_sch: bool = False
    batch: int = 5
    checkpoint_freq: int = 50
    threshold: float = 0.3
    seed: int = 42

    loss_params: LossParams = LossParams()
    network_params: GeneralNet = GeneralNet()
    data: str = "ACDC"
    dataset: ACDC_dataset = ACDC_dataset()
    net: str = "teds"
    network: TEDS_Arch = TEDS_Arch()
