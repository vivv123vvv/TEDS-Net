from dataclasses import dataclass, field
from typing import List

from dataclasses_json import dataclass_json
from enforce_typing import enforce_types


@enforce_types
@dataclass_json
@dataclass
class ACDCAugmentation:
    """ACDC 训练阶段的数据增强配置。"""

    enabled: bool = True
    brightness: float = 0.15
    contrast: float = 0.15
    gamma: float = 0.15
    noise_std: float = 0.03
    translate_px: int = 10
    rotate_deg: float = 10.0
    scale_min: float = 0.9
    scale_max: float = 1.1


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
    sdf_temperature: float = 1.5
    pose_translation_limit: float = 0.35
    pose_scale_limit: float = 0.35
    topology_margin: float = 3.0
    topology_projection: bool = True
    projection_angles: int = 128
    projection_threshold: float = 0.5
    augmentation: ACDCAugmentation = field(default_factory=ACDCAugmentation)


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
    dec_depth: List[int] = field(default_factory=lambda: [2])
    integrator: str = "lc_resnet_constrained"
    r2net_blocks: int = 7
    step_alpha: float = 0.08
    pose_enabled: bool = True


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
    """ACDC 训练损失配置。"""

    loss: List[str] = field(default_factory=lambda: ["dice", "grad"])
    weight: List[float] = field(default_factory=lambda: [1.0, 150.0])
    dice_weight: float = 1.0
    boundary_weight: float = 0.5
    smooth_weight: float = 0.1
    jacobian_weight_max: float = 10.0
    ring_weight: float = 2.0
    pose_weight: float = 0.5
    stage_a_end: int = 40
    stage_b_end: int = 160
    jacobian_warmup_epochs: int = 60
    jacobian_epsilon: float = 0.01
    boundary_sdf_scale: float = 5.0


@enforce_types
@dataclass_json
@dataclass
class TrainingParams:
    """训练调度与选模策略。"""

    optimizer: str = "adamw"
    weight_decay: float = 0.0001
    warmup_epochs: int = 10
    scheduler: str = "cosine"
    threshold_candidates: List[float] = field(
        default_factory=lambda: [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    )
    threshold_sweep_start: int = 161
    monitor_priority: List[str] = field(
        default_factory=lambda: ["topology_keep_rate", "hd95_mean", "dice_mean"]
    )


@enforce_types
@dataclass_json
@dataclass
class Parameters:
    """ACDC 训练与评估参数。"""

    epoch: int = 240
    lr: float = 0.0003
    lr_sch: bool = True
    batch: int = 64
    eval_batch: int = 1
    num_workers: int = 0
    checkpoint_freq: int = 20
    threshold: float = 0.3
    plot_predictions: bool = True
    max_train_batches: int = 0
    max_validation_batches: int = 0
    max_test_batches: int = 0
    seed: int = 42
    data_path: str = "results/preprocessed/acdc_ring_144x208"
    run_name: str = "acdc_lcresnet_topology"
    output_root: str = "results/acdc"
    evaluate_only: bool = False

    loss_params: LossParams = field(default_factory=LossParams)
    training: TrainingParams = field(default_factory=TrainingParams)
    network_params: GeneralNet = field(default_factory=GeneralNet)
    data: str = "ACDC"
    dataset: ACDC_dataset = field(default_factory=ACDC_dataset)
    net: str = "teds"
    network: TEDS_Arch = field(default_factory=TEDS_Arch)
