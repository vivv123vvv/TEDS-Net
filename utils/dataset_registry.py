import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_DATASET_ID = "acdc"
DEFAULT_DATASET_REGISTRY = Path("parameters") / "datasets.json"


def normalize_dataset_id(dataset_id):
    return str(dataset_id).strip().lower().replace("-", "_")


def _deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _format_dataset_value(value, dataset_id):
    if isinstance(value, str):
        return value.format(dataset_id=dataset_id)
    return value


def _optional_path(value):
    if value in (None, ""):
        return None
    return Path(value)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    display_name: str
    data_dir: Path
    split_manifest: Path
    reports_dir: Path
    checkpoint_root: Path
    experiment_log_dir: Path
    loader: str
    image_key: str
    prior_key: str
    label_key: str
    target_rule: str
    label_value: Optional[int]
    case_id_pattern: Optional[str]
    model: Dict[str, Any]
    isolation: Dict[str, Any]
    raw: Dict[str, Any]

    def loader_kwargs(self):
        return {
            "dataset_id": self.dataset_id,
            "image_key": self.image_key,
            "prior_key": self.prior_key,
            "label_key": self.label_key,
            "target_rule": self.target_rule,
            "label_value": self.label_value,
            "case_id_pattern": self.case_id_pattern,
        }

    def to_dict(self):
        return {
            "dataset_id": self.dataset_id,
            "display_name": self.display_name,
            "data_dir": str(self.data_dir),
            "split_manifest": str(self.split_manifest),
            "reports_dir": str(self.reports_dir),
            "checkpoint_root": str(self.checkpoint_root),
            "experiment_log_dir": str(self.experiment_log_dir),
            "loader": self.loader,
            "image_key": self.image_key,
            "prior_key": self.prior_key,
            "label_key": self.label_key,
            "target_rule": self.target_rule,
            "label_value": self.label_value,
            "case_id_pattern": self.case_id_pattern,
            "model": self.model,
            "isolation": self.isolation,
        }


def load_dataset_registry(registry_path=DEFAULT_DATASET_REGISTRY):
    registry_path = Path(registry_path)
    with registry_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_dataset_spec(
    dataset_id=DEFAULT_DATASET_ID,
    registry_path=DEFAULT_DATASET_REGISTRY,
):
    normalized_id = normalize_dataset_id(dataset_id)
    registry = load_dataset_registry(registry_path)
    datasets = registry.get("datasets", {})
    if normalized_id not in datasets:
        available = ", ".join(sorted(datasets.keys()))
        raise KeyError(f"Unknown dataset '{dataset_id}'. Available datasets: {available}")

    config = copy.deepcopy(registry.get("defaults", {}))
    _deep_update(config, datasets[normalized_id])

    def cfg(key, default=None):
        value = config.get(key, default)
        return _format_dataset_value(value, normalized_id)

    data_dir = _optional_path(cfg("data_dir"))
    split_manifest = _optional_path(cfg("split_manifest"))
    if data_dir is None:
        raise ValueError(f"Dataset '{normalized_id}' must define data_dir")
    if split_manifest is None:
        raise ValueError(f"Dataset '{normalized_id}' must define split_manifest")

    return DatasetSpec(
        dataset_id=normalized_id,
        display_name=cfg("display_name", normalized_id.upper()),
        data_dir=data_dir,
        split_manifest=split_manifest,
        reports_dir=Path(cfg("reports_dir", f"reports/benchmarks/{normalized_id}")),
        checkpoint_root=Path(cfg("checkpoint_root", f"checkpoints/{normalized_id}")),
        experiment_log_dir=Path(cfg("experiment_log_dir", f"logs/experiments/{normalized_id}")),
        loader=cfg("loader", "teds_npz"),
        image_key=cfg("image_key", "image"),
        prior_key=cfg("prior_key", "prior"),
        label_key=cfg("label_key", "label"),
        target_rule=cfg("target_rule", "label_equals_value"),
        label_value=config.get("label_value"),
        case_id_pattern=cfg("case_id_pattern"),
        model=copy.deepcopy(config.get("model", {})),
        isolation=copy.deepcopy(config.get("isolation", {})),
        raw=config,
    )


def apply_dataset_spec_to_params(params, dataset_spec):
    params.data = dataset_spec.dataset_id
    if hasattr(params.dataset, "datapath"):
        params.dataset.datapath = str(dataset_spec.data_dir)

    dataset_payload = dataset_spec.model.get("dataset", {})
    for key, value in dataset_payload.items():
        setattr(params.dataset, key, copy.deepcopy(value))

    network_payload = dataset_spec.model.get("network", {})
    for key, value in network_payload.items():
        setattr(params.network, key, copy.deepcopy(value))

    network_params_payload = dataset_spec.model.get("network_params", {})
    for key, value in network_params_payload.items():
        setattr(params.network_params, key, copy.deepcopy(value))

    return params
