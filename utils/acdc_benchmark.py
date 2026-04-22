import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    from scipy.ndimage import distance_transform_edt, label as connected_components
except ImportError:
    distance_transform_edt = None
    connected_components = None


DEFAULT_DATA_DIR = Path("Resources") / "database" / "processed_2d"
DEFAULT_SPLIT_MANIFEST = Path("parameters") / "acdc_split.json"
DEFAULT_REPORTS_DIR = Path("reports") / "benchmarks"
DEFAULT_CHECKPOINT_ROOT = Path("checkpoints")
DEFAULT_BEST_CHECKPOINT_NAME = "best_teds_net.pth"
_CASE_PATTERN = re.compile(r"^(patient\d+_frame\d+)")


def normalize_split_name(split):
    normalized = str(split).strip().lower()
    if normalized in {"validation", "valid"}:
        return "val"
    return normalized


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        requested = str(device)
    else:
        requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def sync_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_gpu_memory_mb(device):
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def sample_id_from_path(file_path):
    return Path(file_path).stem


def case_id_from_sample_id(sample_id):
    match = _CASE_PATTERN.match(sample_id)
    return match.group(1) if match else sample_id


def list_npz_sample_names(data_dir):
    data_dir = Path(data_dir)
    return sorted(path.name for path in data_dir.glob("*.npz"))


def load_split_manifest(manifest_path, data_dir):
    manifest_path = Path(manifest_path)
    data_dir = Path(data_dir)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    splits = payload.get("splits", payload)
    expected_files = []
    split_map = {}
    for split_name, file_names in splits.items():
        normalized_name = normalize_split_name(split_name)
        split_map[normalized_name] = list(file_names)
        expected_files.extend(file_names)

    actual_files = list_npz_sample_names(data_dir)
    actual_set = set(actual_files)
    expected_set = set(expected_files)
    duplicates = len(expected_files) != len(expected_set)

    if duplicates:
        raise ValueError(f"Duplicate sample names found in split manifest: {manifest_path}")
    if actual_set != expected_set:
        missing = sorted(actual_set - expected_set)
        extras = sorted(expected_set - actual_set)
        raise ValueError(
            "Split manifest does not match dataset files. "
            f"Missing in manifest: {missing[:5]} Extras in manifest: {extras[:5]}"
        )

    payload["splits"] = split_map
    payload["counts"] = {split_name: len(file_names) for split_name, file_names in split_map.items()}
    return payload


def get_split_filenames(manifest, split):
    split_name = normalize_split_name(split)
    try:
        return manifest["splits"][split_name]
    except KeyError as exc:
        available = ", ".join(sorted(manifest["splits"].keys()))
        raise KeyError(f"Unknown split '{split_name}'. Available splits: {available}") from exc


def make_run_dir(output_dir, run_name):
    return ensure_dir(Path(output_dir) / run_name)


def write_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path, fieldnames, rows):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def align_mask_tensors(pred_mask, gt_mask):
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.unsqueeze(1)
    if gt_mask.ndim == 3:
        gt_mask = gt_mask.unsqueeze(1)
    if pred_mask.shape[-2:] != gt_mask.shape[-2:]:
        gt_mask = torch.nn.functional.interpolate(
            gt_mask,
            size=pred_mask.shape[-2:],
            mode="nearest",
        )
    return pred_mask, gt_mask


def dice_score(pred, target):
    smooth = 1e-5
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return float((2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))


def compute_jacobian_determinant_2d(flow):
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"Expected flow to have shape [B, 2, H, W], got {tuple(flow.shape)}")

    flow_np = flow.detach().cpu().numpy()
    jacobians = []
    for batch_idx in range(flow_np.shape[0]):
        u_component = flow_np[batch_idx, 0]
        v_component = flow_np[batch_idx, 1]
        dy_u, dx_u = np.gradient(u_component)
        dy_v, dx_v = np.gradient(v_component)
        det_j = (1.0 + dx_u) * (1.0 + dy_v) - (dy_u * dx_v)
        jacobians.append(det_j)
    return np.asarray(jacobians)


def jacobian_negative_ratio(flow):
    jacobian_det = compute_jacobian_determinant_2d(flow)
    negative = np.sum(jacobian_det < 0)
    total = jacobian_det.size
    return float(negative / total)


def _mask_to_bool_numpy(mask):
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    mask_np = np.squeeze(np.asarray(mask))
    if mask_np.ndim != 2:
        raise ValueError(f"Expected a 2D mask after squeeze, got shape {mask_np.shape}")
    return mask_np.astype(bool)


def _require_scipy_metric(metric_name):
    if distance_transform_edt is None or connected_components is None:
        raise ImportError(f"scipy is required to compute {metric_name}.")


def hausdorff_distance(pred, target):
    _require_scipy_metric("Hausdorff distance")
    pred_np = _mask_to_bool_numpy(pred)
    target_np = _mask_to_bool_numpy(target)
    if pred_np.shape != target_np.shape:
        raise ValueError(f"Mask shapes do not match: {pred_np.shape} vs {target_np.shape}")

    pred_has_fg = bool(pred_np.any())
    target_has_fg = bool(target_np.any())
    if not pred_has_fg and not target_has_fg:
        return 0.0
    if pred_has_fg != target_has_fg:
        height, width = pred_np.shape
        return float(np.hypot(height, width))

    target_distance = distance_transform_edt(~target_np)
    pred_distance = distance_transform_edt(~pred_np)
    pred_to_target = float(target_distance[pred_np].max())
    target_to_pred = float(pred_distance[target_np].max())
    return max(pred_to_target, target_to_pred)


def topology_signature(mask):
    _require_scipy_metric("topology metrics")
    mask_np = _mask_to_bool_numpy(mask)
    structure = np.ones((3, 3), dtype=np.int8)
    _, foreground_components = connected_components(mask_np, structure=structure)
    background_labels, background_components = connected_components(~mask_np, structure=structure)

    holes = 0
    for component_idx in range(1, background_components + 1):
        component = background_labels == component_idx
        touches_border = (
            component[0, :].any()
            or component[-1, :].any()
            or component[:, 0].any()
            or component[:, -1].any()
        )
        if not touches_border:
            holes += 1
    return int(foreground_components), int(holes)


def correct_topology_score(pred, target):
    return 1.0 if topology_signature(pred) == topology_signature(target) else 0.0


def model_parameter_count(model, trainable_only=False):
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if not trainable_only or parameter.requires_grad
        )
    )


def parameter_count_x1e5(parameter_count):
    return float(parameter_count / 100000.0)


def model_parameter_count_x1e5(model, trainable_only=False):
    return parameter_count_x1e5(model_parameter_count(model, trainable_only=trainable_only))


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def aggregate_metric_rows(rows, group_key):
    metric_keys = ["forward_ms", "dice", "hd", "correct_topology", "jacobian_neg_ratio"]
    grouped = defaultdict(lambda: {"sample_count": 0, **{key: [] for key in metric_keys}})
    for row in rows:
        group_id = row[group_key]
        grouped[group_id]["sample_count"] += 1
        for metric_key in metric_keys:
            if metric_key in row and row[metric_key] not in (None, ""):
                grouped[group_id][metric_key].append(float(row[metric_key]))

    aggregated = []
    for group_id in sorted(grouped.keys()):
        metrics = grouped[group_id]
        row = {group_key: group_id, "sample_count": metrics["sample_count"]}
        for metric_key in metric_keys:
            values = metrics[metric_key]
            if not values:
                continue
            if metric_key == "correct_topology":
                row["correct_topology_rate"] = float(np.mean(values))
            else:
                row[f"mean_{metric_key}"] = float(np.mean(values))
        aggregated.append(row)
    return aggregated


def build_eval_summary(
    sample_rows,
    peak_mem_mb,
    benchmark_case,
    split,
    checkpoint_path,
    parameter_count=None,
):
    forward_values = [float(row["forward_ms"]) for row in sample_rows]
    dice_values = [float(row["dice"]) for row in sample_rows]
    hd_values = [float(row["hd"]) for row in sample_rows if "hd" in row]
    topology_values = [
        float(row["correct_topology"]) for row in sample_rows if "correct_topology" in row
    ]
    jacobian_values = [float(row["jacobian_neg_ratio"]) for row in sample_rows]
    return {
        "benchmark_case": benchmark_case,
        "split": normalize_split_name(split),
        "checkpoint_path": str(checkpoint_path),
        "sample_count": len(sample_rows),
        "mean_forward_ms": float(np.mean(forward_values)) if forward_values else None,
        "p50_forward_ms": percentile(forward_values, 50),
        "p95_forward_ms": percentile(forward_values, 95),
        "mean_dice": float(np.mean(dice_values)) if dice_values else None,
        "mean_hd": float(np.mean(hd_values)) if hd_values else None,
        "hd_unit": "pixel",
        "correct_topology_rate": float(np.mean(topology_values)) if topology_values else None,
        "mean_jacobian_neg_ratio": float(np.mean(jacobian_values)) if jacobian_values else None,
        "peak_gpu_mem_mb": None if peak_mem_mb is None else float(peak_mem_mb),
        "parameter_count": None if parameter_count is None else int(parameter_count),
        "parameter_count_x1e5": None
        if parameter_count is None
        else parameter_count_x1e5(parameter_count),
    }


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_run_dirs(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    return sorted(path for path in output_dir.iterdir() if path.is_dir())


def comparison_rows_from_run_dirs(run_dirs):
    rows = []
    for run_dir in [Path(run_dir) for run_dir in run_dirs]:
        train_summary_path = run_dir / "train_summary.json"
        eval_summary_path = run_dir / "eval_summary.json"
        if not train_summary_path.exists() and not eval_summary_path.exists():
            continue

        train_summary = load_json(train_summary_path) if train_summary_path.exists() else {}
        eval_summary = load_json(eval_summary_path) if eval_summary_path.exists() else {}
        mean_epoch_sec = train_summary.get("mean_epoch_sec")
        mean_epoch_min = train_summary.get("mean_epoch_min")
        if mean_epoch_min is None and mean_epoch_sec is not None:
            mean_epoch_min = float(mean_epoch_sec) / 60.0
        parameter_count = train_summary.get("parameter_count", eval_summary.get("parameter_count"))
        parameter_count_scaled = train_summary.get(
            "parameter_count_x1e5",
            eval_summary.get("parameter_count_x1e5"),
        )
        if parameter_count_scaled is None and parameter_count is not None:
            parameter_count_scaled = parameter_count_x1e5(parameter_count)
        rows.append(
            {
                "case": eval_summary.get("benchmark_case", train_summary.get("run_name", run_dir.name)),
                "dice": eval_summary.get("mean_dice"),
                "hd": eval_summary.get("mean_hd"),
                "correct_topology": eval_summary.get("correct_topology_rate"),
                "time_per_epoch_min": mean_epoch_min,
                "parameter_count_x1e5": parameter_count_scaled,
                "mean_forward_ms": eval_summary.get("mean_forward_ms"),
                "mean_epoch_sec": mean_epoch_sec,
                "peak_gpu_mem_mb_train": train_summary.get("max_peak_gpu_mem_mb"),
                "peak_gpu_mem_mb_eval": eval_summary.get("peak_gpu_mem_mb"),
                "jacobian_neg_ratio": eval_summary.get("mean_jacobian_neg_ratio"),
            }
        )
    return rows


def _format_markdown_value(value):
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.6f}" if abs(value) < 1000 else f"{value:.2f}"
    return str(value)


def write_comparison_artifacts(run_dirs, output_dir):
    output_dir = ensure_dir(output_dir)
    fieldnames = [
        "case",
        "dice",
        "hd",
        "correct_topology",
        "time_per_epoch_min",
        "parameter_count_x1e5",
        "mean_forward_ms",
        "mean_epoch_sec",
        "peak_gpu_mem_mb_train",
        "peak_gpu_mem_mb_eval",
        "jacobian_neg_ratio",
    ]
    rows = comparison_rows_from_run_dirs(run_dirs)
    csv_path = output_dir / "comparison.csv"
    md_path = output_dir / "comparison.md"
    write_csv(csv_path, fieldnames, rows)

    lines = [
        "| case | Dice (up) | HD (down) | Correct topology (up) | Time per epoch [min] (down) | # Parameters [x10^5] | mean_forward_ms | peak_gpu_mem_mb_train | peak_gpu_mem_mb_eval | jacobian_neg_ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {dice} | {hd} | {correct_topology} | {time_per_epoch_min} | {parameter_count_x1e5} | {mean_forward_ms} | {peak_gpu_mem_mb_train} | {peak_gpu_mem_mb_eval} | {jacobian_neg_ratio} |".format(
                case=_format_markdown_value(row["case"]),
                dice=_format_markdown_value(row["dice"]),
                hd=_format_markdown_value(row["hd"]),
                correct_topology=_format_markdown_value(row["correct_topology"]),
                time_per_epoch_min=_format_markdown_value(row["time_per_epoch_min"]),
                parameter_count_x1e5=_format_markdown_value(row["parameter_count_x1e5"]),
                mean_forward_ms=_format_markdown_value(row["mean_forward_ms"]),
                peak_gpu_mem_mb_train=_format_markdown_value(row["peak_gpu_mem_mb_train"]),
                peak_gpu_mem_mb_eval=_format_markdown_value(row["peak_gpu_mem_mb_eval"]),
                jacobian_neg_ratio=_format_markdown_value(row["jacobian_neg_ratio"]),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": rows, "csv_path": csv_path, "md_path": md_path}
