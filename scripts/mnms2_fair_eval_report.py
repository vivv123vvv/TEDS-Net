import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.ndimage import binary_closing, distance_transform_edt, label as connected_components
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_results import load_model, make_eval_loader, set_random_seed
from utils.acdc_benchmark import (
    assd_distance,
    dice_score,
    ensure_dir,
    hausdorff_distance,
    hd95_distance,
    iou_score,
    jacobian_negative_ratio,
    precision_score,
    recall_score,
    resolve_device,
    symmetric_surface_distances,
    sync_cuda,
    topology_signature,
)
from utils.dataset_registry import DEFAULT_DATASET_REGISTRY, resolve_dataset_spec


DEFAULT_DATA_DIR = Path("Resources") / "database" / "mnms2_stratified_seed42_20260615_preprocess_v2_2d"
DEFAULT_SPLIT = Path("parameters") / "mnms2_stratified_seed42_20260615_preprocess_v2_split.json"
DEFAULT_R2NET = (
    Path("checkpoints")
    / "mnms2"
    / "mnms2-r2net-stratified-preprocess-v2-s42-20260615-e20-warm"
    / "best_teds_net.pth"
)
DEFAULT_BASELINE = (
    Path("checkpoints")
    / "mnms2"
    / "mnms2-original-teds-stratified-preprocess-v2-s42-20260615-e20-warm"
    / "best_teds_net.pth"
)
DEFAULT_REPORT_DIR = (
    Path("reports")
    / "experiment_reports"
    / "mnms2"
    / "mnms2_stratified_preprocess_v2_seed42_20260615"
)

METRICS = [
    "dice",
    "iou",
    "hd",
    "hd95",
    "assd",
    "precision",
    "recall",
    "correct_topology",
]

POSTPROCESS_CONFIG = {
    "name": "closing_r1_lcc_fill_extra_holes_preserve_largest_hole",
    "closing_radius": 1,
    "keep_largest_component": True,
    "fill_extra_holes_preserve_largest": True,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Fair M&Ms-2 stratified split evaluation and Chinese report.")
    parser.add_argument("--dataset", default="mnms2")
    parser.add_argument("--dataset-registry", default=str(DEFAULT_DATASET_REGISTRY))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT))
    parser.add_argument("--r2net-checkpoint", default=str(DEFAULT_R2NET))
    parser.add_argument("--baseline-checkpoint", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--threshold-min", type=float, default=0.20)
    parser.add_argument("--threshold-max", type=float, default=0.80)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--hd-epsilon", type=float, default=0.01)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def threshold_values(start, stop, step):
    values = []
    current = float(start)
    while current <= float(stop) + 1e-9:
        values.append(round(current, 4))
        current += float(step)
    return values


def sample_metadata_map(split_manifest):
    manifest = read_json(split_manifest)
    records = manifest.get("sample_records", [])
    return {Path(record["file_name"]).stem: record for record in records}


def output_from_model(outputs):
    if len(outputs) == 2:
        return outputs[0], outputs[1]
    if len(outputs) == 3:
        return outputs[0], outputs[2]
    raise ValueError(f"Unexpected model output length: {len(outputs)}")


def collect_predictions(
    checkpoint,
    model_name,
    split,
    dataset_spec,
    data_dir,
    split_manifest,
    metadata_by_sample,
    device,
    seed,
    max_samples=None,
):
    model, _, params = load_model(
        checkpoint,
        device,
        seed=seed,
        dataset_spec=dataset_spec,
    )
    set_random_seed(params.seed)
    _, loader = make_eval_loader(data_dir, split_manifest, split, dataset_spec)
    predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"{model_name} {split} predictions")):
            if max_samples is not None and batch_idx >= max_samples:
                break
            images, priors, gt_masks, sample_ids, case_ids = batch
            images = images.float().to(device)
            priors = priors.float().to(device)
            gt_masks = gt_masks.float().cpu()

            sync_cuda(device)
            start = time.perf_counter()
            pred_prob, flow_upsamp = output_from_model(model(images, priors))
            sync_cuda(device)
            forward_ms = (time.perf_counter() - start) * 1000.0

            pred_prob = pred_prob.detach().cpu()
            if pred_prob.ndim == 3:
                pred_prob = pred_prob.unsqueeze(1)
            if gt_masks.ndim == 3:
                gt_masks = gt_masks.unsqueeze(1)
            if pred_prob.shape[-2:] != gt_masks.shape[-2:]:
                gt_masks = torch.nn.functional.interpolate(
                    gt_masks,
                    size=pred_prob.shape[-2:],
                    mode="nearest",
                )

            sample_id = str(sample_ids[0])
            metadata = dict(metadata_by_sample.get(sample_id, {}))
            predictions.append(
                {
                    "sample_id": sample_id,
                    "case_id": str(case_ids[0]),
                    "patient_id": metadata.get("patient_id", sample_id.split("_", 1)[0]),
                    "image": np.squeeze(images.detach().cpu().numpy()).astype(np.float32),
                    "pred_prob": np.squeeze(pred_prob.numpy()).astype(np.float32),
                    "target": np.squeeze(gt_masks.numpy()).astype(bool),
                    "forward_ms": float(forward_ms),
                    "jacobian_neg_ratio": jacobian_negative_ratio(flow_upsamp.detach().cpu()),
                    "metadata": metadata,
                }
            )

    return predictions, {
        "model": model_name,
        "checkpoint": str(checkpoint),
        "integrator": params.network.integrator,
        "r2net_blocks": int(params.network.r2net_blocks),
        "seed": int(params.seed),
    }


def disk_structure(radius):
    radius = int(radius)
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    coords = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return (xx * xx + yy * yy) <= (radius * radius)


def keep_largest_component(mask):
    labels, count = connected_components(mask, structure=np.ones((3, 3), dtype=np.int8))
    if count <= 1:
        return np.asarray(mask, dtype=bool)
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    return labels == int(np.argmax(areas))


def interior_hole_labels(mask):
    labels, count = connected_components(~mask, structure=np.ones((3, 3), dtype=np.int8))
    holes = []
    for idx in range(1, count + 1):
        component = labels == idx
        touches_border = (
            component[0, :].any()
            or component[-1, :].any()
            or component[:, 0].any()
            or component[:, -1].any()
        )
        if not touches_border:
            holes.append((idx, int(component.sum())))
    return labels, holes


def fill_extra_holes_preserve_largest(mask):
    mask = np.asarray(mask, dtype=bool).copy()
    labels, holes = interior_hole_labels(mask)
    if len(holes) <= 1:
        return mask
    keep_label = max(holes, key=lambda item: item[1])[0]
    for label_idx, _ in holes:
        if label_idx != keep_label:
            mask[labels == label_idx] = True
    return mask


def apply_postprocess(mask, config=None):
    if not config or config.get("name") == "none":
        return np.asarray(mask, dtype=bool).copy()
    processed = np.asarray(mask, dtype=bool).copy()
    if config.get("closing_radius", 0) > 0:
        processed = binary_closing(processed, structure=disk_structure(config["closing_radius"]))
    if config.get("keep_largest_component"):
        processed = keep_largest_component(processed)
    if config.get("fill_extra_holes_preserve_largest"):
        processed = fill_extra_holes_preserve_largest(processed)
    return processed.astype(bool)


def false_positive_diagnostics(pred_mask, target_mask, remote_distance=20.0):
    pred_mask = np.asarray(pred_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    false_positive = pred_mask & ~target_mask
    if not false_positive.any():
        return 0, 0.0
    distances = distance_transform_edt(~target_mask)
    remote_pixels = false_positive & (distances > float(remote_distance))
    return int(remote_pixels.any()), float(distances[false_positive].max())


def evaluate_predictions(predictions, threshold, benchmark_case, postprocess_config=None):
    rows = []
    postprocess_name = "none" if not postprocess_config else postprocess_config["name"]
    for item in predictions:
        metadata = item["metadata"]
        pred_mask = item["pred_prob"] > float(threshold)
        pred_mask = apply_postprocess(pred_mask, postprocess_config)
        target_mask = item["target"].astype(bool)
        pred_components, pred_holes = topology_signature(pred_mask)
        target_components, target_holes = topology_signature(target_mask)
        surface_distances = symmetric_surface_distances(pred_mask, target_mask)
        remote_fp, max_fp_distance = false_positive_diagnostics(pred_mask, target_mask)
        slice_id = int(metadata.get("slice_id", -1))
        source_slices = int(metadata.get("source_slices", -1))
        basal_apical = int(source_slices > 0 and (slice_id <= 1 or slice_id >= source_slices - 2))

        row = {
            "benchmark_case": benchmark_case,
            "sample_id": item["sample_id"],
            "case_id": item["case_id"],
            "patient_id": item["patient_id"],
            "phase": metadata.get("phase", ""),
            "slice_id": slice_id,
            "pathology": metadata.get("pathology", ""),
            "vendor": metadata.get("vendor", ""),
            "field_strength": metadata.get("field_strength", ""),
            "threshold": float(threshold),
            "postprocess": postprocess_name,
            "forward_ms": item["forward_ms"],
            "dice": dice_score(pred_mask, target_mask),
            "iou": iou_score(pred_mask, target_mask),
            "hd": hausdorff_distance(pred_mask, target_mask),
            "hd95": hd95_distance(pred_mask, target_mask),
            "assd": assd_distance(pred_mask, target_mask),
            "precision": precision_score(pred_mask, target_mask),
            "recall": recall_score(pred_mask, target_mask),
            "correct_topology": float((pred_components, pred_holes) == (target_components, target_holes)),
            "pred_components": pred_components,
            "pred_holes": pred_holes,
            "target_components": target_components,
            "target_holes": target_holes,
            "jacobian_neg_ratio": item["jacobian_neg_ratio"],
            "extra_components": int(pred_components > target_components),
            "missing_holes": int(pred_holes < target_holes),
            "extra_holes": int(pred_holes > target_holes),
            "remote_false_positive": remote_fp,
            "max_false_positive_distance": max_fp_distance,
            "basal_or_apical": basal_apical,
            "surface_distance_p95": float(np.percentile(surface_distances, 95)),
        }
        rows.append(row)
    return rows


def mean_or_none(values):
    values = [float(value) for value in values if value not in (None, "")]
    if not values:
        return None
    return float(np.mean(values))


def summarize_rows(rows, group_key=None):
    groups = defaultdict(list)
    if group_key is None:
        groups["overall"] = rows
    else:
        for row in rows:
            groups[row.get(group_key, "") or "UNKNOWN"].append(row)
    summary_rows = []
    for group, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        out = {
            "group": group,
            "sample_count": len(group_rows),
            "topology_failure_count": int(sum(float(row["correct_topology"]) < 1.0 for row in group_rows)),
            "extra_components_count": int(sum(int(row["extra_components"]) for row in group_rows)),
            "missing_holes_count": int(sum(int(row["missing_holes"]) for row in group_rows)),
            "extra_holes_count": int(sum(int(row["extra_holes"]) for row in group_rows)),
            "remote_false_positive_count": int(sum(int(row["remote_false_positive"]) for row in group_rows)),
        }
        for metric in METRICS:
            values = [row[metric] for row in group_rows]
            key = "topology_success_rate" if metric == "correct_topology" else f"mean_{metric}"
            out[key] = mean_or_none(values)
        summary_rows.append(out)
    return summary_rows


def select_best_threshold(sweep_rows, hd_epsilon):
    min_hd = min(float(row["mean_hd"]) for row in sweep_rows)
    candidates = [row for row in sweep_rows if float(row["mean_hd"]) <= min_hd + float(hd_epsilon)]
    return sorted(
        candidates,
        key=lambda row: (
            -float(row["topology_success_rate"]),
            -float(row["mean_dice"]),
            abs(float(row["threshold"]) - 0.5),
        ),
    )[0]


def threshold_sweep(predictions, thresholds, model_name, output_dir, hd_epsilon):
    rows = []
    for threshold in tqdm(thresholds, desc=f"{model_name} threshold sweep"):
        eval_rows = evaluate_predictions(
            predictions,
            threshold=threshold,
            benchmark_case=f"{model_name}@val_thr{threshold:.2f}",
            postprocess_config=None,
        )
        summary = summarize_rows(eval_rows)[0]
        rows.append(
            {
                "model": model_name,
                "threshold": float(threshold),
                "sample_count": summary["sample_count"],
                "mean_dice": summary["mean_dice"],
                "mean_iou": summary["mean_iou"],
                "mean_hd": summary["mean_hd"],
                "mean_hd95": summary["mean_hd95"],
                "mean_assd": summary["mean_assd"],
                "mean_precision": summary["mean_precision"],
                "mean_recall": summary["mean_recall"],
                "topology_success_rate": summary["topology_success_rate"],
                "topology_failure_count": summary["topology_failure_count"],
            }
        )
    best = select_best_threshold(rows, hd_epsilon=hd_epsilon)
    write_csv(
        output_dir / f"{model_name}_val_threshold_sweep.csv",
        list(rows[0].keys()),
        rows,
    )
    write_json(output_dir / f"{model_name}_val_best_threshold.json", best)
    return rows, best


def overlay_rgb(image, pred_mask, target_mask):
    image = np.asarray(image, dtype=np.float32)
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    base = np.stack([image, image, image], axis=-1)
    pred_mask = np.asarray(pred_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    tp = pred_mask & target_mask
    fp = pred_mask & ~target_mask
    fn = ~pred_mask & target_mask
    out = base.copy()
    out[tp] = 0.55 * out[tp] + 0.45 * np.array([0.0, 0.85, 0.0])
    out[fp] = 0.45 * out[fp] + 0.55 * np.array([1.0, 0.25, 0.0])
    out[fn] = 0.45 * out[fn] + 0.55 * np.array([0.1, 0.4, 1.0])
    return np.clip(out, 0.0, 1.0)


def save_overview(path, title, predictions, rows, threshold, postprocess_config=None, limit=30):
    rows = rows[:limit]
    pred_by_sample = {item["sample_id"]: item for item in predictions}
    columns = 5
    nrows = max(1, math.ceil(len(rows) / columns))
    fig, axes = plt.subplots(nrows, columns, figsize=(4.0 * columns, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, row in zip(axes, rows):
        item = pred_by_sample[row["sample_id"]]
        pred_mask = apply_postprocess(item["pred_prob"] > float(threshold), postprocess_config)
        target = item["target"]
        ax.imshow(overlay_rgb(item["image"], pred_mask, target))
        ax.contour(target, levels=[0.5], colors=["lime"], linewidths=0.8)
        ax.contour(pred_mask, levels=[0.5], colors=["red"], linewidths=0.8)
        ax.set_title(
            f"{row['sample_id']}\nD={row['dice']:.3f} HD={row['hd']:.1f}",
            fontsize=8,
        )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_side_by_side(path, title, baseline_items, r2net_items, baseline_rows, r2net_rows, r2net_post_rows, thresholds, limit=12):
    base_map = {item["sample_id"]: item for item in baseline_items}
    r2_map = {item["sample_id"]: item for item in r2net_items}
    base_row_map = {row["sample_id"]: row for row in baseline_rows}
    r2_row_map = {row["sample_id"]: row for row in r2net_rows}
    post_row_map = {row["sample_id"]: row for row in r2net_post_rows}
    sample_ids = []
    for row in sorted(r2net_rows, key=lambda r: float(r["hd"]), reverse=True):
        if row["sample_id"] in base_map:
            sample_ids.append(row["sample_id"])
        if len(sample_ids) >= limit:
            break

    fig, axes = plt.subplots(len(sample_ids), 4, figsize=(15, 3.4 * max(1, len(sample_ids))))
    axes = np.atleast_2d(axes)
    for row_idx, sample_id in enumerate(sample_ids):
        base_item = base_map[sample_id]
        r2_item = r2_map[sample_id]
        target = r2_item["target"]
        base_mask = base_item["pred_prob"] > thresholds["baseline"]
        r2_mask = r2_item["pred_prob"] > thresholds["r2net"]
        r2_post = apply_postprocess(r2_mask, POSTPROCESS_CONFIG)
        panels = [
            ("GT", target, None),
            (f"baseline HD={base_row_map[sample_id]['hd']:.1f}", base_mask, base_item),
            (f"r2net HD={r2_row_map[sample_id]['hd']:.1f}", r2_mask, r2_item),
            (f"r2net+post HD={post_row_map[sample_id]['hd']:.1f}", r2_post, r2_item),
        ]
        for col_idx, (panel_title, mask, item) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if item is None:
                ax.imshow(r2_item["image"], cmap="gray")
                ax.contour(target, levels=[0.5], colors=["lime"], linewidths=0.9)
            else:
                ax.imshow(overlay_rgb(item["image"], mask, target))
                ax.contour(target, levels=[0.5], colors=["lime"], linewidths=0.8)
                ax.contour(mask, levels=[0.5], colors=["red"], linewidths=0.8)
            ax.set_title(f"{sample_id}\n{panel_title}", fontsize=8)
            ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def markdown_table(rows, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def comparison_row(label, rows, threshold, checkpoint, postprocess):
    summary = summarize_rows(rows)[0]
    return {
        "case": label,
        "checkpoint": str(checkpoint),
        "threshold": float(threshold),
        "postprocess": postprocess,
        "sample_count": summary["sample_count"],
        "dice": summary["mean_dice"],
        "iou": summary["mean_iou"],
        "hd": summary["mean_hd"],
        "hd95": summary["mean_hd95"],
        "assd": summary["mean_assd"],
        "precision": summary["mean_precision"],
        "recall": summary["mean_recall"],
        "topology_success_rate": summary["topology_success_rate"],
        "topology_failure_count": summary["topology_failure_count"],
        "extra_components_count": summary["extra_components_count"],
        "missing_holes_count": summary["missing_holes_count"],
        "extra_holes_count": summary["extra_holes_count"],
        "remote_false_positive_count": summary["remote_false_positive_count"],
    }


def write_group_outputs(output_dir, case_name, rows):
    fields = [
        "group",
        "sample_count",
        "mean_dice",
        "mean_iou",
        "mean_hd",
        "mean_hd95",
        "mean_assd",
        "mean_precision",
        "mean_recall",
        "topology_success_rate",
        "topology_failure_count",
        "extra_components_count",
        "missing_holes_count",
        "extra_holes_count",
        "remote_false_positive_count",
    ]
    for group_key in ("pathology", "vendor", "field_strength", "phase"):
        group_rows = summarize_rows(rows, group_key=group_key)
        write_csv(output_dir / f"{case_name}_by_{group_key}.csv", fields, group_rows)


def write_failure_outputs(output_dir, case_name, rows, top_k):
    fields = list(rows[0].keys()) if rows else []
    top_hd = sorted(rows, key=lambda row: float(row["hd"]), reverse=True)[:top_k]
    topo_fail = [row for row in rows if float(row["correct_topology"]) < 1.0]
    write_csv(output_dir / f"{case_name}_hd_top{top_k}.csv", fields, top_hd)
    write_csv(output_dir / f"{case_name}_topology_failures.csv", fields, topo_fail)
    write_csv(output_dir / f"{case_name}_extra_components.csv", fields, [r for r in rows if int(r["extra_components"])])
    write_csv(output_dir / f"{case_name}_missing_holes.csv", fields, [r for r in rows if int(r["missing_holes"])])
    write_csv(output_dir / f"{case_name}_extra_holes.csv", fields, [r for r in rows if int(r["extra_holes"])])
    write_csv(output_dir / f"{case_name}_remote_false_positives.csv", fields, [r for r in rows if int(r["remote_false_positive"])])
    return top_hd, topo_fail


def write_report(output_dir, context):
    comparison_rows = context["comparison_rows"]
    best = {row["case"]: row for row in comparison_rows}
    columns = [
        "case",
        "threshold",
        "postprocess",
        "dice",
        "iou",
        "hd",
        "hd95",
        "assd",
        "precision",
        "recall",
        "topology_success_rate",
        "topology_failure_count",
    ]
    split_overview = Path(context["data_dir"]) / "split_stats" / "split_overview.csv"
    split_distributions = Path(context["data_dir"]) / "split_stats" / "split_distributions.csv"
    r2_no = best["r2net_no_post"]
    base_no = best["baseline_no_post"]
    r2_post = best["r2net_post"]
    base_post = best["baseline_post"]
    conclusion = (
        "新模型在 Dice/IoU 上略高于 baseline，但 HD/HD95 和拓扑成功率未超过 baseline。"
        if r2_no["hd"] > base_no["hd"] or r2_post["hd"] > base_post["hd"]
        else "新模型在主要指标上超过 baseline。"
    )
    lines = [
        "# M&Ms-2 stratified preprocess-v2 实验报告",
        "",
        "## 实验目的",
        "",
        "本实验修正原顺序切分导致的 pathology/vendor/field 分布偏移，使用 patient-level stratified split 和更透明的预处理记录，公平比较 original TEDS baseline 与 R2Net-TEDS。",
        "",
        "## 新旧 split 差异",
        "",
        "- 旧 split: readme 顺序切分 160/40/160，之前发现 train 中缺 RV/TRI，而 test 中集中出现 RV/TRI。",
        "- 新 split: seed=42，train/val/test=216/72/72；按 pathology 硬配额，并平衡 vendor 与 field strength。",
        f"- split overview: `{split_overview}`",
        f"- split distributions: `{split_distributions}`",
        "",
        "## 预处理修改",
        "",
        "- patient-level split 后再生成 processed dataset。",
        "- 任务仍为 SA ED/ES myocardium segmentation，MYO label=2。",
        "- 图像使用 per-slice robust percentile normalization，spacing 不重采样但完整记录。",
        "- 无 MYO slice 默认不进入训练样本，但写入 slice_records；topology abnormal slice 不再过滤，只记录。",
        f"- processed dataset: `{context['data_dir']}`",
        f"- split manifest: `{context['split_manifest']}`",
        "",
        "## 训练配置",
        "",
        "- R2Net: warm start 自旧最佳 smooth/hd-topology checkpoint，20 epoch，lr=2e-5，flow smooth=2500，boundary=0.2，clDice=0.1。",
        "- baseline: original TEDS integrator，warm start 自旧 original TEDS checkpoint，20 epoch，lr=2e-5，原始 dice+grad loss。",
        f"- R2Net checkpoint: `{context['r2net_checkpoint']}`",
        f"- baseline checkpoint: `{context['baseline_checkpoint']}`",
        f"- R2Net log: `{context['r2net_log']}`",
        f"- baseline log: `{context['baseline_log']}`",
        "",
        "## 总体结果",
        "",
        *markdown_table(comparison_rows, columns),
        "",
        "## 后处理",
        "",
        f"- 策略: `{POSTPROCESS_CONFIG['name']}`，closing radius=1，keep largest connected component，填充额外小孔但保留最大内腔。",
        "- 四组结果均报告：baseline/new model 的无后处理与有后处理。",
        "",
        "## 分组结果",
        "",
        f"- disease/pathology CSV: `{output_dir / 'tables' / 'r2net_no_post_by_pathology.csv'}` 等同目录文件。",
        f"- vendor CSV: `{output_dir / 'tables' / 'r2net_no_post_by_vendor.csv'}` 等同目录文件。",
        f"- field CSV: `{output_dir / 'tables' / 'r2net_no_post_by_field_strength.csv'}` 等同目录文件。",
        f"- phase CSV: `{output_dir / 'tables' / 'r2net_no_post_by_phase.csv'}` 等同目录文件。",
        "",
        "## HD 与 topology failure 分析",
        "",
        f"- R2Net no-post HD top-{context['top_k']}: `{output_dir / 'failures' / ('r2net_no_post_hd_top' + str(context['top_k']) + '.csv')}`",
        f"- R2Net no-post topology failures: `{output_dir / 'failures' / 'r2net_no_post_topology_failures.csv'}`",
        f"- R2Net post remaining failures: `{output_dir / 'failures' / 'r2net_post_topology_failures.csv'}`",
        "- 失败类型已拆分为 extra components、missing holes、extra holes、remote false positives 和 basal/apical 标记。",
        "",
        "## 可视化",
        "",
        f"![new no post topology failures]({context['figures']['r2net_no_post_topology']})",
        "",
        f"![new post topology failures]({context['figures']['r2net_post_topology']})",
        "",
        f"![baseline vs r2net HD outliers]({context['figures']['hd_compare']})",
        "",
        "## 是否超过 baseline",
        "",
        conclusion,
        "",
        "## 下一步建议",
        "",
        "- 对 R2Net 在新 split 上从头训练 200 epoch，或至少延长 warm-start 至 50-100 epoch。",
        "- 将 HD/topology loss 的训练目标与评估失败类型对齐，重点抑制远端假阳性和多连通域。",
        "- 尝试 spacing-aware crop/resample 或记录物理单位 HD，减少 vendor/field spacing 差异的解释偏差。",
        "- 对 basal/apical slice 单独建模或降低其对 topology ring 的硬约束。",
    ]
    (output_dir / "mnms2_stratified_preprocess_v2_report_zh.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    tables_dir = ensure_dir(output_dir / "tables")
    failures_dir = ensure_dir(output_dir / "failures")
    figures_dir = ensure_dir(output_dir / "figures")
    predictions_dir = ensure_dir(output_dir / "predictions")
    dataset_spec = resolve_dataset_spec(args.dataset, args.dataset_registry)
    data_dir = Path(args.data_dir)
    split_manifest = Path(args.split_manifest)
    metadata_by_sample = sample_metadata_map(split_manifest)
    device = resolve_device(args.device)
    thresholds = threshold_values(args.threshold_min, args.threshold_max, args.threshold_step)

    r2net_val, r2net_info = collect_predictions(
        Path(args.r2net_checkpoint),
        "r2net",
        "val",
        dataset_spec,
        data_dir,
        split_manifest,
        metadata_by_sample,
        device,
        args.seed,
        args.max_samples,
    )
    baseline_val, baseline_info = collect_predictions(
        Path(args.baseline_checkpoint),
        "baseline",
        "val",
        dataset_spec,
        data_dir,
        split_manifest,
        metadata_by_sample,
        device,
        args.seed,
        args.max_samples,
    )
    _, r2net_best = threshold_sweep(r2net_val, thresholds, "r2net", tables_dir, args.hd_epsilon)
    _, baseline_best = threshold_sweep(baseline_val, thresholds, "baseline", tables_dir, args.hd_epsilon)

    r2net_test, _ = collect_predictions(
        Path(args.r2net_checkpoint),
        "r2net",
        "test",
        dataset_spec,
        data_dir,
        split_manifest,
        metadata_by_sample,
        device,
        args.seed,
        args.max_samples,
    )
    baseline_test, _ = collect_predictions(
        Path(args.baseline_checkpoint),
        "baseline",
        "test",
        dataset_spec,
        data_dir,
        split_manifest,
        metadata_by_sample,
        device,
        args.seed,
        args.max_samples,
    )

    eval_sets = {
        "baseline_no_post": (baseline_test, baseline_best["threshold"], None, Path(args.baseline_checkpoint)),
        "baseline_post": (baseline_test, baseline_best["threshold"], POSTPROCESS_CONFIG, Path(args.baseline_checkpoint)),
        "r2net_no_post": (r2net_test, r2net_best["threshold"], None, Path(args.r2net_checkpoint)),
        "r2net_post": (r2net_test, r2net_best["threshold"], POSTPROCESS_CONFIG, Path(args.r2net_checkpoint)),
    }

    all_rows = {}
    comparison_rows = []
    sample_fields = None
    for case_name, (predictions, threshold, postprocess_config, checkpoint) in eval_sets.items():
        rows = evaluate_predictions(predictions, threshold, case_name, postprocess_config)
        all_rows[case_name] = rows
        sample_fields = sample_fields or list(rows[0].keys())
        write_csv(predictions_dir / f"{case_name}_eval_per_sample.csv", sample_fields, rows)
        write_group_outputs(tables_dir, case_name, rows)
        write_failure_outputs(failures_dir, case_name, rows, args.top_k)
        comparison_rows.append(
            comparison_row(
                case_name,
                rows,
                threshold,
                checkpoint,
                "none" if not postprocess_config else postprocess_config["name"],
            )
        )

    comparison_fields = list(comparison_rows[0].keys())
    write_csv(output_dir / "comparison.csv", comparison_fields, comparison_rows)
    write_json(
        output_dir / "summary.json",
        {
            "r2net_info": r2net_info,
            "baseline_info": baseline_info,
            "r2net_best_threshold": r2net_best,
            "baseline_best_threshold": baseline_best,
            "postprocess": POSTPROCESS_CONFIG,
            "comparison": comparison_rows,
        },
    )

    r2_topology_rows = sorted(
        [row for row in all_rows["r2net_no_post"] if float(row["correct_topology"]) < 1.0],
        key=lambda row: float(row["hd"]),
        reverse=True,
    )
    r2_post_topology_rows = sorted(
        [row for row in all_rows["r2net_post"] if float(row["correct_topology"]) < 1.0],
        key=lambda row: float(row["hd"]),
        reverse=True,
    )
    r2_topology_fig = figures_dir / "r2net_no_post_topology_failure_overview.png"
    r2_post_topology_fig = figures_dir / "r2net_post_remaining_failure_overview.png"
    hd_compare_fig = figures_dir / "baseline_vs_r2net_hd_outlier_comparison.png"
    save_overview(
        r2_topology_fig,
        "R2Net no-post topology failures",
        r2net_test,
        r2_topology_rows,
        r2net_best["threshold"],
        None,
        args.top_k,
    )
    save_overview(
        r2_post_topology_fig,
        "R2Net postprocess remaining topology failures",
        r2net_test,
        r2_post_topology_rows,
        r2net_best["threshold"],
        POSTPROCESS_CONFIG,
        args.top_k,
    )
    save_side_by_side(
        hd_compare_fig,
        "GT / baseline / R2Net / R2Net+post for HD outliers",
        baseline_test,
        r2net_test,
        all_rows["baseline_no_post"],
        all_rows["r2net_no_post"],
        all_rows["r2net_post"],
        {"baseline": baseline_best["threshold"], "r2net": r2net_best["threshold"]},
        limit=12,
    )

    context = {
        "data_dir": str(data_dir),
        "split_manifest": str(split_manifest),
        "r2net_checkpoint": str(args.r2net_checkpoint),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "r2net_log": "logs/mnms2_r2net_stratified_preprocess_v2_seed42_20260615.out.log",
        "baseline_log": "logs/mnms2_baseline_stratified_preprocess_v2_seed42_20260615.out.log",
        "comparison_rows": comparison_rows,
        "top_k": args.top_k,
        "figures": {
            "r2net_no_post_topology": str(r2_topology_fig),
            "r2net_post_topology": str(r2_post_topology_fig),
            "hd_compare": str(hd_compare_fig),
        },
    }
    write_report(output_dir, context)
    print(f"Wrote fair evaluation report to {output_dir}")
    print(json.dumps({"comparison": comparison_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
