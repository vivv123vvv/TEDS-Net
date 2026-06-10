import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_closing, label as connected_components
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_results import load_model, make_eval_loader, set_random_seed
from utils.acdc_benchmark import (
    build_eval_summary,
    dice_score,
    ensure_dir,
    hausdorff_distance,
    iou_score,
    jacobian_negative_ratio,
    model_parameter_count,
    precision_score,
    recall_score,
    resolve_device,
    symmetric_surface_distances,
    sync_cuda,
    topology_signature,
    write_csv,
    write_json,
)
from utils.dataset_registry import DEFAULT_DATASET_REGISTRY, resolve_dataset_spec


DEFAULT_R2NET_CHECKPOINT = (
    Path("checkpoints") / "mnms2" / "mnms2-r2net-s42-e200" / "best_teds_net.pth"
)
DEFAULT_BASELINE_CHECKPOINT = (
    Path("checkpoints") / "mnms2" / "mnms2-original-teds-s42-e200" / "best_teds_net.pth"
)

SWEEP_FIELDS = [
    "model",
    "split",
    "threshold",
    "sample_count",
    "mean_dice",
    "mean_iou",
    "mean_hd",
    "mean_hd95",
    "mean_assd",
    "mean_precision",
    "mean_recall",
    "correct_topology_rate",
    "topology_failures",
    "hd_p95",
    "hd_max",
    "mean_forward_ms",
    "mean_jacobian_neg_ratio",
]

SAMPLE_FIELDS = [
    "benchmark_case",
    "sample_id",
    "case_id",
    "threshold",
    "postprocess",
    "forward_ms",
    "dice",
    "iou",
    "hd",
    "hd95",
    "assd",
    "precision",
    "recall",
    "correct_topology",
    "pred_components",
    "pred_holes",
    "target_components",
    "target_holes",
    "jacobian_neg_ratio",
]

COMPARISON_FIELDS = [
    "case",
    "model",
    "checkpoint",
    "split",
    "threshold",
    "postprocess",
    "sample_count",
    "dice",
    "iou",
    "hd",
    "hd95",
    "assd",
    "precision",
    "recall",
    "correct_topology",
    "topology_failures",
    "mean_forward_ms",
    "mean_jacobian_neg_ratio",
    "eval_per_sample",
]

POSTPROCESS_FIELDS = [
    "model",
    "split",
    "threshold",
    "postprocess",
    "sample_count",
    "mean_dice",
    "mean_iou",
    "mean_hd",
    "mean_hd95",
    "mean_assd",
    "mean_precision",
    "mean_recall",
    "correct_topology_rate",
    "topology_failures",
    "hd_p95",
    "hd_max",
    "mean_forward_ms",
    "mean_jacobian_neg_ratio",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run M&Ms-2 validation threshold sweep and lightweight postprocess checks."
    )
    parser.add_argument("--dataset", default="mnms2")
    parser.add_argument("--dataset-registry", default=str(DEFAULT_DATASET_REGISTRY))
    parser.add_argument("--r2net-checkpoint", default=str(DEFAULT_R2NET_CHECKPOINT))
    parser.add_argument("--baseline-checkpoint", default=str(DEFAULT_BASELINE_CHECKPOINT))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument(
        "--output-dir",
        default=str(Path("reports") / "benchmarks" / "mnms2" / "threshold_sweep"),
    )
    parser.add_argument(
        "--postprocess-output-dir",
        default=str(Path("reports") / "benchmarks" / "mnms2" / "postprocess_sweep"),
    )
    parser.add_argument("--threshold-min", type=float, default=0.20)
    parser.add_argument("--threshold-max", type=float, default=0.80)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--hd-epsilon", type=float, default=0.01)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-postprocess", action="store_true")
    return parser.parse_args()


def threshold_values(start, stop, step):
    values = []
    current = float(start)
    while current <= float(stop) + 1e-9:
        values.append(round(current, 2))
        current += float(step)
    return values


def output_from_model(outputs):
    if len(outputs) == 2:
        return outputs
    if len(outputs) == 3:
        return outputs[0], outputs[2]
    raise ValueError(f"Unexpected model outputs length: {len(outputs)}")


def collect_predictions(
    checkpoint_path,
    model_name,
    split,
    dataset_spec,
    data_dir,
    split_manifest,
    device,
    seed,
    max_samples=None,
):
    checkpoint_path = Path(checkpoint_path)
    model, _, params = load_model(
        checkpoint_path,
        device,
        seed=seed,
        dataset_spec=dataset_spec,
    )
    set_random_seed(params.seed)
    parameter_count = model_parameter_count(model)
    _, loader = make_eval_loader(data_dir, split_manifest, split, dataset_spec)

    predictions = []
    description = f"{model_name} {split}"
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=description)):
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

            predictions.append(
                {
                    "sample_id": sample_ids[0],
                    "case_id": case_ids[0],
                    "pred_prob": np.squeeze(pred_prob.numpy()).astype(np.float32),
                    "target": np.squeeze(gt_masks.numpy()).astype(bool),
                    "forward_ms": float(forward_ms),
                    "jacobian_neg_ratio": jacobian_negative_ratio(flow_upsamp.detach().cpu()),
                }
            )

    model_info = {
        "model": model_name,
        "checkpoint_path": str(checkpoint_path),
        "integrator": params.network.integrator,
        "seed": int(params.seed),
        "parameter_count": int(parameter_count),
        "parameter_count_x1e5": float(parameter_count / 100000.0),
    }
    return predictions, model_info


def disk_structure(radius):
    radius = int(radius)
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    coords = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return (xx * xx + yy * yy) <= (radius * radius)


def remove_small_objects(mask, min_size):
    mask = np.asarray(mask, dtype=bool).copy()
    labels, count = connected_components(mask, structure=np.ones((3, 3), dtype=np.int8))
    if count == 0:
        return mask
    areas = np.bincount(labels.ravel())
    remove = areas < int(min_size)
    remove[0] = False
    mask[remove[labels]] = False
    return mask


def keep_largest_component(mask):
    mask = np.asarray(mask, dtype=bool)
    labels, count = connected_components(mask, structure=np.ones((3, 3), dtype=np.int8))
    if count <= 1:
        return mask.copy()
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    largest = int(np.argmax(areas))
    return labels == largest


def remove_small_holes(mask, max_size):
    mask = np.asarray(mask, dtype=bool).copy()
    labels, count = connected_components(~mask, structure=np.ones((3, 3), dtype=np.int8))
    for component_idx in range(1, count + 1):
        component = labels == component_idx
        touches_border = (
            component[0, :].any()
            or component[-1, :].any()
            or component[:, 0].any()
            or component[:, -1].any()
        )
        if not touches_border and int(component.sum()) <= int(max_size):
            mask[component] = True
    return mask


def apply_postprocess(mask, config):
    if not config or config.get("name") == "none":
        return np.asarray(mask, dtype=bool).copy()

    processed = np.asarray(mask, dtype=bool).copy()
    radius = config.get("closing_radius")
    if radius:
        processed = binary_closing(processed, structure=disk_structure(radius))
    if config.get("remove_small_objects"):
        processed = remove_small_objects(processed, config["remove_small_objects"])
    if config.get("keep_largest_component"):
        processed = keep_largest_component(processed)
    if config.get("remove_small_holes"):
        processed = remove_small_holes(processed, config["remove_small_holes"])
    return processed.astype(bool)


def postprocess_configs():
    configs = [{"name": "none"}]
    for min_size in [4, 8, 16, 32, 64, 128]:
        configs.append(
            {
                "name": f"remove_small_objects_{min_size}",
                "remove_small_objects": min_size,
            }
        )
    configs.append({"name": "keep_largest_component", "keep_largest_component": True})
    for radius in [1, 2]:
        configs.append({"name": f"closing_r{radius}", "closing_radius": radius})
        for min_size in [16, 32, 64]:
            configs.append(
                {
                    "name": f"closing_r{radius}_remove_small_objects_{min_size}",
                    "closing_radius": radius,
                    "remove_small_objects": min_size,
                }
            )
        configs.append(
            {
                "name": f"closing_r{radius}_keep_largest_component",
                "closing_radius": radius,
                "keep_largest_component": True,
            }
        )
    for max_size in [8, 16, 32]:
        configs.append({"name": f"remove_small_holes_{max_size}", "remove_small_holes": max_size})
    for radius in [1, 2]:
        configs.append(
            {
                "name": f"closing_r{radius}_remove_small_holes_16",
                "closing_radius": radius,
                "remove_small_holes": 16,
            }
        )
    return configs


def evaluate_predictions(
    predictions,
    threshold,
    benchmark_case,
    split,
    checkpoint_path,
    parameter_count,
    postprocess_config=None,
):
    postprocess_name = "none" if not postprocess_config else postprocess_config.get("name", "custom")
    sample_rows = []
    for item in predictions:
        pred_mask = item["pred_prob"] > float(threshold)
        pred_mask = apply_postprocess(pred_mask, postprocess_config)
        target_mask = item["target"].astype(bool)

        pred_components, pred_holes = topology_signature(pred_mask)
        target_components, target_holes = topology_signature(target_mask)
        correct_topology = float(
            (pred_components, pred_holes) == (target_components, target_holes)
        )
        surface_distances = symmetric_surface_distances(pred_mask, target_mask)

        sample_rows.append(
            {
                "benchmark_case": benchmark_case,
                "sample_id": item["sample_id"],
                "case_id": item["case_id"],
                "threshold": float(threshold),
                "postprocess": postprocess_name,
                "forward_ms": item["forward_ms"],
                "dice": dice_score(pred_mask.astype(np.float32), target_mask.astype(np.float32)),
                "iou": iou_score(pred_mask, target_mask),
                "hd": hausdorff_distance(pred_mask, target_mask),
                "hd95": float(np.percentile(surface_distances, 95)),
                "assd": float(np.mean(surface_distances)),
                "precision": precision_score(pred_mask, target_mask),
                "recall": recall_score(pred_mask, target_mask),
                "correct_topology": correct_topology,
                "pred_components": pred_components,
                "pred_holes": pred_holes,
                "target_components": target_components,
                "target_holes": target_holes,
                "jacobian_neg_ratio": item["jacobian_neg_ratio"],
            }
        )

    summary = build_eval_summary(
        sample_rows,
        peak_mem_mb=None,
        benchmark_case=benchmark_case,
        split=split,
        checkpoint_path=checkpoint_path,
        parameter_count=parameter_count,
    )
    summary.update(
        {
            "threshold": float(threshold),
            "postprocess": postprocess_name,
            "topology_failures": int(
                sum(1 for row in sample_rows if float(row["correct_topology"]) < 1.0)
            ),
            "hd_p95": percentile([row["hd"] for row in sample_rows], 95),
            "hd_max": max([row["hd"] for row in sample_rows], default=None),
        }
    )
    return sample_rows, summary


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def sweep_thresholds(predictions, thresholds, model_info, split, model_name):
    rows = []
    for threshold in tqdm(thresholds, desc=f"{model_name} thresholds"):
        _, summary = evaluate_predictions(
            predictions,
            threshold=threshold,
            benchmark_case=f"{model_name}@thr{threshold:.2f}",
            split=split,
            checkpoint_path=model_info["checkpoint_path"],
            parameter_count=model_info["parameter_count"],
        )
        rows.append(
            {
                "model": model_name,
                "split": split,
                "threshold": float(threshold),
                "sample_count": summary["sample_count"],
                "mean_dice": summary["mean_dice"],
                "mean_iou": summary["mean_iou"],
                "mean_hd": summary["mean_hd"],
                "mean_hd95": summary["mean_hd95"],
                "mean_assd": summary["mean_assd"],
                "mean_precision": summary["mean_precision"],
                "mean_recall": summary["mean_recall"],
                "correct_topology_rate": summary["correct_topology_rate"],
                "topology_failures": summary["topology_failures"],
                "hd_p95": summary["hd_p95"],
                "hd_max": summary["hd_max"],
                "mean_forward_ms": summary["mean_forward_ms"],
                "mean_jacobian_neg_ratio": summary["mean_jacobian_neg_ratio"],
            }
        )
    return rows


def select_best_row(rows, hd_epsilon=0.01):
    min_hd = min(float(row["mean_hd"]) for row in rows)
    close_rows = [row for row in rows if float(row["mean_hd"]) <= min_hd + float(hd_epsilon)]
    return sorted(
        close_rows,
        key=lambda row: (
            -float(row["correct_topology_rate"]),
            -float(row["mean_dice"]),
            abs(float(row.get("threshold", 0.5)) - 0.5),
            str(row.get("postprocess", "")) != "none",
            str(row.get("postprocess", "")),
        ),
    )[0]


def projected_rows(rows, fieldnames):
    return [{field: row.get(field) for field in fieldnames} for row in rows]


def write_projected_csv(path, fieldnames, rows):
    write_csv(path, fieldnames, projected_rows(rows, fieldnames))


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, (float, int)):
        return f"{value:.6f}"
    return str(value)


def write_threshold_markdown(path, title, rows, best_row, hd_epsilon):
    lines = [
        f"# {title}",
        "",
        f"- Selection: lowest validation HD; within `{hd_epsilon:.4f}` HD, prefer topology rate then Dice.",
        f"- Best threshold: `{best_row['threshold']:.2f}`",
        f"- Best Dice: `{best_row['mean_dice']:.6f}`",
        f"- Best IoU: `{best_row['mean_iou']:.6f}`",
        f"- Best HD: `{best_row['mean_hd']:.6f}`",
        f"- Best HD95: `{best_row['mean_hd95']:.6f}`",
        f"- Best ASSD: `{best_row['mean_assd']:.6f}`",
        f"- Best Precision: `{best_row['mean_precision']:.6f}`",
        f"- Best Recall: `{best_row['mean_recall']:.6f}`",
        f"- Best correct topology: `{best_row['correct_topology_rate']:.6f}`",
        "",
        "| threshold | Dice | IoU | HD | HD95 | ASSD | Precision | Recall | Correct topology | topology failures | HD p95 | HD max |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {threshold:.2f} | {dice} | {iou} | {hd} | {hd95} | {assd} | {precision} | {recall} | {topology} | {failures} | {hd_p95} | {hd_max} |".format(
                threshold=float(row["threshold"]),
                dice=fmt(row["mean_dice"]),
                iou=fmt(row["mean_iou"]),
                hd=fmt(row["mean_hd"]),
                hd95=fmt(row["mean_hd95"]),
                assd=fmt(row["mean_assd"]),
                precision=fmt(row["mean_precision"]),
                recall=fmt(row["mean_recall"]),
                topology=fmt(row["correct_topology_rate"]),
                failures=row["topology_failures"],
                hd_p95=fmt(row["hd_p95"]),
                hd_max=fmt(row["hd_max"]),
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_markdown(path, rows, title):
    lines = [
        f"# {title}",
        "",
        "| case | threshold | postprocess | Dice | IoU | HD | HD95 | ASSD | Precision | Recall | Correct topology | topology failures | per-sample CSV |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        csv_path = row.get("eval_per_sample", "")
        csv_rel = Path(csv_path).name if csv_path else ""
        lines.append(
            "| {case} | {threshold:.2f} | {postprocess} | {dice} | {iou} | {hd} | {hd95} | {assd} | {precision} | {recall} | {topology} | {failures} | {csv} |".format(
                case=row["case"],
                threshold=float(row["threshold"]),
                postprocess=row["postprocess"],
                dice=fmt(row["dice"]),
                iou=fmt(row["iou"]),
                hd=fmt(row["hd"]),
                hd95=fmt(row["hd95"]),
                assd=fmt(row["assd"]),
                precision=fmt(row["precision"]),
                recall=fmt(row["recall"]),
                topology=fmt(row["correct_topology"]),
                failures=row["topology_failures"],
                csv=f"[csv]({csv_rel})" if csv_rel else "-",
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_postprocess_markdown(path, rows, best_row, hd_epsilon):
    lines = [
        "# M&Ms-2 R2Net Postprocess Sweep",
        "",
        f"- Selection: lowest validation HD; within `{hd_epsilon:.4f}` HD, prefer topology rate then Dice.",
        f"- Best postprocess: `{best_row['postprocess']}`",
        f"- Threshold: `{best_row['threshold']:.2f}`",
        f"- Best Dice: `{best_row['mean_dice']:.6f}`",
        f"- Best IoU: `{best_row['mean_iou']:.6f}`",
        f"- Best HD: `{best_row['mean_hd']:.6f}`",
        f"- Best HD95: `{best_row['mean_hd95']:.6f}`",
        f"- Best ASSD: `{best_row['mean_assd']:.6f}`",
        f"- Best Precision: `{best_row['mean_precision']:.6f}`",
        f"- Best Recall: `{best_row['mean_recall']:.6f}`",
        f"- Best correct topology: `{best_row['correct_topology_rate']:.6f}`",
        "",
        "| postprocess | Dice | IoU | HD | HD95 | ASSD | Precision | Recall | Correct topology | topology failures | HD p95 | HD max |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {postprocess} | {dice} | {iou} | {hd} | {hd95} | {assd} | {precision} | {recall} | {topology} | {failures} | {hd_p95} | {hd_max} |".format(
                postprocess=row["postprocess"],
                dice=fmt(row["mean_dice"]),
                iou=fmt(row["mean_iou"]),
                hd=fmt(row["mean_hd"]),
                hd95=fmt(row["mean_hd95"]),
                assd=fmt(row["mean_assd"]),
                precision=fmt(row["mean_precision"]),
                recall=fmt(row["mean_recall"]),
                topology=fmt(row["correct_topology_rate"]),
                failures=row["topology_failures"],
                hd_p95=fmt(row["hd_p95"]),
                hd_max=fmt(row["hd_max"]),
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_eval_files(output_dir, file_stem, sample_rows, summary):
    sample_csv = Path(output_dir) / f"{file_stem}_eval_per_sample.csv"
    summary_json = Path(output_dir) / f"{file_stem}_eval_summary.json"
    write_projected_csv(sample_csv, SAMPLE_FIELDS, sample_rows)
    write_json(summary_json, summary)
    return sample_csv, summary_json


def comparison_row(label, model_name, model_info, threshold, postprocess, summary, sample_csv):
    return {
        "case": label,
        "model": model_name,
        "checkpoint": model_info["checkpoint_path"],
        "split": summary["split"],
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
        "correct_topology": summary["correct_topology_rate"],
        "topology_failures": summary["topology_failures"],
        "mean_forward_ms": summary["mean_forward_ms"],
        "mean_jacobian_neg_ratio": summary["mean_jacobian_neg_ratio"],
        "eval_per_sample": str(sample_csv),
    }


def run_test_eval(predictions, model_info, model_name, split, threshold, output_dir, label, postprocess_config=None):
    postprocess_name = "none" if not postprocess_config else postprocess_config["name"]
    safe_threshold = f"{threshold:.2f}".replace(".", "p")
    safe_postprocess = postprocess_name.replace(".", "p")
    file_stem = f"{model_name}_test_thr{safe_threshold}_{safe_postprocess}"
    sample_rows, summary = evaluate_predictions(
        predictions,
        threshold=threshold,
        benchmark_case=label,
        split=split,
        checkpoint_path=model_info["checkpoint_path"],
        parameter_count=model_info["parameter_count"],
        postprocess_config=postprocess_config,
    )
    sample_csv, _ = write_eval_files(output_dir, file_stem, sample_rows, summary)
    return comparison_row(
        label,
        model_name,
        model_info,
        threshold,
        postprocess_name,
        summary,
        sample_csv,
    ), sample_csv, sample_rows, summary


def run_postprocess_sweep(
    r2net_val_predictions,
    r2net_test_predictions,
    r2net_info,
    baseline_best_test_row,
    best_threshold,
    val_split,
    test_split,
    output_dir,
    hd_epsilon,
):
    output_dir = ensure_dir(output_dir)
    rows = []
    configs = postprocess_configs()
    for config in tqdm(configs, desc="r2net postprocess"):
        _, summary = evaluate_predictions(
            r2net_val_predictions,
            threshold=best_threshold,
            benchmark_case=f"r2net@thr{best_threshold:.2f}+{config['name']}",
            split=val_split,
            checkpoint_path=r2net_info["checkpoint_path"],
            parameter_count=r2net_info["parameter_count"],
            postprocess_config=config,
        )
        rows.append(
            {
                "model": "r2net",
                "split": val_split,
                "threshold": float(best_threshold),
                "postprocess": config["name"],
                "sample_count": summary["sample_count"],
                "mean_dice": summary["mean_dice"],
                "mean_iou": summary["mean_iou"],
                "mean_hd": summary["mean_hd"],
                "mean_hd95": summary["mean_hd95"],
                "mean_assd": summary["mean_assd"],
                "mean_precision": summary["mean_precision"],
                "mean_recall": summary["mean_recall"],
                "correct_topology_rate": summary["correct_topology_rate"],
                "topology_failures": summary["topology_failures"],
                "hd_p95": summary["hd_p95"],
                "hd_max": summary["hd_max"],
                "mean_forward_ms": summary["mean_forward_ms"],
                "mean_jacobian_neg_ratio": summary["mean_jacobian_neg_ratio"],
                "config": config,
            }
        )

    best_row = select_best_row(rows, hd_epsilon=hd_epsilon)
    best_config = next(config for config in configs if config["name"] == best_row["postprocess"])

    write_projected_csv(
        output_dir / "r2net_val_postprocess_sweep.csv",
        POSTPROCESS_FIELDS,
        rows,
    )
    write_postprocess_markdown(
        output_dir / "r2net_val_postprocess_sweep.md",
        rows,
        best_row,
        hd_epsilon,
    )

    comparison_rows = []
    no_post_row, _, _, no_post_summary = run_test_eval(
        r2net_test_predictions,
        r2net_info,
        "r2net",
        test_split,
        best_threshold,
        output_dir,
        f"r2net@val_best_thr{best_threshold:.2f}",
        postprocess_config={"name": "none"},
    )
    comparison_rows.append(no_post_row)

    post_row, _, _, post_summary = run_test_eval(
        r2net_test_predictions,
        r2net_info,
        "r2net",
        test_split,
        best_threshold,
        output_dir,
        f"r2net@val_best_thr{best_threshold:.2f}+{best_config['name']}",
        postprocess_config=best_config,
    )
    comparison_rows.append(post_row)
    comparison_rows.append(baseline_best_test_row)

    write_projected_csv(output_dir / "comparison.csv", COMPARISON_FIELDS, comparison_rows)
    write_comparison_markdown(
        output_dir / "comparison.md",
        comparison_rows,
        "M&Ms-2 Postprocess Test Comparison",
    )
    write_json(
        output_dir / "summary.json",
        {
            "best_threshold": float(best_threshold),
            "best_postprocess": best_config,
            "val_best_row": {key: value for key, value in best_row.items() if key != "config"},
            "test_no_postprocess": no_post_summary,
            "test_best_postprocess": post_summary,
            "baseline_best_threshold_test": baseline_best_test_row,
        },
    )
    return best_row, best_config, post_row


def main():
    args = parse_args()
    dataset_spec = resolve_dataset_spec(args.dataset, args.dataset_registry)
    data_dir = Path(args.data_dir) if args.data_dir else dataset_spec.data_dir
    split_manifest = Path(args.split_manifest) if args.split_manifest else dataset_spec.split_manifest
    output_dir = ensure_dir(args.output_dir)
    thresholds = threshold_values(args.threshold_min, args.threshold_max, args.threshold_step)
    device = resolve_device(args.device)

    print(f"Using dataset: {dataset_spec.display_name}")
    print(f"Using device: {device}")
    print(f"Thresholds: {thresholds[0]:.2f}..{thresholds[-1]:.2f} step {args.threshold_step:.2f}")

    r2net_val, r2net_info = collect_predictions(
        args.r2net_checkpoint,
        "r2net",
        args.val_split,
        dataset_spec,
        data_dir,
        split_manifest,
        device,
        args.seed,
        max_samples=args.max_samples,
    )
    baseline_val, baseline_info = collect_predictions(
        args.baseline_checkpoint,
        "baseline",
        args.val_split,
        dataset_spec,
        data_dir,
        split_manifest,
        device,
        args.seed,
        max_samples=args.max_samples,
    )

    r2net_sweep_rows = sweep_thresholds(r2net_val, thresholds, r2net_info, args.val_split, "r2net")
    baseline_sweep_rows = sweep_thresholds(
        baseline_val,
        thresholds,
        baseline_info,
        args.val_split,
        "baseline",
    )
    r2net_best = select_best_row(r2net_sweep_rows, hd_epsilon=args.hd_epsilon)
    baseline_best = select_best_row(baseline_sweep_rows, hd_epsilon=args.hd_epsilon)

    write_projected_csv(
        output_dir / "r2net_val_threshold_sweep.csv",
        SWEEP_FIELDS,
        r2net_sweep_rows,
    )
    write_threshold_markdown(
        output_dir / "r2net_val_threshold_sweep.md",
        "M&Ms-2 R2Net Validation Threshold Sweep",
        r2net_sweep_rows,
        r2net_best,
        args.hd_epsilon,
    )
    write_projected_csv(
        output_dir / "baseline_val_threshold_sweep.csv",
        SWEEP_FIELDS,
        baseline_sweep_rows,
    )
    write_threshold_markdown(
        output_dir / "baseline_val_threshold_sweep.md",
        "M&Ms-2 Baseline Validation Threshold Sweep",
        baseline_sweep_rows,
        baseline_best,
        args.hd_epsilon,
    )

    r2net_test, _ = collect_predictions(
        args.r2net_checkpoint,
        "r2net",
        args.test_split,
        dataset_spec,
        data_dir,
        split_manifest,
        device,
        args.seed,
        max_samples=args.max_samples,
    )
    baseline_test, _ = collect_predictions(
        args.baseline_checkpoint,
        "baseline",
        args.test_split,
        dataset_spec,
        data_dir,
        split_manifest,
        device,
        args.seed,
        max_samples=args.max_samples,
    )

    comparison_rows = []
    r2net_default_row, _, _, r2net_default_summary = run_test_eval(
        r2net_test,
        r2net_info,
        "r2net",
        args.test_split,
        0.5,
        output_dir,
        "r2net@thr0.50",
    )
    comparison_rows.append(r2net_default_row)
    r2net_best_row, r2net_best_csv, _, r2net_best_summary = run_test_eval(
        r2net_test,
        r2net_info,
        "r2net",
        args.test_split,
        r2net_best["threshold"],
        output_dir,
        f"r2net@val_best_thr{r2net_best['threshold']:.2f}",
    )
    comparison_rows.append(r2net_best_row)
    baseline_default_row, _, _, baseline_default_summary = run_test_eval(
        baseline_test,
        baseline_info,
        "baseline",
        args.test_split,
        0.5,
        output_dir,
        "baseline@thr0.50",
    )
    comparison_rows.append(baseline_default_row)
    baseline_best_row, _, _, baseline_best_summary = run_test_eval(
        baseline_test,
        baseline_info,
        "baseline",
        args.test_split,
        baseline_best["threshold"],
        output_dir,
        f"baseline@val_best_thr{baseline_best['threshold']:.2f}",
    )
    comparison_rows.append(baseline_best_row)

    write_projected_csv(output_dir / "comparison.csv", COMPARISON_FIELDS, comparison_rows)
    write_comparison_markdown(
        output_dir / "comparison.md",
        comparison_rows,
        "M&Ms-2 Threshold Test Comparison",
    )
    write_projected_csv(
        output_dir / "test_threshold_comparison.csv",
        COMPARISON_FIELDS,
        comparison_rows,
    )
    write_comparison_markdown(
        output_dir / "test_threshold_comparison.md",
        comparison_rows,
        "M&Ms-2 Threshold Test Comparison",
    )

    summary = {
        "dataset": dataset_spec.to_dict(),
        "threshold_range": {
            "min": float(args.threshold_min),
            "max": float(args.threshold_max),
            "step": float(args.threshold_step),
        },
        "selection": {
            "primary": "lowest validation mean_hd",
            "hd_epsilon": float(args.hd_epsilon),
            "tie_breakers": ["higher correct_topology_rate", "higher mean_dice"],
        },
        "r2net": {
            "checkpoint": r2net_info["checkpoint_path"],
            "val_best_threshold": float(r2net_best["threshold"]),
            "val_best_row": r2net_best,
            "test_default_threshold": r2net_default_summary,
            "test_best_threshold": r2net_best_summary,
            "test_best_eval_per_sample": str(r2net_best_csv),
        },
        "baseline": {
            "checkpoint": baseline_info["checkpoint_path"],
            "val_best_threshold": float(baseline_best["threshold"]),
            "val_best_row": baseline_best,
            "test_default_threshold": baseline_default_summary,
            "test_best_threshold": baseline_best_summary,
        },
        "comparison_csv": str(output_dir / "comparison.csv"),
        "comparison_md": str(output_dir / "comparison.md"),
    }
    write_json(output_dir / "summary.json", summary)

    if not args.skip_postprocess:
        best_post_row, best_config, post_test_row = run_postprocess_sweep(
            r2net_val,
            r2net_test,
            r2net_info,
            baseline_best_row,
            r2net_best["threshold"],
            args.val_split,
            args.test_split,
            args.postprocess_output_dir,
            args.hd_epsilon,
        )
        summary["postprocess"] = {
            "best_val_row": {key: value for key, value in best_post_row.items() if key != "config"},
            "best_config": best_config,
            "test_row": post_test_row,
            "output_dir": str(args.postprocess_output_dir),
        }
        write_json(output_dir / "summary.json", summary)

    print(f"R2Net best val threshold: {r2net_best['threshold']:.2f}")
    print(f"Baseline best val threshold: {baseline_best['threshold']:.2f}")
    print(f"Wrote threshold sweep artifacts to {output_dir}")


if __name__ == "__main__":
    main()
