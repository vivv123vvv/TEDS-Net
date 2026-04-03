import json
import os
import random
import subprocess

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

from utils.acdc_metrics import (
    calculate_betti_numbers,
    compute_distance_metrics,
    compose_backward_flows,
    dice_coefficient,
    folding_ratio,
    project_probability_to_annulus,
)
from utils.acdc_reporting import write_acdc_report

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _resize_2d_array(array_2d, target_shape, is_label):
    tensor = torch.from_numpy(array_2d).unsqueeze(0).unsqueeze(0).float()
    mode = "nearest" if is_label else "bilinear"
    kwargs = {} if is_label else {"align_corners": False}
    resized = F.interpolate(tensor, size=tuple(target_shape), mode=mode, **kwargs)
    resized = resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    if is_label:
        return (resized > 0.5).astype(np.float32)
    return resized


def _git_output(project_root, *args):
    try:
        return subprocess.check_output(args, cwd=project_root, text=True).strip()
    except Exception:
        return "unknown"


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _label_slug(label):
    mapping = {
        "Dice 最好": "best",
        "Dice 中位": "median",
        "Dice 最差": "worst",
        "固定随机样本": "random",
    }
    return mapping.get(label, "case")


def _select_visualization_cases(per_case_metrics, seed):
    if not per_case_metrics:
        return []

    sorted_cases = sorted(per_case_metrics, key=lambda item: item["dice"])
    chosen = []
    used = set()

    def add_case(label, case):
        if case["sample_id"] in used:
            return
        chosen.append((label, case))
        used.add(case["sample_id"])

    add_case("Dice 最好", sorted_cases[-1])
    add_case("Dice 中位", sorted_cases[len(sorted_cases) // 2])
    add_case("Dice 最差", sorted_cases[0])

    remaining = [case for case in per_case_metrics if case["sample_id"] not in used]
    random_case = random.Random(seed).choice(remaining or per_case_metrics)
    add_case("固定随机样本", random_case)
    return chosen


def _render_visualization(image, prior, prediction, label, title, output_path):
    fig, axes = plt.subplots(ncols=4, figsize=(18, 5))
    panels = [
        ("原图切片", None, None),
        ("先验形状", prior, "deepskyblue"),
        ("预测结果", prediction, "red"),
        ("预测与标注对比", (prediction, label), ("red", "lime")),
    ]

    for axis, (panel_title, overlay, colour) in zip(axes, panels):
        axis.imshow(image, cmap="gray")
        if overlay is None:
            pass
        elif isinstance(overlay, tuple):
            axis.contour(overlay[1], levels=[0.5], colors=colour[1], linewidths=2, linestyles="--")
            axis.contour(overlay[0], levels=[0.5], colors=colour[0], linewidths=2)
        else:
            axis.contour(overlay, levels=[0.5], colors=colour, linewidths=2)
        axis.set_title(panel_title)
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _extract_model_outputs(outputs):
    if isinstance(outputs, dict):
        final_probability = outputs["final_mask"][0, 0].detach().cpu().numpy().astype(np.float32)
        posed_prior = outputs.get("posed_prior_mask")
        posed_prior = None if posed_prior is None else posed_prior[0, 0].detach().cpu().numpy().astype(np.float32)
        pose_params = outputs.get("pose_params")
        pose_params = None if pose_params is None else pose_params[0].detach().cpu().numpy().astype(np.float32)
        final_flow = outputs.get("composed_flow")
        final_flow = None if final_flow is None else final_flow[0].detach().cpu().numpy()
        return final_probability, posed_prior, final_flow, pose_params

    prediction = outputs[0][0, 0].detach().cpu().numpy().astype(np.float32)
    if len(outputs) == 3:
        final_flow = compose_backward_flows(outputs[1], outputs[2])[0].detach().cpu().numpy()
    else:
        final_flow = outputs[1][0].detach().cpu().numpy()
    return prediction, None, final_flow, None


def _build_case_cache(model, dataloader, device, params, max_batches=0):
    case_cache = []
    model.eval()
    with torch.no_grad():
        for case_index, (image, prior_shape, target, metadata) in enumerate(dataloader):
            if max_batches and case_index >= max_batches:
                break

            image_batch = image.unsqueeze(0).to(device)
            prior_batch = prior_shape.unsqueeze(0).to(device)
            outputs = model(image_batch, prior_batch)
            probability, posed_prior, final_flow, pose_params = _extract_model_outputs(outputs)

            original_shape = metadata["original_shape"]
            sample_id = f"{metadata['patient_id']}_{metadata['frame_id']}_slice{int(metadata['slice_index']):03d}"
            restored_probability = _resize_2d_array(probability, original_shape, is_label=False)
            restored_prior = None
            if posed_prior is not None:
                restored_prior = _resize_2d_array(posed_prior, original_shape, is_label=False)

            case_cache.append(
                {
                    "sample_id": sample_id,
                    "relative_path": metadata["relative_path"],
                    "patient_id": metadata["patient_id"],
                    "frame_id": metadata["frame_id"],
                    "slice_index": int(metadata["slice_index"]),
                    "source_subset": metadata["source_subset"],
                    "spacing": tuple(float(v) for v in metadata["spacing"]),
                    "gt_betti": list(metadata["betti"]),
                    "pose_target": tuple(float(v) for v in metadata.get("pose_target", (0.0, 0.0, 1.0, 1.0))),
                    "original_shape": tuple(int(v) for v in metadata["original_shape"]),
                    "original_image": metadata["original_image"].astype(np.float32),
                    "original_label": metadata["original_label"].astype(np.float32),
                    "probability": restored_probability.astype(np.float32),
                    "posed_prior": restored_prior,
                    "pose_params": np.asarray(
                        pose_params if pose_params is not None else metadata.get("pose_target", (0.0, 0.0, 1.0, 1.0)),
                        dtype=np.float32,
                    ),
                    "folding_ratio": float(folding_ratio(final_flow)) if final_flow is not None else 0.0,
                }
            )
    return case_cache


def _metrics_for_threshold(case_cache, params, threshold):
    per_case_metrics = []
    base_outer = float(params.dataset.ps_meas[0])
    base_inner = max(float(params.dataset.ps_meas[0] - params.dataset.ps_meas[1]), 1.0)
    projection_count = 0

    for case in case_cache:
        prediction = (case["probability"] >= threshold).astype(np.float32)
        pred_betti = calculate_betti_numbers(prediction)
        projection_applied = 0
        if params.dataset.topology_projection and pred_betti != (1, 1):
            prediction = project_probability_to_annulus(
                probability=case["probability"],
                pose_params=case["pose_params"],
                base_outer_radius=base_outer,
                base_inner_radius=base_inner,
                margin=float(params.dataset.topology_margin),
                threshold=float(params.dataset.projection_threshold),
                num_angles=int(params.dataset.projection_angles),
            ).astype(np.float32)
            pred_betti = calculate_betti_numbers(prediction)
            projection_applied = 1
            projection_count += 1

        hd, hd95, assd = compute_distance_metrics(prediction, case["original_label"], case["spacing"])
        per_case_metrics.append(
            {
                "sample_id": case["sample_id"],
                "relative_path": case["relative_path"],
                "patient_id": case["patient_id"],
                "frame_id": case["frame_id"],
                "slice_index": case["slice_index"],
                "source_subset": case["source_subset"],
                "dice": float(dice_coefficient(prediction, case["original_label"])),
                "hd": float(hd),
                "hd95": float(hd95),
                "assd": float(assd),
                "gt_betti": list(case["gt_betti"]),
                "pred_betti": list(pred_betti),
                "topology_match": int(tuple(case["gt_betti"]) == tuple(pred_betti)),
                "folding_ratio": float(case["folding_ratio"]),
                "projection_applied": projection_applied,
            }
        )

    summary_metrics = {
        "dice_mean": float(np.mean([item["dice"] for item in per_case_metrics])),
        "dice_std": float(np.std([item["dice"] for item in per_case_metrics])),
        "hd_mean": float(np.mean([item["hd"] for item in per_case_metrics])),
        "hd_std": float(np.std([item["hd"] for item in per_case_metrics])),
        "hd95_mean": float(np.mean([item["hd95"] for item in per_case_metrics])),
        "hd95_std": float(np.std([item["hd95"] for item in per_case_metrics])),
        "assd_mean": float(np.mean([item["assd"] for item in per_case_metrics])),
        "assd_std": float(np.std([item["assd"] for item in per_case_metrics])),
        "topology_keep_rate": float(np.mean([item["topology_match"] for item in per_case_metrics])),
        "folding_ratio_mean": float(np.mean([item["folding_ratio"] for item in per_case_metrics])),
        "folding_ratio_std": float(np.std([item["folding_ratio"] for item in per_case_metrics])),
        "num_cases": len(per_case_metrics),
        "projection_count": projection_count,
        "selected_threshold": float(threshold),
    }
    return summary_metrics, per_case_metrics


def _select_best_threshold(case_cache, params, threshold_candidates):
    best_summary = None
    best_cases = None
    best_key = None

    for threshold in threshold_candidates:
        summary_metrics, per_case_metrics = _metrics_for_threshold(case_cache, params, threshold)
        current_key = (
            summary_metrics["topology_keep_rate"],
            -summary_metrics["hd95_mean"],
            summary_metrics["dice_mean"],
        )
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_summary = summary_metrics
            best_cases = per_case_metrics

    return best_summary, best_cases


def _collect_visualization_payloads(case_cache, selected_cases):
    lookup = {case["sample_id"]: case for case in case_cache}
    payloads = {}
    for _, selected_case in selected_cases:
        case = lookup[selected_case["sample_id"]]
        payloads[selected_case["sample_id"]] = {
            "image": case["original_image"],
            "prior": case["posed_prior"] if case["posed_prior"] is not None else case["probability"],
            "probability": case["probability"],
            "label": case["original_label"],
        }
    return payloads


def evaluate_acdc_subset(
    model,
    dataloader,
    device,
    params,
    subset_name,
    threshold_candidates=None,
):
    max_batches = 0
    if subset_name == "validation":
        max_batches = int(getattr(params, "max_validation_batches", 0))
    elif subset_name == "test":
        max_batches = int(getattr(params, "max_test_batches", 0))

    case_cache = _build_case_cache(model, dataloader, device, params, max_batches=max_batches)
    if not case_cache:
        raise RuntimeError(f"未执行任何 {subset_name} 样本评估。")

    if threshold_candidates is None:
        threshold_candidates = [float(params.threshold)]
    summary_metrics, per_case_metrics = _select_best_threshold(case_cache, params, threshold_candidates)
    return summary_metrics, per_case_metrics, case_cache


def evaluate_acdc_model(model, dataloader, device, params, run_output_dir, project_root, command):
    """在测试集上执行完整 ACDC 评估并生成报告。"""

    metrics_dir = _ensure_dir(os.path.join(run_output_dir, "metrics"))
    report_dir = _ensure_dir(os.path.join(run_output_dir, "report"))
    visualization_dir = _ensure_dir(os.path.join(run_output_dir, "visualizations"))
    docs_asset_dir = _ensure_dir(os.path.join(project_root, "docs", "experiments", "assets", "acdc_batch200"))
    docs_report_path = os.path.join(project_root, "docs", "experiments", "acdc_batch200_report.md")
    run_report_path = os.path.join(report_dir, "acdc_batch200_report.md")

    selected_threshold = getattr(params, "selected_validation_threshold", float(params.threshold))
    summary_metrics, per_case_metrics, case_cache = evaluate_acdc_subset(
        model=model,
        dataloader=dataloader,
        device=device,
        params=params,
        subset_name="test",
        threshold_candidates=[selected_threshold],
    )

    visualization_entries = []
    if getattr(params, "plot_predictions", True):
        selected_cases = _select_visualization_cases(per_case_metrics, params.seed)
        payloads = _collect_visualization_payloads(case_cache, selected_cases)
        for label, case in selected_cases:
            payload = payloads[case["sample_id"]]
            prediction = (payload["probability"] >= summary_metrics["selected_threshold"]).astype(np.float32)
            if params.dataset.topology_projection and calculate_betti_numbers(prediction) != (1, 1):
                prediction = project_probability_to_annulus(
                    probability=payload["probability"],
                    pose_params=next(item for item in case_cache if item["sample_id"] == case["sample_id"])["pose_params"],
                    base_outer_radius=float(params.dataset.ps_meas[0]),
                    base_inner_radius=max(float(params.dataset.ps_meas[0] - params.dataset.ps_meas[1]), 1.0),
                    margin=float(params.dataset.topology_margin),
                    threshold=float(params.dataset.projection_threshold),
                    num_angles=int(params.dataset.projection_angles),
                ).astype(np.float32)

            filename = f"{_label_slug(label)}_{case['sample_id']}.png"
            run_image_path = os.path.join(visualization_dir, filename)
            docs_image_path = os.path.join(docs_asset_dir, filename)
            _render_visualization(payload["image"], payload["prior"], prediction, payload["label"], label, run_image_path)
            _render_visualization(payload["image"], payload["prior"], prediction, payload["label"], label, docs_image_path)
            visualization_entries.append(
                {
                    "label": label,
                    "sample_id": case["sample_id"],
                    "dice": case["dice"],
                    "hd": case["hd"],
                    "hd95": case["hd95"],
                    "assd": case["assd"],
                    "gt_betti": case["gt_betti"],
                    "pred_betti": case["pred_betti"],
                    "folding_ratio": case["folding_ratio"],
                    "report_image_path": f"assets/acdc_batch200/{filename}",
                }
            )

    per_case_path = os.path.join(metrics_dir, "per_slice_metrics.json")
    summary_path = os.path.join(metrics_dir, "summary_metrics.json")
    with open(per_case_path, "w", encoding="utf-8") as handle:
        json.dump(per_case_metrics, handle, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_metrics, handle, ensure_ascii=False, indent=2)

    context = {
        "git_branch": _git_output(project_root, "git", "rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": _git_output(project_root, "git", "rev-parse", "HEAD"),
        "run_name": params.run_name,
        "raw_data_path": params.dataset.raw_data_path,
        "processed_data_path": params.dataset.processed_data_path,
        "run_output_dir": run_output_dir,
        "command": command,
        "device": str(device),
        "epochs": params.epoch,
        "batch_size": params.batch,
        "threshold": summary_metrics["selected_threshold"],
        "split_counts": getattr(params, "acdc_split_counts", {}),
        "preprocess_summary": getattr(params, "acdc_manifest_summary", {}),
    }
    write_acdc_report(docs_report_path, context, summary_metrics, visualization_entries)
    write_acdc_report(run_report_path, context, summary_metrics, visualization_entries)
    return summary_metrics
