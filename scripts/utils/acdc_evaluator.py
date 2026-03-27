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
    compose_backward_flows,
    compute_hd95_and_assd,
    dice_coefficient,
    folding_ratio,
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


def evaluate_acdc_model(model, dataloader, device, params, run_output_dir, project_root, command):
    """在测试集上执行完整 ACDC 评估并生成报告。"""

    metrics_dir = _ensure_dir(os.path.join(run_output_dir, "metrics"))
    report_dir = _ensure_dir(os.path.join(run_output_dir, "report"))
    visualization_dir = _ensure_dir(os.path.join(run_output_dir, "visualizations"))
    docs_asset_dir = _ensure_dir(os.path.join(project_root, "docs", "experiments", "assets", "acdc_batch200"))
    docs_report_path = os.path.join(project_root, "docs", "experiments", "acdc_batch200_report.md")
    run_report_path = os.path.join(report_dir, "acdc_batch200_report.md")

    per_case_metrics = []
    visualization_payloads = {}
    model.eval()

    with torch.no_grad():
        for case_index, (image, prior_shape, _, metadata) in enumerate(dataloader):
            if getattr(params, "max_test_batches", 0) and case_index >= params.max_test_batches:
                break
            image_batch = image.unsqueeze(0).to(device)
            prior_batch = prior_shape.unsqueeze(0).to(device)
            outputs = model(image_batch, prior_batch)

            prediction = (outputs[0] > params.threshold).float()[0, 0].detach().cpu().numpy().astype(np.float32)
            if len(outputs) == 3:
                final_flow = compose_backward_flows(outputs[1], outputs[2])[0].detach().cpu().numpy()
            else:
                final_flow = outputs[1][0].detach().cpu().numpy()

            original_shape = metadata["original_shape"]
            original_image = metadata["original_image"].astype(np.float32)
            original_label = metadata["original_label"].astype(np.float32)
            restored_prediction = _resize_2d_array(prediction, original_shape, is_label=True)
            restored_prior = _resize_2d_array(
                prior_shape.squeeze(0).detach().cpu().numpy(),
                original_shape,
                is_label=True,
            )

            sample_id = (
                f"{metadata['patient_id']}_{metadata['frame_id']}_slice{int(metadata['slice_index']):03d}"
            )
            dice = dice_coefficient(restored_prediction, original_label)
            hd95, assd = compute_hd95_and_assd(restored_prediction, original_label, metadata["spacing"])
            pred_betti = calculate_betti_numbers(restored_prediction)
            topology_match = int(tuple(metadata["betti"]) == tuple(pred_betti))
            current_folding_ratio = folding_ratio(final_flow)
            visualization_payloads[sample_id] = {
                "image": original_image,
                "prior": restored_prior,
                "prediction": restored_prediction,
                "label": original_label,
            }

            per_case_metrics.append(
                {
                    "sample_id": sample_id,
                    "relative_path": metadata["relative_path"],
                    "patient_id": metadata["patient_id"],
                    "frame_id": metadata["frame_id"],
                    "slice_index": int(metadata["slice_index"]),
                    "source_subset": metadata["source_subset"],
                    "dice": float(dice),
                    "hd95": float(hd95),
                    "assd": float(assd),
                    "gt_betti": list(metadata["betti"]),
                    "pred_betti": list(pred_betti),
                    "topology_match": topology_match,
                    "folding_ratio": float(current_folding_ratio),
                }
            )

    if not per_case_metrics:
        raise RuntimeError("未执行任何 ACDC 测试样本评估。")

    summary_metrics = {
        "dice_mean": float(np.mean([item["dice"] for item in per_case_metrics])),
        "dice_std": float(np.std([item["dice"] for item in per_case_metrics])),
        "hd95_mean": float(np.mean([item["hd95"] for item in per_case_metrics])),
        "hd95_std": float(np.std([item["hd95"] for item in per_case_metrics])),
        "assd_mean": float(np.mean([item["assd"] for item in per_case_metrics])),
        "assd_std": float(np.std([item["assd"] for item in per_case_metrics])),
        "topology_keep_rate": float(np.mean([item["topology_match"] for item in per_case_metrics])),
        "folding_ratio_mean": float(np.mean([item["folding_ratio"] for item in per_case_metrics])),
        "folding_ratio_std": float(np.std([item["folding_ratio"] for item in per_case_metrics])),
        "num_cases": len(per_case_metrics),
    }

    visualization_entries = []
    if getattr(params, "plot_predictions", True):
        for label, case in _select_visualization_cases(per_case_metrics, params.seed):
            payload = visualization_payloads[case["sample_id"]]
            image = np.asarray(payload["image"], dtype=np.float32)
            prior = np.asarray(payload["prior"], dtype=np.float32)
            prediction = np.asarray(payload["prediction"], dtype=np.float32)
            gt_label = np.asarray(payload["label"], dtype=np.float32)

            filename = f"{_label_slug(label)}_{case['sample_id']}.png"
            run_image_path = os.path.join(visualization_dir, filename)
            docs_image_path = os.path.join(docs_asset_dir, filename)
            _render_visualization(image, prior, prediction, gt_label, label, run_image_path)
            _render_visualization(image, prior, prediction, gt_label, label, docs_image_path)

            visualization_entries.append(
                {
                    "label": label,
                    "sample_id": case["sample_id"],
                    "dice": case["dice"],
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
        "split_counts": getattr(params, "acdc_split_counts", {}),
        "preprocess_summary": getattr(params, "acdc_manifest_summary", {}),
    }
    write_acdc_report(docs_report_path, context, summary_metrics, visualization_entries)
    write_acdc_report(run_report_path, context, summary_metrics, visualization_entries)
    return summary_metrics
