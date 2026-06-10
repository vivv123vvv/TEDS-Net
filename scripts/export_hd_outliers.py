import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_results import load_model, set_random_seed
from utils.acdc_benchmark import (
    DEFAULT_DATA_DIR,
    align_mask_tensors,
    assd_distance,
    dice_score,
    ensure_dir,
    hd95_distance,
    hausdorff_distance,
    iou_score,
    jacobian_negative_ratio,
    precision_score,
    recall_score,
    resolve_device,
    sync_cuda,
    topology_signature,
    write_csv,
)


def optional_float(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export high-HD samples into CSV/Markdown reports and PNG visualizations."
    )
    parser.add_argument(
        "--eval-csv",
        default="reports/benchmarks/acdc-compare-r2net-s42-e200/eval_per_sample.csv",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/acdc-compare-r2net-s42-e200/best_teds_net.pth",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--output-dir",
        default="reports/benchmarks/acdc-compare-r2net-s42-e200/hd_outliers",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--hd-percentile", type=float, default=95.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-hd", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_eval_rows(eval_csv_path):
    rows = []
    with Path(eval_csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "benchmark_case": row["benchmark_case"],
                    "sample_id": row["sample_id"],
                    "case_id": row["case_id"],
                    "original_forward_ms": float(row["forward_ms"]),
                    "original_dice": float(row["dice"]),
                    "original_iou": optional_float(row, "iou"),
                    "original_hd": float(row["hd"]),
                    "original_hd95": optional_float(row, "hd95"),
                    "original_assd": optional_float(row, "assd"),
                    "original_precision": optional_float(row, "precision"),
                    "original_recall": optional_float(row, "recall"),
                    "original_correct_topology": float(row["correct_topology"]),
                    "original_pred_components": int(float(row["pred_components"])),
                    "original_pred_holes": int(float(row["pred_holes"])),
                    "original_target_components": int(float(row["target_components"])),
                    "original_target_holes": int(float(row["target_holes"])),
                    "original_jacobian_neg_ratio": float(row["jacobian_neg_ratio"]),
                }
            )
    return rows


def select_hd_outliers(rows, hd_percentile=95.0, top_k=None, min_hd=None):
    if not rows:
        raise RuntimeError("No eval rows were loaded.")

    all_hd = np.array([row["original_hd"] for row in rows], dtype=float)
    percentile_threshold = float(np.percentile(all_hd, hd_percentile))
    effective_threshold = percentile_threshold if min_hd is None else max(percentile_threshold, float(min_hd))

    selected = [row for row in rows if row["original_hd"] >= effective_threshold]
    selected.sort(key=lambda row: row["original_hd"], reverse=True)

    if top_k is not None:
        selected = selected[: int(top_k)]

    for rank, row in enumerate(selected, start=1):
        row["hd_rank"] = rank
        row["hd_percentile"] = float(hd_percentile)

    return selected, percentile_threshold, effective_threshold


def sample_path_map(data_dir):
    return {path.stem: path for path in Path(data_dir).glob("*.npz")}


def load_sample(sample_path):
    payload = np.load(sample_path)
    image = payload["image"].astype(np.float32)
    prior = payload["prior"].astype(np.float32)
    label = payload["label"]
    target = (label == 2).astype(np.float32)
    return image, prior, target


def overlay_rgb(image, pred_mask, target_mask):
    image = np.asarray(image, dtype=np.float32)
    pred_mask = np.asarray(pred_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)

    base = np.stack([image, image, image], axis=-1)
    base = np.clip(base, 0.0, 1.0)

    fp = pred_mask & ~target_mask
    fn = ~pred_mask & target_mask
    tp = pred_mask & target_mask

    overlay = base.copy()
    overlay[tp] = 0.55 * overlay[tp] + 0.45 * np.array([0.0, 0.85, 0.0], dtype=np.float32)
    overlay[fp] = 0.45 * overlay[fp] + 0.55 * np.array([1.0, 0.2, 0.0], dtype=np.float32)
    overlay[fn] = 0.45 * overlay[fn] + 0.55 * np.array([0.15, 0.45, 1.0], dtype=np.float32)
    return np.clip(overlay, 0.0, 1.0)


def save_sample_figure(output_path, row, image, prior, pred_prob, pred_mask, target_mask):
    output_path = Path(output_path)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.ravel()

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Image")

    axes[1].imshow(prior, cmap="gray")
    axes[1].set_title("Prior")

    prob_im = axes[2].imshow(pred_prob, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("Prediction Prob")
    fig.colorbar(prob_im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(target_mask, cmap="gray", vmin=0.0, vmax=1.0)
    axes[3].set_title(
        f"GT Mask\ncomp={row['target_components']} hole={row['target_holes']}"
    )

    axes[4].imshow(pred_mask, cmap="gray", vmin=0.0, vmax=1.0)
    axes[4].set_title(
        f"Pred Mask\ncomp={row['pred_components']} hole={row['pred_holes']}"
    )

    axes[5].imshow(overlay_rgb(image, pred_mask, target_mask))
    axes[5].contour(target_mask, levels=[0.5], colors=["lime"], linewidths=1.2)
    axes[5].contour(pred_mask, levels=[0.5], colors=["red"], linewidths=1.0)
    axes[5].set_title("Overlay\nGreen=TP Orange=FP Blue=FN")

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(
        (
            f"rank={row['hd_rank']} | {row['sample_id']} | case={row['case_id']} | "
            f"dice={row['dice']:.4f} | iou={row['iou']:.4f} | "
            f"hd={row['hd']:.4f} | hd95={row['hd95']:.4f} | "
            f"topology={row['pred_components']}/{row['pred_holes']} vs "
            f"{row['target_components']}/{row['target_holes']}"
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_overview_figure(output_path, items, hd_threshold, hd_percentile):
    if not items:
        return

    columns = 4
    rows = math.ceil(len(items) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.axis("off")

    for axis, item in zip(axes, items):
        axis.imshow(overlay_rgb(item["image"], item["pred_mask"], item["target_mask"]))
        axis.contour(item["target_mask"], levels=[0.5], colors=["lime"], linewidths=1.0)
        axis.contour(item["pred_mask"], levels=[0.5], colors=["red"], linewidths=0.9)
        axis.set_title(
            (
                f"#{item['row']['hd_rank']} {item['row']['sample_id']}\n"
                f"HD={item['row']['hd']:.2f} Dice={item['row']['dice']:.3f}"
            ),
            fontsize=9,
        )
        axis.axis("off")

    fig.suptitle(
        f"R2Net High-HD Samples (P{int(hd_percentile)} threshold = {hd_threshold:.4f})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_markdown(output_path, rows, overview_rel_path, hd_percentile, hd_threshold, mask_threshold):
    lines = [
        "# R2Net High-HD Samples",
        "",
        f"- Selection rule: `HD >= P{hd_percentile:.1f}`",
        f"- Effective HD threshold: `{hd_threshold:.6f}`",
        f"- Sample count: `{len(rows)}`",
        f"- Mask threshold: `{mask_threshold:.2f}`",
        f"- Overview: `![overview]({overview_rel_path})`",
        "",
        "| rank | sample_id | case_id | HD | Dice | IoU | HD95 | ASSD | Precision | Recall | topology | figure |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        topology = f"{row['pred_components']} / {row['pred_holes']} vs {row['target_components']} / {row['target_holes']}"
        lines.append(
            "| {rank} | {sample_id} | {case_id} | {hd:.6f} | {dice:.6f} | {iou:.6f} | {hd95:.6f} | {assd:.6f} | {precision:.6f} | {recall:.6f} | {topology} | [png]({figure_rel_path}) |".format(
                rank=row["hd_rank"],
                sample_id=row["sample_id"],
                case_id=row["case_id"],
                hd=row["hd"],
                dice=row["dice"],
                iou=row["iou"],
                hd95=row["hd95"],
                assd=row["assd"],
                precision=row["precision"],
                recall=row["recall"],
                topology=topology,
                figure_rel_path=row["figure_rel_path"],
            )
        )

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_hd_outliers(args):
    eval_csv = Path(args.eval_csv)
    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    image_dir = ensure_dir(output_dir / "images")
    overview_path = output_dir / "hd_outliers_overview.png"
    csv_path = output_dir / "hd_outliers.csv"
    md_path = output_dir / "hd_outliers.md"

    eval_rows = load_eval_rows(eval_csv)
    selected_rows, percentile_threshold, effective_threshold = select_hd_outliers(
        eval_rows,
        hd_percentile=args.hd_percentile,
        top_k=args.top_k,
        min_hd=args.min_hd,
    )
    if not selected_rows:
        raise RuntimeError("No HD outlier rows matched the selection rule.")

    sample_to_path = sample_path_map(data_dir)
    missing = [row["sample_id"] for row in selected_rows if row["sample_id"] not in sample_to_path]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} sample files, first few: {missing[:5]}")

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    model, _, params = load_model(checkpoint_path, device, seed=args.seed)
    set_random_seed(params.seed)

    exported_rows = []
    overview_items = []

    with torch.no_grad():
        for row in selected_rows:
            sample_id = row["sample_id"]
            image_np, prior_np, target_np = load_sample(sample_to_path[sample_id])

            image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).to(device)
            prior_tensor = torch.from_numpy(prior_np).unsqueeze(0).unsqueeze(0).to(device)
            target_tensor = torch.from_numpy(target_np).unsqueeze(0).unsqueeze(0)

            sync_cuda(device)
            infer_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            infer_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if infer_start is not None:
                infer_start.record()

            outputs = model(image_tensor, prior_tensor)

            if infer_end is not None:
                infer_end.record()
                sync_cuda(device)
                forward_ms = float(infer_start.elapsed_time(infer_end))
            else:
                forward_ms = None

            if len(outputs) == 2:
                pred_prob_tensor, flow_upsamp = outputs
            elif len(outputs) == 3:
                pred_prob_tensor, _, flow_upsamp = outputs
            else:
                raise ValueError(f"Unexpected model outputs length: {len(outputs)}")

            pred_prob = pred_prob_tensor.detach().cpu()
            pred_mask, target_mask = align_mask_tensors(
                (pred_prob > args.mask_threshold).float(),
                target_tensor,
            )

            pred_prob_np = np.squeeze(pred_prob.numpy())
            pred_mask_np = np.squeeze(pred_mask.numpy()).astype(np.float32)
            target_mask_np = np.squeeze(target_mask.numpy()).astype(np.float32)
            pred_components, pred_holes = topology_signature(pred_mask)
            target_components, target_holes = topology_signature(target_mask)

            figure_name = f"{row['hd_rank']:02d}_{sample_id}.png"
            figure_path = image_dir / figure_name

            exported_row = {
                **row,
                "mask_threshold": float(args.mask_threshold),
                "percentile_threshold": percentile_threshold,
                "effective_threshold": effective_threshold,
                "forward_ms_rerun": forward_ms,
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
                "jacobian_neg_ratio": jacobian_negative_ratio(flow_upsamp.detach().cpu()),
                "figure_path": str(figure_path),
                "figure_rel_path": str(Path("images") / figure_name).replace("\\", "/"),
            }

            save_sample_figure(
                figure_path,
                exported_row,
                image_np,
                prior_np,
                pred_prob_np,
                pred_mask_np,
                target_mask_np,
            )

            exported_rows.append(exported_row)
            overview_items.append(
                {
                    "row": exported_row,
                    "image": image_np,
                    "pred_mask": pred_mask_np,
                    "target_mask": target_mask_np,
                }
            )

    write_csv(
        csv_path,
        [
            "hd_rank",
            "benchmark_case",
            "sample_id",
            "case_id",
            "hd_percentile",
            "percentile_threshold",
            "effective_threshold",
            "mask_threshold",
            "original_forward_ms",
            "forward_ms_rerun",
            "original_dice",
            "dice",
            "original_iou",
            "iou",
            "original_hd",
            "hd",
            "original_hd95",
            "hd95",
            "original_assd",
            "assd",
            "original_precision",
            "precision",
            "original_recall",
            "recall",
            "original_correct_topology",
            "correct_topology",
            "original_pred_components",
            "original_pred_holes",
            "pred_components",
            "pred_holes",
            "original_target_components",
            "original_target_holes",
            "target_components",
            "target_holes",
            "original_jacobian_neg_ratio",
            "jacobian_neg_ratio",
            "figure_path",
            "figure_rel_path",
        ],
        exported_rows,
    )
    save_overview_figure(overview_path, overview_items, effective_threshold, args.hd_percentile)
    write_markdown(
        md_path,
        exported_rows,
        overview_path.name,
        args.hd_percentile,
        effective_threshold,
        args.mask_threshold,
    )

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote Markdown: {md_path}")
    print(f"Wrote overview: {overview_path}")
    print(f"Wrote {len(exported_rows)} sample figure(s) to {image_dir}")


if __name__ == "__main__":
    export_hd_outliers(parse_args())
