import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.acdc_npz import ACDCNpzDataset
from network.TEDS_Net import TEDS_Net
from parameters.acdc_parameters import Parameters
from utils.acdc_benchmark import (
    DEFAULT_BEST_CHECKPOINT_NAME,
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATA_DIR,
    DEFAULT_REPORTS_DIR,
    DEFAULT_SPLIT_MANIFEST,
    aggregate_metric_rows,
    align_mask_tensors,
    build_eval_summary,
    dice_score,
    get_split_filenames,
    hausdorff_distance,
    jacobian_negative_ratio,
    load_split_manifest,
    make_run_dir,
    model_parameter_count,
    peak_gpu_memory_mb,
    reset_peak_memory,
    resolve_device,
    sync_cuda,
    topology_signature,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a TEDS-Net checkpoint and save local benchmark reports.")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_ROOT / DEFAULT_BEST_CHECKPOINT_NAME),
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT_MANIFEST))
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        return checkpoint_path

    default_checkpoint = DEFAULT_CHECKPOINT_ROOT / DEFAULT_BEST_CHECKPOINT_NAME
    if checkpoint_path == default_checkpoint:
        candidates = sorted(
            DEFAULT_CHECKPOINT_ROOT.glob(f"*/{DEFAULT_BEST_CHECKPOINT_NAME}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "params" in checkpoint:
        params = Parameters.from_dict(checkpoint["params"])
    else:
        params = Parameters()

    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model = TEDS_Net(params).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, checkpoint, params


def make_eval_loader(data_dir, split_manifest, split):
    manifest = load_split_manifest(split_manifest, data_dir)
    dataset = ACDCNpzDataset(
        data_dir,
        file_list=get_split_filenames(manifest, split),
        include_metadata=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    return manifest, loader


def infer_run_name(run_name, checkpoint_path, checkpoint_payload):
    if run_name:
        return run_name
    if isinstance(checkpoint_payload, dict) and checkpoint_payload.get("run_name"):
        return checkpoint_payload["run_name"]
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.parent.name != "checkpoints":
        return checkpoint_path.parent.name
    return checkpoint_path.stem


def run_evaluation(
    checkpoint_path,
    data_dir=DEFAULT_DATA_DIR,
    split_manifest=DEFAULT_SPLIT_MANIFEST,
    split="test",
    run_name=None,
    output_dir=DEFAULT_REPORTS_DIR,
    warmup_batches=1,
    max_samples=None,
    device=None,
):
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    data_dir = Path(data_dir)
    split_manifest = Path(split_manifest)
    output_dir = Path(output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    if not split_manifest.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest}")

    device = resolve_device(device)
    print(f"Evaluating checkpoint {checkpoint_path} on device {device}")

    model, checkpoint_payload, _ = load_model(checkpoint_path, device)
    parameter_count = model_parameter_count(model)
    benchmark_case = infer_run_name(run_name, checkpoint_path, checkpoint_payload)
    run_dir = make_run_dir(output_dir, benchmark_case)

    _, warmup_loader = make_eval_loader(data_dir, split_manifest, split)
    if warmup_batches and warmup_batches > 0:
        print(f"Running {warmup_batches} warmup batch(es)...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(warmup_loader):
                if batch_idx >= warmup_batches:
                    break
                images, priors, _, _, _ = batch
                images = images.float().to(device)
                priors = priors.float().to(device)
                _ = model(images, priors)
        sync_cuda(device)

    reset_peak_memory(device)
    _, eval_loader = make_eval_loader(data_dir, split_manifest, split)
    sample_rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc=f"Eval {benchmark_case}")):
            if max_samples is not None and batch_idx >= max_samples:
                break

            images, priors, gt_masks, sample_ids, case_ids = batch
            images = images.float().to(device)
            priors = priors.float().to(device)
            gt_masks = gt_masks.float().to(device)

            sync_cuda(device)
            infer_start = time.perf_counter()
            outputs = model(images, priors)
            sync_cuda(device)
            forward_ms = (time.perf_counter() - infer_start) * 1000.0

            if len(outputs) == 2:
                pred_warped, flow_upsamp = outputs
            elif len(outputs) == 3:
                pred_warped, _, flow_upsamp = outputs
            else:
                raise ValueError(f"Unexpected number of outputs from model: {len(outputs)}")

            pred_mask = (pred_warped > 0.5).float()
            pred_mask, gt_masks = align_mask_tensors(pred_mask, gt_masks)
            pred_components, pred_holes = topology_signature(pred_mask)
            target_components, target_holes = topology_signature(gt_masks)
            correct_topology = float(
                (pred_components, pred_holes) == (target_components, target_holes)
            )

            sample_rows.append(
                {
                    "benchmark_case": benchmark_case,
                    "sample_id": sample_ids[0],
                    "case_id": case_ids[0],
                    "forward_ms": float(forward_ms),
                    "dice": dice_score(pred_mask, gt_masks),
                    "hd": hausdorff_distance(pred_mask, gt_masks),
                    "correct_topology": correct_topology,
                    "pred_components": pred_components,
                    "pred_holes": pred_holes,
                    "target_components": target_components,
                    "target_holes": target_holes,
                    "jacobian_neg_ratio": jacobian_negative_ratio(flow_upsamp),
                }
            )

    peak_mem_mb = peak_gpu_memory_mb(device)
    per_case_rows = aggregate_metric_rows(sample_rows, "case_id")
    summary = build_eval_summary(
        sample_rows,
        peak_mem_mb,
        benchmark_case,
        split,
        checkpoint_path,
        parameter_count=parameter_count,
    )
    summary.update(
        {
            "run_name": benchmark_case,
            "data_dir": str(data_dir),
            "split_manifest": str(split_manifest),
        }
    )

    write_csv(
        run_dir / "eval_per_sample.csv",
        [
            "benchmark_case",
            "sample_id",
            "case_id",
            "forward_ms",
            "dice",
            "hd",
            "correct_topology",
            "pred_components",
            "pred_holes",
            "target_components",
            "target_holes",
            "jacobian_neg_ratio",
        ],
        sample_rows,
    )
    write_csv(
        run_dir / "eval_per_case.csv",
        [
            "case_id",
            "sample_count",
            "mean_forward_ms",
            "mean_dice",
            "mean_hd",
            "correct_topology_rate",
            "mean_jacobian_neg_ratio",
        ],
        per_case_rows,
    )
    write_json(run_dir / "eval_summary.json", summary)

    memory_text = f"{peak_mem_mb:.2f} MB" if peak_mem_mb is not None else "N/A (CPU)"
    print(
        "Evaluation Summary | Dice: {dice:.4f} | HD: {hd:.4f} px | "
        "Correct topology: {topology:.2%} | Jacobian < 0: {jac:.6f} | "
        "Mean Forward: {forward:.2f} ms | Peak GPU Mem: {memory}".format(
            dice=summary["mean_dice"] or 0.0,
            hd=summary["mean_hd"] or 0.0,
            topology=summary["correct_topology_rate"] or 0.0,
            jac=summary["mean_jacobian_neg_ratio"] or 0.0,
            forward=summary["mean_forward_ms"] or 0.0,
            memory=memory_text,
        )
    )
    print(f"Wrote evaluation artifacts to {run_dir}")

    return {
        "run_dir": run_dir,
        "sample_rows": sample_rows,
        "per_case_rows": per_case_rows,
        "summary": summary,
    }


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split_manifest=args.split_manifest,
        split=args.split,
        run_name=args.run_name,
        output_dir=args.output_dir,
        warmup_batches=args.warmup_batches,
        max_samples=args.max_samples,
        device=args.device,
    )
