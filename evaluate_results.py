import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.acdc_npz import ACDCNpzDataset
from network.TEDS_Net import TEDS_Net
from parameters.acdc_parameters import Parameters, normalize_integrator_name
from utils.acdc_benchmark import (
    DEFAULT_BEST_CHECKPOINT_NAME,
    DEFAULT_CHECKPOINT_ROOT,
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
from utils.dataset_registry import (
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_REGISTRY,
    apply_dataset_spec_to_params,
    resolve_dataset_spec,
)
from utils.experiment_logging import format_command, write_experiment_log


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a TEDS-Net checkpoint and save local benchmark reports.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-registry", default=str(DEFAULT_DATASET_REGISTRY))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-log-dir", default=None)
    parser.add_argument("--experiment-purpose", default=None)
    parser.add_argument("--experiment-notes", default=None)
    parser.add_argument("--integrator", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint_path, checkpoint_root=DEFAULT_CHECKPOINT_ROOT):
    checkpoint_root = Path(checkpoint_root)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else checkpoint_root / DEFAULT_BEST_CHECKPOINT_NAME
    if checkpoint_path.exists():
        return checkpoint_path

    default_checkpoint = checkpoint_root / DEFAULT_BEST_CHECKPOINT_NAME
    if checkpoint_path == default_checkpoint:
        candidates = sorted(
            checkpoint_root.glob(f"*/{DEFAULT_BEST_CHECKPOINT_NAME}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def infer_integrator_from_state_dict(state_dict):
    state_keys = state_dict.keys() if hasattr(state_dict, "keys") else []
    if any("r2net_integrator" in key for key in state_keys):
        return "r2net"
    return "original_teds"


def load_model(checkpoint_path, device, integrator=None, seed=None, dataset_spec=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    if isinstance(checkpoint, dict) and "params" in checkpoint:
        params = Parameters.from_dict(checkpoint["params"])
        network_payload = checkpoint["params"].get("network", {})
        if "integrator" not in network_payload:
            params.network.integrator = infer_integrator_from_state_dict(state_dict)
    else:
        params = apply_dataset_spec_to_params(Parameters(), dataset_spec) if dataset_spec else Parameters()
        params.network.integrator = infer_integrator_from_state_dict(state_dict)

    if integrator is not None:
        params.network.integrator = normalize_integrator_name(integrator)
    if seed is not None:
        params.seed = int(seed)

    model = TEDS_Net(params).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, checkpoint, params


def make_eval_loader(data_dir, split_manifest, split, dataset_spec):
    manifest = load_split_manifest(split_manifest, data_dir)
    dataset = ACDCNpzDataset(
        data_dir,
        file_list=get_split_filenames(manifest, split),
        include_metadata=True,
        **dataset_spec.loader_kwargs(),
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
    data_dir=None,
    split_manifest=None,
    split="test",
    run_name=None,
    output_dir=None,
    checkpoint_root=None,
    integrator=None,
    seed=None,
    warmup_batches=1,
    max_samples=None,
    device=None,
    dataset_id=DEFAULT_DATASET_ID,
    dataset_registry_path=DEFAULT_DATASET_REGISTRY,
    dataset_spec=None,
):
    dataset_spec = dataset_spec or resolve_dataset_spec(dataset_id, dataset_registry_path)
    data_dir = Path(data_dir) if data_dir else dataset_spec.data_dir
    split_manifest = Path(split_manifest) if split_manifest else dataset_spec.split_manifest
    output_dir = Path(output_dir) if output_dir else dataset_spec.reports_dir
    checkpoint_root = Path(checkpoint_root) if checkpoint_root else dataset_spec.checkpoint_root
    checkpoint_path = resolve_checkpoint_path(checkpoint_path, checkpoint_root)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    if not split_manifest.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest}")

    device = resolve_device(device)
    print(f"Evaluating checkpoint {checkpoint_path} on device {device}")

    model, checkpoint_payload, params = load_model(
        checkpoint_path,
        device,
        integrator=integrator,
        seed=seed,
        dataset_spec=dataset_spec,
    )
    set_random_seed(params.seed)
    print(f"Integrator: {params.network.integrator} | Seed: {params.seed}")

    parameter_count = model_parameter_count(model)
    benchmark_case = infer_run_name(run_name, checkpoint_path, checkpoint_payload)
    run_dir = make_run_dir(output_dir, benchmark_case)

    manifest, warmup_loader = make_eval_loader(data_dir, split_manifest, split, dataset_spec)
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
    _, eval_loader = make_eval_loader(data_dir, split_manifest, split, dataset_spec)
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
            "dataset_id": dataset_spec.dataset_id,
            "dataset_name": dataset_spec.display_name,
            "dataset_registry": str(dataset_registry_path),
            "integrator": params.network.integrator,
            "seed": params.seed,
            "device": str(device),
            "data_dir": str(data_dir),
            "split_manifest": str(split_manifest),
            "split_id": split_manifest.name,
            "train_count": manifest["counts"].get("train", 0),
            "val_count": manifest["counts"].get("val", 0),
            "test_count": manifest["counts"].get("test", 0),
            "dataset_spec": dataset_spec.to_dict(),
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
    dataset_spec = resolve_dataset_spec(args.dataset, args.dataset_registry)
    experiment_log_dir = Path(args.experiment_log_dir) if args.experiment_log_dir else dataset_spec.experiment_log_dir
    research_purpose = args.experiment_purpose or (
        f"评估 TEDS-Net checkpoint 在 {dataset_spec.display_name} 数据集 {args.split} split 上的表现。"
    )
    started_at = datetime.now().isoformat(timespec="seconds")
    command = format_command(sys.argv)

    try:
        result = run_evaluation(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            split_manifest=args.split_manifest,
            split=args.split,
            run_name=args.run_name,
            output_dir=args.output_dir,
            checkpoint_root=args.checkpoint_root,
            integrator=args.integrator,
            seed=args.seed,
            warmup_batches=args.warmup_batches,
            max_samples=args.max_samples,
            device=args.device,
            dataset_id=dataset_spec.dataset_id,
            dataset_registry_path=args.dataset_registry,
            dataset_spec=dataset_spec,
        )
        summary = result["summary"]
        log_paths = write_experiment_log(
            experiment_log_dir,
            summary["run_name"],
            dataset_spec,
            research_purpose,
            status="completed",
            eval_summary=summary,
            artifact_paths={
                "run_dir": str(result["run_dir"]),
                "eval_summary": str(result["run_dir"] / "eval_summary.json"),
                "eval_per_sample": str(result["run_dir"] / "eval_per_sample.csv"),
                "eval_per_case": str(result["run_dir"] / "eval_per_case.csv"),
            },
            command=command,
            notes=args.experiment_notes,
            started_at=started_at,
        )
        print(f"Wrote experiment log to {log_paths['md_path']}")
    except Exception as exc:
        fallback_run_name = args.run_name or f"eval-{dataset_spec.dataset_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        log_paths = write_experiment_log(
            experiment_log_dir,
            fallback_run_name,
            dataset_spec,
            research_purpose,
            status="failed",
            command=command,
            notes=args.experiment_notes,
            error=repr(exc),
            started_at=started_at,
        )
        print(f"Wrote failed experiment log to {log_paths['md_path']}")
        raise
