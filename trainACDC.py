import argparse
import json
import random
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataloaders.acdc_npz import ACDCNpzDataset
from evaluate_results import run_evaluation
from network.TEDS_Net import TEDS_Net
from parameters.acdc_parameters import Parameters, normalize_integrator_name
from utils.acdc_benchmark import (
    DEFAULT_BEST_CHECKPOINT_NAME,
    discover_run_dirs,
    ensure_dir,
    get_split_filenames,
    load_split_manifest,
    make_run_dir,
    model_parameter_count,
    model_parameter_count_x1e5,
    peak_gpu_memory_mb,
    reset_peak_memory,
    resolve_device,
    sample_id_from_path,
    sync_cuda,
    write_comparison_artifacts,
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
from utils.losses import (
    boundary_distance_loss,
    dice_loss,
    flow_smoothness_loss,
    grad_loss,
    soft_cldice_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train TEDS-Net on an isolated dataset and emit benchmark reports.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-registry", default=str(DEFAULT_DATASET_REGISTRY))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--experiment-log-dir", default=None)
    parser.add_argument("--experiment-purpose", default=None)
    parser.add_argument("--experiment-notes", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--integrator", default=None)
    parser.add_argument("--r2net-blocks", type=int, default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--eval-warmup-batches", type=int, default=1)
    parser.add_argument("--eval-max-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--best-checkpoint-metric", choices=["dice", "total"], default="dice")
    parser.add_argument("--flow-smooth-weight", type=float, default=None)
    parser.add_argument("--flow-smooth-penalty", choices=["l1", "l2"], default=None)
    parser.add_argument("--boundary-distance-weight", type=float, default=None)
    parser.add_argument("--boundary-distance-max", type=float, default=None)
    parser.add_argument("--boundary-distance-min-weight", type=float, default=None)
    parser.add_argument("--cldice-weight", type=float, default=None)
    parser.add_argument("--cldice-iterations", type=int, default=None)
    parser.add_argument("--hard-sample-manifest", default=None)
    parser.add_argument("--hard-sample-weight", type=float, default=3.0)
    parser.add_argument("--hard-sample-sampling", choices=["none", "weighted_random"], default="none")
    parser.add_argument("--hard-sample-target-repeats", type=float, default=4.0)
    return parser.parse_args()


def default_run_name(dataset_id=DEFAULT_DATASET_ID):
    return f"teds-{dataset_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_hard_sample_sampler(dataset, hard_sample_id_set, target_repeats, seed):
    sample_ids = [sample_id_from_path(file_path) for file_path in dataset.file_list]
    matched_sample_ids = sorted(set(sample_ids) & hard_sample_id_set)
    if not matched_sample_ids:
        raise ValueError("Hard-sample oversampling requested, but no manifest ids matched the train split.")
    if target_repeats <= 0:
        raise ValueError("--hard-sample-target-repeats must be positive.")

    sample_count = len(sample_ids)
    hard_count = len(matched_sample_ids)
    max_repeats = sample_count / hard_count
    if target_repeats >= max_repeats:
        raise ValueError(
            "--hard-sample-target-repeats must be lower than {limit:.3f} for this split.".format(
                limit=max_repeats
            )
        )

    non_hard_count = sample_count - hard_count
    hard_weight = (
        1.0
        if non_hard_count == 0
        else (target_repeats * non_hard_count) / (sample_count - target_repeats * hard_count)
    )
    sample_weights = [
        hard_weight if sample_id in hard_sample_id_set else 1.0
        for sample_id in sample_ids
    ]
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=sample_count,
        replacement=True,
        generator=generator,
    )
    metadata = {
        "mode": "weighted_random",
        "target_repeats_per_hard_sample": float(target_repeats),
        "sampler_hard_weight": float(hard_weight),
        "num_samples_per_epoch": sample_count,
        "matched_train_sample_count": hard_count,
        "matched_train_sample_ids": matched_sample_ids,
        "expected_hard_draws_per_epoch": float(target_repeats * hard_count),
    }
    return sampler, metadata


def build_dataloaders(
    params,
    data_dir,
    split_manifest_path,
    dataset_spec,
    include_train_metadata=False,
    hard_sample_id_set=None,
    hard_sample_sampling="none",
    hard_sample_target_repeats=4.0,
    seed=42,
):
    manifest = load_split_manifest(split_manifest_path, data_dir)
    loader_kwargs = dataset_spec.loader_kwargs()
    train_dataset = ACDCNpzDataset(
        data_dir,
        file_list=get_split_filenames(manifest, "train"),
        include_metadata=include_train_metadata,
        **loader_kwargs,
    )
    val_dataset = ACDCNpzDataset(
        data_dir,
        file_list=get_split_filenames(manifest, "val"),
        **loader_kwargs,
    )
    train_sampler = None
    hard_sample_sampling_metadata = {"mode": "none"}
    if hard_sample_sampling == "weighted_random":
        train_sampler, hard_sample_sampling_metadata = build_hard_sample_sampler(
            train_dataset,
            hard_sample_id_set or set(),
            hard_sample_target_repeats,
            seed,
        )
    elif hard_sample_sampling != "none":
        raise ValueError(f"Unsupported hard-sample sampling mode: {hard_sample_sampling}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params.batch,
        shuffle=False,
        num_workers=0,
    )
    return manifest, train_loader, val_loader, hard_sample_sampling_metadata


def checkpoint_payload(
    model,
    params,
    run_name,
    epoch,
    best_val_dice,
    data_dir,
    split_manifest,
    dataset_spec,
    extra_metadata=None,
):
    payload = {
        "state_dict": model.state_dict(),
        "params": asdict(params),
        "run_name": run_name,
        "dataset_id": dataset_spec.dataset_id,
        "dataset_spec": dataset_spec.to_dict(),
        "epoch": epoch,
        "best_val_dice": float(best_val_dice),
        "data_dir": str(data_dir),
        "split_manifest": str(split_manifest),
    }
    if extra_metadata:
        payload.update(extra_metadata)
    return payload


def apply_loss_overrides(params, args):
    updates = {}
    override_map = {
        "flow_smooth_weight": args.flow_smooth_weight,
        "flow_smooth_penalty": args.flow_smooth_penalty,
        "boundary_distance_weight": args.boundary_distance_weight,
        "boundary_distance_max": args.boundary_distance_max,
        "boundary_distance_min_weight": args.boundary_distance_min_weight,
        "cldice_weight": args.cldice_weight,
        "cldice_iterations": args.cldice_iterations,
    }
    for key, value in override_map.items():
        if value is not None:
            updates[key] = value
    if updates:
        params.loss_params = replace(params.loss_params, **updates)
    return params


def build_auxiliary_losses(loss_params):
    return {
        "flow_smooth": flow_smoothness_loss(loss_params.flow_smooth_penalty)
        if loss_params.flow_smooth_weight > 0
        else None,
        "boundary_distance": boundary_distance_loss(
            max_distance=loss_params.boundary_distance_max,
            min_weight=loss_params.boundary_distance_min_weight,
        )
        if loss_params.boundary_distance_weight > 0
        else None,
        "cldice": soft_cldice_loss(iterations=loss_params.cldice_iterations)
        if loss_params.cldice_weight > 0
        else None,
    }


def load_initial_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    load_result = model.load_state_dict(state_dict, strict=True)
    return {
        "path": str(checkpoint_path),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def load_hard_sample_manifest(manifest_path):
    if not manifest_path:
        return None
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Hard-sample manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    sample_ids = payload.get("hard_sample_ids")
    if sample_ids is None:
        sample_ids = [item["sample_id"] for item in payload.get("hard_samples", [])]
    sample_ids = sorted({str(sample_id) for sample_id in sample_ids})
    return {
        "path": str(manifest_path),
        "payload": payload,
        "sample_ids": sample_ids,
        "sample_id_set": set(sample_ids),
    }


def dataset_sample_ids(dataset):
    return sorted(sample_id_from_path(file_path) for file_path in dataset.file_list)


def parse_model_outputs(outputs):
    if len(outputs) == 3:
        pred_seg, flow_bulk, flow_ft = outputs
        return pred_seg, [flow_bulk, flow_ft]
    if len(outputs) == 2:
        pred_seg, flow = outputs
        return pred_seg, [flow]
    raise ValueError(f"Unexpected number of outputs from model: {len(outputs)}")


def slice_model_outputs(outputs, sample_idx):
    return tuple(output[sample_idx : sample_idx + 1] for output in outputs)


def compute_loss_terms(outputs, label, calc_dice, calc_grad, aux_losses, loss_params):
    pred_seg, flows = parse_model_outputs(outputs)
    loss_dice = calc_dice.loss(label, pred_seg)
    loss_reg = sum(calc_grad.loss(None, flow) for flow in flows)
    total_loss = loss_dice + loss_params.grad_weight * loss_reg

    loss_flow_smooth = pred_seg.new_tensor(0.0)
    if aux_losses["flow_smooth"] is not None:
        loss_flow_smooth = sum(aux_losses["flow_smooth"].loss(None, flow) for flow in flows)
        total_loss = total_loss + loss_params.flow_smooth_weight * loss_flow_smooth

    loss_boundary_distance = pred_seg.new_tensor(0.0)
    if aux_losses["boundary_distance"] is not None:
        loss_boundary_distance = aux_losses["boundary_distance"].loss(label, pred_seg)
        total_loss = total_loss + loss_params.boundary_distance_weight * loss_boundary_distance

    loss_cldice = pred_seg.new_tensor(0.0)
    if aux_losses["cldice"] is not None:
        loss_cldice = aux_losses["cldice"].loss(label, pred_seg)
        total_loss = total_loss + loss_params.cldice_weight * loss_cldice

    return {
        "total": total_loss,
        "dice": loss_dice,
        "grad": loss_reg,
        "flow_smooth": loss_flow_smooth,
        "boundary_distance": loss_boundary_distance,
        "cldice": loss_cldice,
    }


def compute_weighted_loss_terms(outputs, label, sample_weights, calc_dice, calc_grad, aux_losses, loss_params):
    if sample_weights is None:
        return compute_loss_terms(outputs, label, calc_dice, calc_grad, aux_losses, loss_params)

    sample_weights = sample_weights.to(device=label.device, dtype=label.dtype).flatten()
    denominator = torch.clamp(sample_weights.sum(), min=1e-6)
    weighted_terms = None
    for sample_idx, sample_weight in enumerate(sample_weights):
        sample_terms = compute_loss_terms(
            slice_model_outputs(outputs, sample_idx),
            label[sample_idx : sample_idx + 1],
            calc_dice,
            calc_grad,
            aux_losses,
            loss_params,
        )
        if weighted_terms is None:
            weighted_terms = {key: value * sample_weight for key, value in sample_terms.items()}
        else:
            for key, value in sample_terms.items():
                weighted_terms[key] = weighted_terms[key] + value * sample_weight
    return {key: value / denominator for key, value in weighted_terms.items()}


def unpack_batch(batch):
    if len(batch) == 3:
        image, prior, label = batch
        return image, prior, label, None, None
    if len(batch) == 5:
        image, prior, label, sample_ids, case_ids = batch
        return image, prior, label, sample_ids, case_ids
    raise ValueError(f"Unexpected batch structure with {len(batch)} fields.")


def batch_sample_weights(sample_ids, hard_sample_id_set, hard_sample_weight, device):
    if sample_ids is None or not hard_sample_id_set:
        return None, 0
    weights = [
        float(hard_sample_weight) if str(sample_id) in hard_sample_id_set else 1.0
        for sample_id in sample_ids
    ]
    hard_count = sum(1 for weight in weights if weight != 1.0)
    if hard_count == 0:
        return None, 0
    return torch.tensor(weights, device=device, dtype=torch.float32), hard_count


def empty_loss_accumulator():
    return {
        "total": 0.0,
        "dice": 0.0,
        "grad": 0.0,
        "flow_smooth": 0.0,
        "boundary_distance": 0.0,
        "cldice": 0.0,
    }


def add_loss_terms(accumulator, terms):
    for key in accumulator:
        accumulator[key] += float(terms[key].detach().item())


def average_loss_terms(accumulator, count):
    return {key: value / count for key, value in accumulator.items()}


def train(args):
    dataset_spec = resolve_dataset_spec(args.dataset, args.dataset_registry)
    run_name = args.run_name or default_run_name(dataset_spec.dataset_id)
    data_dir = Path(args.data_dir) if args.data_dir else dataset_spec.data_dir
    split_manifest_path = Path(args.split_manifest) if args.split_manifest else dataset_spec.split_manifest
    output_dir = Path(args.output_dir) if args.output_dir else dataset_spec.reports_dir
    checkpoint_root = Path(args.checkpoint_root) if args.checkpoint_root else dataset_spec.checkpoint_root
    experiment_log_dir = Path(args.experiment_log_dir) if args.experiment_log_dir else dataset_spec.experiment_log_dir
    research_purpose = args.experiment_purpose or (
        f"训练并评估 TEDS-Net 在 {dataset_spec.display_name} 数据集上的分割与拓扑保持表现。"
    )
    command = format_command(sys.argv)
    started_at = datetime.now().isoformat(timespec="seconds")

    run_dir = None
    best_checkpoint_path = None
    train_summary = None
    eval_result = None
    comparison = None

    try:
        device = resolve_device(args.device)
        print(f"Using device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(device)}")
        else:
            print("CUDA not available, training on CPU.")

        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        if not split_manifest_path.exists():
            raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")

        params = apply_dataset_spec_to_params(Parameters(), dataset_spec)
        if args.epochs is not None:
            params.epoch = args.epochs
        if args.lr is not None:
            params.lr = float(args.lr)
        if args.integrator is not None:
            params.network.integrator = normalize_integrator_name(args.integrator)
        else:
            params.network.integrator = normalize_integrator_name(params.network.integrator)
        if args.r2net_blocks is not None:
            if args.r2net_blocks <= 0:
                raise ValueError("--r2net-blocks must be a positive integer.")
            params.network.r2net_blocks = int(args.r2net_blocks)
        if args.seed is not None:
            params.seed = int(args.seed)
        params = apply_loss_overrides(params, args)
        hard_sample_info = load_hard_sample_manifest(args.hard_sample_manifest)
        hard_sample_id_set = hard_sample_info["sample_id_set"] if hard_sample_info else set()
        if args.hard_sample_sampling != "none" and not hard_sample_info:
            raise ValueError("--hard-sample-sampling requires --hard-sample-manifest.")

        set_random_seed(params.seed)
        print(
            "Dataset: {name} ({dataset_id}) | data_dir={data_dir}".format(
                name=dataset_spec.display_name,
                dataset_id=dataset_spec.dataset_id,
                data_dir=data_dir,
            )
        )
        print(
            f"Integrator: {params.network.integrator} | "
            f"R2Net blocks: {params.network.r2net_blocks} | Seed: {params.seed}"
        )
        print(
            "Loss weights | grad={grad} flow_smooth={flow} boundary_distance={boundary} cldice={cldice}".format(
                grad=params.loss_params.grad_weight,
                flow=params.loss_params.flow_smooth_weight,
                boundary=params.loss_params.boundary_distance_weight,
                cldice=params.loss_params.cldice_weight,
            )
        )
        if hard_sample_info:
            print(
                "Hard-sample weighting | manifest={manifest} hard_ids={count} hard_weight={weight}".format(
                    manifest=hard_sample_info["path"],
                    count=len(hard_sample_info["sample_ids"]),
                    weight=args.hard_sample_weight,
                )
            )
            if args.hard_sample_sampling != "none":
                print(
                    "Hard-sample sampling | mode={mode} target_repeats={target:.2f}".format(
                        mode=args.hard_sample_sampling,
                        target=args.hard_sample_target_repeats,
                    )
                )

        manifest, train_loader, val_loader, hard_sample_sampling_metadata = build_dataloaders(
            params,
            data_dir,
            split_manifest_path,
            dataset_spec,
            include_train_metadata=bool(hard_sample_info),
            hard_sample_id_set=hard_sample_id_set,
            hard_sample_sampling=args.hard_sample_sampling,
            hard_sample_target_repeats=args.hard_sample_target_repeats,
            seed=params.seed,
        )
        train_sample_ids = dataset_sample_ids(train_loader.dataset)
        matched_hard_train_samples = sorted(set(train_sample_ids) & hard_sample_id_set)
        if hard_sample_info:
            print(
                "Hard-sample weighting matched {matched}/{train_count} train samples.".format(
                    matched=len(matched_hard_train_samples),
                    train_count=len(train_sample_ids),
                )
            )
            if hard_sample_sampling_metadata["mode"] != "none":
                print(
                    "Hard-sample sampler matched {matched} train samples | sampler hard weight={weight:.4f} | "
                    "expected hard draws/epoch={draws:.1f}".format(
                        matched=hard_sample_sampling_metadata["matched_train_sample_count"],
                        weight=hard_sample_sampling_metadata["sampler_hard_weight"],
                        draws=hard_sample_sampling_metadata["expected_hard_draws_per_epoch"],
                    )
                )
        print(
            "Loaded split manifest {manifest} | train={train_count} val={val_count} test={test_count}".format(
                manifest=split_manifest_path,
                train_count=manifest["counts"].get("train", 0),
                val_count=manifest["counts"].get("val", 0),
                test_count=manifest["counts"].get("test", 0),
            )
        )

        run_dir = make_run_dir(output_dir, run_name)
        checkpoint_root = ensure_dir(checkpoint_root)
        run_checkpoint_dir = ensure_dir(checkpoint_root / run_name)
        best_checkpoint_path = run_checkpoint_dir / DEFAULT_BEST_CHECKPOINT_NAME

        model = TEDS_Net(params).to(device)
        init_checkpoint_metadata = None
        if args.init_checkpoint:
            init_checkpoint_metadata = load_initial_checkpoint(model, args.init_checkpoint, device)
            print(f"Loaded initial checkpoint from {init_checkpoint_metadata['path']}")
        parameter_count = model_parameter_count(model)
        parameter_count_scaled = model_parameter_count_x1e5(model)
        print(
            "Model parameters: {count} ({scaled:.4f} x10^5)".format(
                count=parameter_count,
                scaled=parameter_count_scaled,
            )
        )
        optimizer = optim.Adam(model.parameters(), lr=params.lr)
        calc_dice = dice_loss()
        calc_grad = grad_loss(params)
        aux_losses = build_auxiliary_losses(params.loss_params)
        hard_sample_metadata = None
        if hard_sample_info:
            hard_sample_metadata = {
                "manifest_path": hard_sample_info["path"],
                "manifest_hard_sample_count": len(hard_sample_info["sample_ids"]),
                "hard_sample_weight": float(args.hard_sample_weight),
                "matched_train_sample_count": len(matched_hard_train_samples),
                "matched_train_sample_ids": matched_hard_train_samples,
                "sampling": hard_sample_sampling_metadata,
            }

        best_val_score = float("inf")
        best_val_loss = float("inf")
        epoch_rows = []

        print(f"Starting training run '{run_name}'...")
        for epoch_idx in range(params.epoch):
            model.train()
            reset_peak_memory(device)
            epoch_start = time.perf_counter()
            train_loss_sums = empty_loss_accumulator()
            processed_train_batches = 0
            processed_train_samples = 0
            processed_hard_train_samples = 0
            train_sample_weight_sum = 0.0

            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch_idx + 1}/{params.epoch}")
            for batch_idx, batch in enumerate(train_pbar):
                if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                    break

                image, prior, label, sample_ids, _ = unpack_batch(batch)
                image = image.to(device)
                prior = prior.to(device)
                label = label.to(device)
                sample_weights, batch_hard_count = batch_sample_weights(
                    sample_ids,
                    hard_sample_id_set,
                    args.hard_sample_weight,
                    device,
                )

                optimizer.zero_grad()
                outputs = model(image, prior)
                loss_terms = compute_weighted_loss_terms(
                    outputs,
                    label,
                    sample_weights,
                    calc_dice,
                    calc_grad,
                    aux_losses,
                    params.loss_params,
                )
                loss_terms["total"].backward()
                optimizer.step()

                add_loss_terms(train_loss_sums, loss_terms)
                processed_train_batches += 1
                processed_train_samples += int(image.shape[0])
                processed_hard_train_samples += int(batch_hard_count)
                if sample_weights is None:
                    train_sample_weight_sum += float(image.shape[0])
                else:
                    train_sample_weight_sum += float(sample_weights.detach().sum().item())
                train_pbar.set_postfix(
                    {
                        "loss": loss_terms["total"].item(),
                        "dice": 1.0 - loss_terms["dice"].item(),
                        "hard": batch_hard_count,
                    }
                )

            if processed_train_batches == 0:
                raise RuntimeError("No training batches were processed. Check --max-train-batches and split sizes.")

            model.eval()
            val_loss_sums = empty_loss_accumulator()
            processed_val_batches = 0
            with torch.no_grad():
                for batch_idx, (v_img, v_prior, v_lbl) in enumerate(val_loader):
                    if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                        break

                    v_img = v_img.to(device)
                    v_prior = v_prior.to(device)
                    v_lbl = v_lbl.to(device)
                    v_out = model(v_img, v_prior)
                    val_loss_terms = compute_loss_terms(
                        v_out,
                        v_lbl,
                        calc_dice,
                        calc_grad,
                        aux_losses,
                        params.loss_params,
                    )
                    add_loss_terms(val_loss_sums, val_loss_terms)
                    processed_val_batches += 1

            if processed_val_batches == 0:
                raise RuntimeError("No validation batches were processed. Check --max-val-batches and split sizes.")

            sync_cuda(device)
            epoch_sec = time.perf_counter() - epoch_start
            avg_train_terms = average_loss_terms(train_loss_sums, processed_train_batches)
            avg_val_terms = average_loss_terms(val_loss_sums, processed_val_batches)
            avg_train_loss = avg_train_terms["total"]
            avg_val_loss = avg_val_terms["dice"]
            avg_val_total_loss = avg_val_terms["total"]
            avg_val_dice = 1.0 - avg_val_loss
            avg_batch_ms = epoch_sec * 1000.0 / processed_train_batches
            peak_mem_mb = peak_gpu_memory_mb(device)
            mean_train_sample_weight = train_sample_weight_sum / max(processed_train_samples, 1)

            epoch_rows.append(
                {
                    "epoch": epoch_idx + 1,
                    "train_loss": avg_train_loss,
                    "train_dice_loss": avg_train_terms["dice"],
                    "train_grad_loss": avg_train_terms["grad"],
                    "train_flow_smooth_loss": avg_train_terms["flow_smooth"],
                    "train_boundary_distance_loss": avg_train_terms["boundary_distance"],
                    "train_cldice_loss": avg_train_terms["cldice"],
                    "val_total_loss": avg_val_total_loss,
                    "val_dice_loss": avg_val_terms["dice"],
                    "val_grad_loss": avg_val_terms["grad"],
                    "val_flow_smooth_loss": avg_val_terms["flow_smooth"],
                    "val_boundary_distance_loss": avg_val_terms["boundary_distance"],
                    "val_cldice_loss": avg_val_terms["cldice"],
                    "val_dice": avg_val_dice,
                    "epoch_sec": epoch_sec,
                    "epoch_min": epoch_sec / 60.0,
                    "avg_batch_ms": avg_batch_ms,
                    "peak_gpu_mem_mb": peak_mem_mb,
                    "train_sample_count": processed_train_samples,
                    "train_hard_sample_count": processed_hard_train_samples,
                    "train_mean_sample_weight": mean_train_sample_weight,
                }
            )

            memory_text = f"{peak_mem_mb:.2f} MB" if peak_mem_mb is not None else "N/A (CPU)"
            print(
                "Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | Val Total: {val_total:.4f} | "
                "Hard samples: {hard_count}/{sample_count} | Epoch Time: {epoch_sec:.2f}s | Avg Batch Time: {avg_batch_ms:.2f} ms | Peak GPU Mem: {memory}".format(
                    epoch=epoch_idx + 1,
                    train_loss=avg_train_loss,
                    val_dice=avg_val_dice,
                    val_total=avg_val_total_loss,
                    hard_count=processed_hard_train_samples,
                    sample_count=processed_train_samples,
                    epoch_sec=epoch_sec,
                    avg_batch_ms=avg_batch_ms,
                    memory=memory_text,
                )
            )

            val_score = avg_val_total_loss if args.best_checkpoint_metric == "total" else avg_val_loss
            if val_score < best_val_score:
                best_val_score = val_score
                best_val_loss = avg_val_loss
                checkpoint_metadata = {
                    "best_checkpoint_metric": args.best_checkpoint_metric,
                    "best_val_score": float(best_val_score),
                    "best_val_total_loss": float(avg_val_total_loss),
                    "init_checkpoint": init_checkpoint_metadata,
                    "hard_sample_training": hard_sample_metadata,
                }
                payload = checkpoint_payload(
                    model,
                    params,
                    run_name,
                    epoch_idx + 1,
                    avg_val_dice,
                    data_dir,
                    split_manifest_path,
                    dataset_spec,
                    extra_metadata=checkpoint_metadata,
                )
                torch.save(payload, best_checkpoint_path)
                print(f"Saved best checkpoint to {best_checkpoint_path}")

            if (epoch_idx + 1) % params.checkpoint_freq == 0:
                torch.save(
                    checkpoint_payload(
                        model,
                        params,
                        run_name,
                        epoch_idx + 1,
                        avg_val_dice,
                        data_dir,
                        split_manifest_path,
                        dataset_spec,
                        extra_metadata={
                            "best_checkpoint_metric": args.best_checkpoint_metric,
                            "best_val_score": float(best_val_score),
                            "best_val_total_loss": float(avg_val_total_loss),
                            "init_checkpoint": init_checkpoint_metadata,
                            "hard_sample_training": hard_sample_metadata,
                        },
                    ),
                    run_checkpoint_dir / f"teds_net_epoch_{epoch_idx + 1}.pth",
                )

        write_csv(
            run_dir / "train_epochs.csv",
            [
                "epoch",
                "train_loss",
                "train_dice_loss",
                "train_grad_loss",
                "train_flow_smooth_loss",
                "train_boundary_distance_loss",
                "train_cldice_loss",
                "val_total_loss",
                "val_dice_loss",
                "val_grad_loss",
                "val_flow_smooth_loss",
                "val_boundary_distance_loss",
                "val_cldice_loss",
                "val_dice",
                "epoch_sec",
                "epoch_min",
                "avg_batch_ms",
                "peak_gpu_mem_mb",
                "train_sample_count",
                "train_hard_sample_count",
                "train_mean_sample_weight",
            ],
            epoch_rows,
        )

        mean_epoch_sec = float(np.mean([row["epoch_sec"] for row in epoch_rows]))
        max_peak_mem = [row["peak_gpu_mem_mb"] for row in epoch_rows if row["peak_gpu_mem_mb"] is not None]
        train_summary = {
            "run_name": run_name,
            "dataset_id": dataset_spec.dataset_id,
            "dataset_name": dataset_spec.display_name,
            "dataset_registry": str(args.dataset_registry),
            "integrator": params.network.integrator,
            "r2net_blocks": params.network.r2net_blocks,
            "seed": params.seed,
            "device": str(device),
            "data_dir": str(data_dir),
            "split_manifest": str(split_manifest_path),
            "split_id": split_manifest_path.name,
            "train_count": manifest["counts"].get("train", 0),
            "val_count": manifest["counts"].get("val", 0),
            "test_count": manifest["counts"].get("test", 0),
            "best_val_dice": float(max(row["val_dice"] for row in epoch_rows)),
            "mean_epoch_sec": mean_epoch_sec,
            "mean_epoch_min": mean_epoch_sec / 60.0,
            "max_peak_gpu_mem_mb": float(max(max_peak_mem)) if max_peak_mem else None,
            "parameter_count": parameter_count,
            "parameter_count_x1e5": parameter_count_scaled,
            "checkpoint_path": str(best_checkpoint_path),
            "config_snapshot": asdict(params),
            "init_checkpoint": init_checkpoint_metadata,
            "best_checkpoint_metric": args.best_checkpoint_metric,
            "best_val_score": best_val_score,
            "best_val_dice_loss": best_val_loss,
            "dataset_spec": dataset_spec.to_dict(),
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "hard_sample_training": hard_sample_metadata,
        }
        write_json(run_dir / "train_summary.json", train_summary)
        print(f"Wrote training artifacts to {run_dir}")

        if not args.skip_final_eval:
            eval_result = run_evaluation(
                checkpoint_path=best_checkpoint_path,
                data_dir=data_dir,
                split_manifest=split_manifest_path,
                split=args.eval_split,
                run_name=run_name,
                output_dir=output_dir,
                checkpoint_root=checkpoint_root,
                warmup_batches=args.eval_warmup_batches,
                max_samples=args.eval_max_samples,
                device=device,
                integrator=params.network.integrator,
                r2net_blocks=params.network.r2net_blocks,
                seed=params.seed,
                dataset_id=dataset_spec.dataset_id,
                dataset_registry_path=args.dataset_registry,
                dataset_spec=dataset_spec,
            )

            comparison = write_comparison_artifacts(discover_run_dirs(output_dir), output_dir)
            print(f"Wrote comparison artifacts to {comparison['csv_path']} and {comparison['md_path']}")

        eval_summary = eval_result["summary"] if eval_result else None
        log_paths = write_experiment_log(
            experiment_log_dir,
            run_name,
            dataset_spec,
            research_purpose,
            status="completed",
            train_summary=train_summary,
            eval_summary=eval_summary,
            comparison=comparison,
            artifact_paths={
                "run_dir": str(run_dir),
                "train_summary": str(run_dir / "train_summary.json"),
                "train_epochs": str(run_dir / "train_epochs.csv"),
                "eval_summary": str(run_dir / "eval_summary.json") if eval_summary else None,
            },
            command=command,
            notes=args.experiment_notes,
            started_at=started_at,
        )
        train_summary["experiment_log_md"] = str(log_paths["md_path"])
        train_summary["experiment_log_json"] = str(log_paths["json_path"])
        write_json(run_dir / "train_summary.json", train_summary)
        print(f"Wrote experiment log to {log_paths['md_path']}")

        result = {
            "run_dir": run_dir,
            "train_summary": train_summary,
            "best_checkpoint_path": best_checkpoint_path,
            "experiment_log": log_paths,
        }
        if eval_result:
            result["eval_result"] = eval_result
        if comparison:
            result["comparison"] = comparison
        return result

    except Exception as exc:
        log_paths = write_experiment_log(
            experiment_log_dir,
            run_name,
            dataset_spec,
            research_purpose,
            status="failed",
            train_summary=train_summary,
            eval_summary=eval_result["summary"] if eval_result else None,
            comparison=comparison,
            artifact_paths={
                "run_dir": str(run_dir) if run_dir else None,
                "checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
            },
            command=command,
            notes=args.experiment_notes,
            error=repr(exc),
            started_at=started_at,
        )
        print(f"Wrote failed experiment log to {log_paths['md_path']}")
        raise


if __name__ == "__main__":
    train(parse_args())
