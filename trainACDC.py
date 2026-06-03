import argparse
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
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
from utils.losses import dice_loss, grad_loss


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
    parser.add_argument("--integrator", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--eval-warmup-batches", type=int, default=1)
    parser.add_argument("--eval-max-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-final-eval", action="store_true")
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


def build_dataloaders(params, data_dir, split_manifest_path, dataset_spec):
    manifest = load_split_manifest(split_manifest_path, data_dir)
    loader_kwargs = dataset_spec.loader_kwargs()
    train_dataset = ACDCNpzDataset(
        data_dir,
        file_list=get_split_filenames(manifest, "train"),
        **loader_kwargs,
    )
    val_dataset = ACDCNpzDataset(
        data_dir,
        file_list=get_split_filenames(manifest, "val"),
        **loader_kwargs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params.batch,
        shuffle=False,
        num_workers=0,
    )
    return manifest, train_loader, val_loader


def checkpoint_payload(model, params, run_name, epoch, best_val_dice, data_dir, split_manifest, dataset_spec):
    return {
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
        if args.integrator is not None:
            params.network.integrator = normalize_integrator_name(args.integrator)
        else:
            params.network.integrator = normalize_integrator_name(params.network.integrator)
        if args.seed is not None:
            params.seed = int(args.seed)

        set_random_seed(params.seed)
        print(
            "Dataset: {name} ({dataset_id}) | data_dir={data_dir}".format(
                name=dataset_spec.display_name,
                dataset_id=dataset_spec.dataset_id,
                data_dir=data_dir,
            )
        )
        print(f"Integrator: {params.network.integrator} | Seed: {params.seed}")

        manifest, train_loader, val_loader = build_dataloaders(
            params,
            data_dir,
            split_manifest_path,
            dataset_spec,
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

        best_val_loss = float("inf")
        epoch_rows = []

        print(f"Starting training run '{run_name}'...")
        for epoch_idx in range(params.epoch):
            model.train()
            reset_peak_memory(device)
            epoch_start = time.perf_counter()
            total_train_loss = 0.0
            processed_train_batches = 0

            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch_idx + 1}/{params.epoch}")
            for batch_idx, (image, prior, label) in enumerate(train_pbar):
                if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                    break

                image = image.to(device)
                prior = prior.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                outputs = model(image, prior)

                if len(outputs) == 3:
                    pred_seg, flow_bulk, flow_ft = outputs
                    loss_reg = calc_grad.loss(None, flow_bulk) + calc_grad.loss(None, flow_ft)
                else:
                    pred_seg, flow = outputs
                    loss_reg = calc_grad.loss(None, flow)

                loss_dice = calc_dice.loss(label, pred_seg)
                total_loss = loss_dice + 10000.0 * loss_reg
                total_loss.backward()
                optimizer.step()

                total_train_loss += total_loss.item()
                processed_train_batches += 1
                train_pbar.set_postfix({"loss": total_loss.item(), "dice": 1.0 - loss_dice.item()})

            if processed_train_batches == 0:
                raise RuntimeError("No training batches were processed. Check --max-train-batches and split sizes.")

            model.eval()
            total_val_loss = 0.0
            processed_val_batches = 0
            with torch.no_grad():
                for batch_idx, (v_img, v_prior, v_lbl) in enumerate(val_loader):
                    if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                        break

                    v_img = v_img.to(device)
                    v_prior = v_prior.to(device)
                    v_lbl = v_lbl.to(device)
                    v_out = model(v_img, v_prior)
                    v_pred = v_out[0] if isinstance(v_out, tuple) else v_out
                    total_val_loss += calc_dice.loss(v_lbl, v_pred).item()
                    processed_val_batches += 1

            if processed_val_batches == 0:
                raise RuntimeError("No validation batches were processed. Check --max-val-batches and split sizes.")

            sync_cuda(device)
            epoch_sec = time.perf_counter() - epoch_start
            avg_train_loss = total_train_loss / processed_train_batches
            avg_val_loss = total_val_loss / processed_val_batches
            avg_val_dice = 1.0 - avg_val_loss
            avg_batch_ms = epoch_sec * 1000.0 / processed_train_batches
            peak_mem_mb = peak_gpu_memory_mb(device)

            epoch_rows.append(
                {
                    "epoch": epoch_idx + 1,
                    "train_loss": avg_train_loss,
                    "val_dice": avg_val_dice,
                    "epoch_sec": epoch_sec,
                    "epoch_min": epoch_sec / 60.0,
                    "avg_batch_ms": avg_batch_ms,
                    "peak_gpu_mem_mb": peak_mem_mb,
                }
            )

            memory_text = f"{peak_mem_mb:.2f} MB" if peak_mem_mb is not None else "N/A (CPU)"
            print(
                "Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | "
                "Epoch Time: {epoch_sec:.2f}s | Avg Batch Time: {avg_batch_ms:.2f} ms | Peak GPU Mem: {memory}".format(
                    epoch=epoch_idx + 1,
                    train_loss=avg_train_loss,
                    val_dice=avg_val_dice,
                    epoch_sec=epoch_sec,
                    avg_batch_ms=avg_batch_ms,
                    memory=memory_text,
                )
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                payload = checkpoint_payload(
                    model,
                    params,
                    run_name,
                    epoch_idx + 1,
                    avg_val_dice,
                    data_dir,
                    split_manifest_path,
                    dataset_spec,
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
                    ),
                    run_checkpoint_dir / f"teds_net_epoch_{epoch_idx + 1}.pth",
                )

        write_csv(
            run_dir / "train_epochs.csv",
            [
                "epoch",
                "train_loss",
                "val_dice",
                "epoch_sec",
                "epoch_min",
                "avg_batch_ms",
                "peak_gpu_mem_mb",
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
            "dataset_spec": dataset_spec.to_dict(),
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
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
