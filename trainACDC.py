import argparse
import json
import os
import random
import shlex
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from dataloaders.setup import setup_acdc_dataloader
from network.TEDS_Net import TEDS_Net
from parameters.acdc_parameters import Parameters
from utils.acdc_metrics import model_parameter_count
from utils.acdc_preprocess import load_manifest
from utils.losses import dice_loss, grad_loss


def _resolve_path(project_root, path_value):
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(project_root, path_value))


def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean_or_nan(values):
    if not values:
        return float("nan")
    return float(np.mean(values))


def _save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_argument_parser():
    defaults = Parameters()
    parser = argparse.ArgumentParser(description="训练 ACDC 上的 TEDS-Net，并保存完整训练产物。")
    parser.add_argument("--raw-data-path", default=defaults.dataset.raw_data_path, help="原始 ACDC 数据目录，仅用于记录到训练摘要。")
    parser.add_argument("--processed-data-path", default=defaults.dataset.processed_data_path, help="预处理缓存目录。")
    parser.add_argument("--output-root", default=defaults.output_root, help="训练输出根目录。")
    parser.add_argument("--run-name", default=defaults.run_name, help="本次实验的输出目录名称。")
    parser.add_argument("--epochs", type=int, default=defaults.epoch, help="训练 epoch 数。")
    parser.add_argument("--batch-size", type=int, default=defaults.batch, help="训练 batch size。")
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers, help="DataLoader worker 数。")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="随机种子。")
    parser.add_argument("--checkpoint-freq", type=int, default=defaults.checkpoint_freq, help="额外保存整轮 checkpoint 的频率。")
    return parser


def configure_params(project_root, args, base_params=None):
    params = base_params or Parameters()
    params.epoch = int(args.epochs)
    params.batch = int(args.batch_size)
    params.num_workers = int(args.num_workers)
    params.seed = int(args.seed)
    params.checkpoint_freq = int(args.checkpoint_freq)
    params.run_name = args.run_name
    params.output_root = _resolve_path(project_root, args.output_root)

    processed_data_path = _resolve_path(project_root, args.processed_data_path)
    params.dataset.processed_data_path = processed_data_path
    params.data_path = processed_data_path

    if args.raw_data_path:
        raw_data_path = _resolve_path(project_root, args.raw_data_path)
        params.dataset.raw_data_path = raw_data_path

    return params


def prepare_output_dir(params):
    output_dir = os.path.join(params.output_root, params.run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    return output_dir


def perform_losses(labels, outputs, params):
    total_loss = 0.0
    for output_tensor, loss_name, weight in zip(
        outputs,
        params.loss_params.loss,
        params.loss_params.weight,
    ):
        if loss_name == "dice":
            current_loss = dice_loss().loss(labels, output_tensor, loss_mult=weight)
        elif "grad" in loss_name:
            current_loss = grad_loss(params).loss(labels, output_tensor, loss_mult=weight)
        else:
            raise ValueError(f"不支持的损失函数：{loss_name}")
        total_loss += current_loss
    return total_loss


def prediction_dice(labels, outputs, threshold):
    prediction = (outputs[0] > threshold).float()
    return 1.0 - dice_loss().np_loss(labels, prediction)


def validate(model, dataloader, device, params):
    model.eval()
    validation_losses = []
    validation_dices = []

    with torch.no_grad():
        for image, prior_shape, labels in dataloader:
            image = image.to(device)
            prior_shape = prior_shape.to(device)
            labels = labels.to(device)

            outputs = model(image, prior_shape)
            loss = perform_losses(labels, outputs, params)
            validation_losses.append(loss.item())
            validation_dices.append(prediction_dice(labels, outputs, params.threshold))

    return {
        "loss": _mean_or_nan(validation_losses),
        "dice": _mean_or_nan(validation_dices),
    }


def save_checkpoint(path, epoch, is_best, model, optimizer, params, history):
    payload = {
        "epoch": epoch,
        "is_best": is_best,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "params": params.to_dict(),
        "history": history,
    }
    torch.save(payload, path)


def train(args=None):
    project_root = os.path.abspath(os.path.dirname(__file__))
    parser = build_argument_parser()
    parsed_args = parser.parse_args(args=args)
    command = "python " + " ".join(shlex.quote(argument) for argument in sys.argv)

    params = configure_params(project_root, parsed_args)
    _set_random_seed(params.seed)
    manifest = load_manifest(params.dataset.processed_data_path)
    if params.dataset.raw_data_path.startswith("<<"):
        params.dataset.raw_data_path = manifest["raw_data_root"]

    output_dir = prepare_output_dir(params)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    history_path = os.path.join(output_dir, "history.json")
    training_summary_path = os.path.join(output_dir, "training_summary.json")
    best_checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
    last_checkpoint_path = os.path.join(checkpoint_dir, "last.pt")
    compatibility_checkpoint_path = os.path.join(checkpoint_dir, "best_teds_net.pth")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"当前训练设备：{device}")

    dataloader_dict = setup_acdc_dataloader(params, ["train", "validation", "test"])
    model = TEDS_Net(params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    parameter_count = model_parameter_count(model)

    history = {
        "epochs": [],
        "best_validation_loss": None,
        "best_epoch": 0,
        "parameter_count": parameter_count,
        "train_command": command,
    }

    best_validation_loss = float("inf")

    for epoch in range(params.epoch):
        epoch_start = time.time()
        model.train()
        training_losses = []
        training_dices = []

        progress = tqdm(dataloader_dict["train"], desc=f"Epoch {epoch + 1}/{params.epoch}")
        for image, prior_shape, labels in progress:
            image = image.to(device)
            prior_shape = prior_shape.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(image, prior_shape)
            loss = perform_losses(labels, outputs, params)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            batch_dice = prediction_dice(labels, outputs, params.threshold)
            training_losses.append(batch_loss)
            training_dices.append(batch_dice)
            progress.set_postfix(loss=f"{batch_loss:.4f}", dice=f"{batch_dice:.4f}")

        validation_metrics = validate(model, dataloader_dict["validation"], device, params)
        epoch_minutes = float((time.time() - epoch_start) / 60.0)
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": _mean_or_nan(training_losses),
            "train_dice": _mean_or_nan(training_dices),
            "validation_loss": validation_metrics["loss"],
            "validation_dice": validation_metrics["dice"],
            "epoch_minutes": epoch_minutes,
        }
        history["epochs"].append(epoch_record)

        save_checkpoint(last_checkpoint_path, epoch + 1, False, model, optimizer, params, history)
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            history["best_validation_loss"] = float(best_validation_loss)
            history["best_epoch"] = epoch + 1
            save_checkpoint(best_checkpoint_path, epoch + 1, True, model, optimizer, params, history)
            torch.save(model.state_dict(), compatibility_checkpoint_path)

        if params.checkpoint_freq and (epoch + 1) % params.checkpoint_freq == 0:
            epoch_checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1:03d}.pt")
            save_checkpoint(epoch_checkpoint_path, epoch + 1, False, model, optimizer, params, history)

        _save_json(history_path, history)
        print(
            f"[{epoch + 1}] "
            f"train_loss={epoch_record['train_loss']:.6f} "
            f"train_dice={epoch_record['train_dice']:.6f} "
            f"validation_loss={epoch_record['validation_loss']:.6f} "
            f"validation_dice={epoch_record['validation_dice']:.6f} "
            f"epoch_minutes={epoch_record['epoch_minutes']:.4f}"
        )

    epoch_minutes = [record["epoch_minutes"] for record in history["epochs"]]
    training_summary = {
        "run_name": params.run_name,
        "output_dir": output_dir,
        "best_checkpoint_path": best_checkpoint_path,
        "compatibility_checkpoint_path": compatibility_checkpoint_path,
        "last_checkpoint_path": last_checkpoint_path,
        "best_epoch": history["best_epoch"],
        "best_validation_loss": history["best_validation_loss"],
        "epoch_time_minutes_mean": _mean_or_nan(epoch_minutes),
        "parameter_count": parameter_count,
        "parameter_count_e5": float(parameter_count / 1e5),
        "device": str(device),
        "train_command": command,
        "processed_data_path": params.dataset.processed_data_path,
        "raw_data_path": params.dataset.raw_data_path,
        "split_counts": params.acdc_split_counts,
        "preprocess_summary": params.acdc_manifest_summary,
    }
    _save_json(training_summary_path, training_summary)

    print("训练完成。")
    print(f"最佳模型：{best_checkpoint_path}")
    print(f"训练摘要：{training_summary_path}")


if __name__ == "__main__":
    train()
