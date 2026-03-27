import argparse
import json
import os
import shlex
import subprocess
import sys

import numpy as np
import torch

from dataloaders.setup import setup_acdc_dataloader
from network.TEDS_Net import TEDS_Net
from parameters.acdc_parameters import Parameters
from utils.acdc_evaluator import evaluate_acdc_model
from utils.acdc_metrics import model_parameter_count


def _resolve_path(project_root, path_value):
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(project_root, path_value))


def _load_json_if_exists(path):
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _git_output(project_root, *args):
    try:
        return subprocess.check_output(args, cwd=project_root, text=True).strip()
    except Exception:
        return "unknown"


def build_argument_parser(description="评估 ACDC 模型并导出报告。"):
    defaults = Parameters()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--raw-data-path", default=None, help="原始 ACDC 数据目录，仅用于写入报告。")
    parser.add_argument("--processed-data-path", default=defaults.dataset.processed_data_path, help="预处理缓存目录。")
    parser.add_argument("--output-root", default=defaults.output_root, help="训练输出根目录。")
    parser.add_argument("--run-name", default=defaults.run_name, help="实验目录名称。")
    parser.add_argument("--checkpoint-path", default=None, help="显式指定要评估的 checkpoint 路径。")
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers, help="DataLoader worker 数。")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="随机种子。")
    parser.add_argument("--threshold", type=float, default=defaults.threshold, help="预测二值化阈值。")
    parser.add_argument("--skip-plot", action="store_true", help="跳过可视化导出。")
    return parser


def _build_params(project_root, parsed_args, checkpoint_payload):
    params_dict = checkpoint_payload.get("params")
    if params_dict:
        params = Parameters.from_dict(params_dict)
    else:
        params = Parameters()

    params.output_root = _resolve_path(project_root, parsed_args.output_root)
    params.run_name = parsed_args.run_name
    params.dataset.processed_data_path = _resolve_path(project_root, parsed_args.processed_data_path)
    params.data_path = params.dataset.processed_data_path
    params.num_workers = int(parsed_args.num_workers)
    params.seed = int(parsed_args.seed)
    params.threshold = float(parsed_args.threshold)
    params.plot_predictions = not parsed_args.skip_plot

    if parsed_args.raw_data_path is not None:
        params.dataset.raw_data_path = _resolve_path(project_root, parsed_args.raw_data_path)
    return params


def run_evaluation(args=None):
    project_root = os.path.abspath(os.path.dirname(__file__))
    if isinstance(args, argparse.Namespace):
        parsed_args = args
    else:
        parser = build_argument_parser()
        parsed_args = parser.parse_args(args=args)

    run_output_dir = os.path.join(_resolve_path(project_root, parsed_args.output_root), parsed_args.run_name)
    history_path = os.path.join(run_output_dir, "history.json")
    training_summary_path = os.path.join(run_output_dir, "training_summary.json")
    training_summary = _load_json_if_exists(training_summary_path)
    history = _load_json_if_exists(history_path)

    checkpoint_path = parsed_args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = training_summary.get("best_checkpoint_path")
    if checkpoint_path is None:
        checkpoint_path = os.path.join(run_output_dir, "checkpoints", "best.pt")
    checkpoint_path = _resolve_path(project_root, checkpoint_path)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"未找到待评估 checkpoint：{checkpoint_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_payload = torch.load(checkpoint_path, map_location=device)
    params = _build_params(project_root, parsed_args, checkpoint_payload)

    model = TEDS_Net(params).to(device)
    model.load_state_dict(checkpoint_payload["model_state"])

    dataloader_dict = setup_acdc_dataloader(params, ["train", "validation", "test"])

    epoch_minutes = [record.get("epoch_minutes") for record in history.get("epochs", []) if "epoch_minutes" in record]
    epoch_time_minutes_mean = training_summary.get("epoch_time_minutes_mean")
    if epoch_time_minutes_mean is None:
        epoch_time_minutes_mean = float(np.mean(epoch_minutes)) if epoch_minutes else float("nan")

    if params.dataset.raw_data_path.startswith("<<"):
        params.dataset.raw_data_path = training_summary.get("raw_data_path", params.dataset.raw_data_path)

    context = {
        "git_branch": _git_output(project_root, "git", "rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": _git_output(project_root, "git", "rev-parse", "HEAD"),
        "run_name": params.run_name,
        "raw_data_path": params.dataset.raw_data_path,
        "processed_data_path": params.dataset.processed_data_path,
        "run_output_dir": run_output_dir,
        "best_checkpoint_path": checkpoint_path,
        "train_command": training_summary.get("train_command", "unknown"),
        "eval_command": "python " + " ".join(shlex.quote(argument) for argument in sys.argv),
        "device": str(device),
        "epochs": params.epoch,
        "batch_size": params.batch,
        "best_epoch": training_summary.get("best_epoch", checkpoint_payload.get("epoch", 0)),
        "epoch_time_minutes_mean": float(epoch_time_minutes_mean),
        "parameter_count": int(training_summary.get("parameter_count", model_parameter_count(model))),
        "split_counts": params.acdc_split_counts,
        "preprocess_summary": params.acdc_manifest_summary,
    }

    summary_metrics = evaluate_acdc_model(
        model=model,
        dataloader=dataloader_dict["test"],
        device=device,
        params=params,
        run_output_dir=run_output_dir,
        project_root=project_root,
        context=context,
    )

    print("评估完成。")
    print(json.dumps(summary_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_evaluation()
