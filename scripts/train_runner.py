import argparse
import os
import shlex
import sys

import torch

from trainer import Trainer
from utils.acdc_preprocess import preprocess_acdc_dataset, resolve_manifest_path


class TrainRunner:
    """训练入口，负责参数解析、ACDC 预处理与 Trainer 调度。"""

    def __init__(self, args):
        self.args = args
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.command = "python " + " ".join(shlex.quote(arg) for arg in sys.argv)
        self.params = self.setup_params(args)
        self.prepare_acdc_data()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"当前训练设备: {device}")

        from network.TEDS_Net import TEDS_Net as Net

        net = Net(self.params)
        net.to(device)

        trainer = Trainer(
            params=self.params,
            device=device,
            net=net,
            project_root=self.project_root,
            command=self.command,
        )
        trainer.run()

    def _resolve_path(self, path_value):
        if path_value is None:
            return None
        if os.path.isabs(path_value):
            return path_value
        return os.path.abspath(os.path.join(self.project_root, path_value))

    def setup_params(self, args):
        if args.dataset == "mnist":
            from parameters.mnist_parameters import Parameters
        elif args.dataset == "ACDC":
            from parameters.acdc_parameters import Parameters
        else:
            raise ValueError(f"不支持的数据集: {args.dataset}")

        params = Parameters()
        params.data = args.dataset

        if args.epochs is not None:
            params.epoch = args.epochs
        if args.batch_size is not None:
            params.batch = args.batch_size
        if args.num_workers is not None:
            params.num_workers = args.num_workers
        if args.seed is not None:
            params.seed = args.seed
        if args.max_train_batches is not None:
            params.max_train_batches = args.max_train_batches
        if args.max_validation_batches is not None:
            params.max_validation_batches = args.max_validation_batches
        if args.max_test_batches is not None:
            params.max_test_batches = args.max_test_batches
        if args.skip_plot:
            params.plot_predictions = False
        if args.run_name:
            params.run_name = args.run_name
        if args.evaluate_only:
            params.evaluate_only = True

        data_path = self._resolve_path(args.data_path) if args.data_path else None
        raw_data_path = self._resolve_path(args.raw_data_path) if args.raw_data_path else None
        processed_data_path = self._resolve_path(args.processed_data_path) if args.processed_data_path else data_path

        if args.dataset == "ACDC":
            if raw_data_path is not None:
                params.dataset.raw_data_path = raw_data_path
            if processed_data_path is not None:
                params.dataset.processed_data_path = processed_data_path
            params.dataset.processed_data_path = self._resolve_path(params.dataset.processed_data_path)
            params.data_path = params.dataset.processed_data_path
            if not params.dataset.raw_data_path.startswith("<<"):
                params.dataset.raw_data_path = self._resolve_path(params.dataset.raw_data_path)
        elif data_path is not None:
            params.data_path = data_path

        self.force_preprocess = bool(args.force_preprocess)
        return params

    def prepare_acdc_data(self):
        if self.params.data != "ACDC":
            return

        processed_data_path = self.params.dataset.processed_data_path
        raw_data_path = self.params.dataset.raw_data_path
        manifest_path = resolve_manifest_path(processed_data_path)
        manifest_exists = os.path.isfile(manifest_path)

        if not manifest_exists and raw_data_path.startswith("<<"):
            raise FileNotFoundError(
                "ACDC 预处理缓存不存在，且未提供有效的 --raw-data-path，无法继续训练。"
            )

        if self.force_preprocess or not manifest_exists:
            preprocess_acdc_dataset(
                raw_data_path=raw_data_path,
                processed_data_path=processed_data_path,
                target_size=self.params.dataset.inshape,
                prior_radius=self.params.dataset.ps_meas[0],
                prior_thickness=self.params.dataset.ps_meas[1],
                force=self.force_preprocess,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 TEDS-Net 训练、评估与报告导出。")
    parser.add_argument("--dataset", choices=["ACDC", "mnist"], default="mnist", help="选择数据集。")
    parser.add_argument("--epochs", type=int, help="覆盖默认训练轮数。")
    parser.add_argument("--batch-size", type=int, dest="batch_size", help="覆盖默认 batch size。")
    parser.add_argument("--num-workers", type=int, help="覆盖 DataLoader worker 数量。")
    parser.add_argument("--seed", type=int, help="覆盖默认随机种子。")
    parser.add_argument("--data-path", help="兼容旧接口：MNIST 的数据目录，或 ACDC 的 processed_data_path。")
    parser.add_argument("--raw-data-path", help="ACDC 原始数据目录。")
    parser.add_argument("--processed-data-path", help="ACDC 预处理缓存目录。")
    parser.add_argument("--run-name", help="本次实验输出目录名称。")
    parser.add_argument("--evaluate-only", action="store_true", help="跳过训练，仅加载 best.pt 做评估。")
    parser.add_argument("--force-preprocess", action="store_true", help="强制重建 ACDC 预处理缓存。")
    parser.add_argument("--max-train-batches", type=int, help="每个 epoch 最多执行多少个训练 batch。")
    parser.add_argument("--max-validation-batches", type=int, help="验证阶段最多执行多少个 batch。")
    parser.add_argument("--max-test-batches", type=int, help="测试阶段最多执行多少个 batch。")
    parser.add_argument("--skip-plot", action="store_true", help="跳过评估可视化导出。")

    TrainRunner(parser.parse_args())
