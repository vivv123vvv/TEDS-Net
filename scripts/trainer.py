import json
import math
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from utils.acdc_evaluator import evaluate_acdc_model, evaluate_acdc_subset
from utils.losses import (
    BendingEnergyLoss,
    BoundarySDFLoss,
    JacobianBarrierLoss,
    PoseRegressionLoss,
    RingValidityLoss,
    dice_loss,
    grad_loss,
)


class Trainer:
    """训练、验证、评估与产物导出总控。"""

    def __init__(self, params, device, net, project_root, command):
        self.params = params
        self.device = device
        self.net = net
        self.project_root = project_root
        self.command = command
        self.output_dir = self._prepare_output_dir()
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.history_path = os.path.join(self.output_dir, "history.json")
        self.training_summary_path = os.path.join(self.output_dir, "training_summary.json")
        self.best_checkpoint_path = os.path.join(self.checkpoint_dir, "best.pt")
        self.last_checkpoint_path = os.path.join(self.checkpoint_dir, "last.pt")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._set_seed()
        self.get_dataloader()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.history = {
            "epochs": [],
            "best_validation_loss": None,
            "best_validation_metrics": None,
        }

        self.boundary_loss = BoundarySDFLoss(scale=self.params.loss_params.boundary_sdf_scale)
        self.smooth_loss = BendingEnergyLoss()
        self.jacobian_loss = JacobianBarrierLoss(epsilon=self.params.loss_params.jacobian_epsilon)
        self.ring_loss = RingValidityLoss(self.params)
        self.pose_loss = PoseRegressionLoss()
        self.selected_validation_threshold = float(self.params.threshold)

    def _prepare_output_dir(self):
        if os.path.isabs(self.params.output_root):
            root_dir = self.params.output_root
        else:
            root_dir = os.path.join(self.project_root, self.params.output_root)
        output_dir = os.path.join(root_dir, self.params.run_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _set_seed(self):
        seed = int(getattr(self.params, "seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _iterate_subset(self, subset, max_batches=0, progress=True):
        iterator = self.dataloader_dic[subset]
        if progress:
            iterator = tqdm(iterator)

        for batch_index, batch in enumerate(iterator):
            if max_batches and batch_index >= max_batches:
                break
            yield batch

    def get_dataloader(self):
        if self.params.data == "mnist":
            from dataloaders.setup import setup_mnist_dataloader as setup_dataloader
        elif self.params.data == "ACDC":
            from dataloaders.setup import setup_acdc_dataloader as setup_dataloader
        else:
            raise ValueError(f"不支持的数据集: {self.params.data}")

        self.dataloader_dic = setup_dataloader(self.params, ["train", "validation", "test"])

    def _build_optimizer(self):
        if self.params.data == "ACDC" and getattr(self.params.training, "optimizer", "adamw").lower() == "adamw":
            return torch.optim.AdamW(
                self.net.parameters(),
                lr=self.params.lr,
                weight_decay=self.params.training.weight_decay,
            )
        return torch.optim.Adam(self.net.parameters(), lr=self.params.lr)

    def _build_scheduler(self):
        if self.params.data != "ACDC" or not getattr(self.params, "lr_sch", False):
            return None

        warmup_epochs = max(int(self.params.training.warmup_epochs), 1)
        total_epochs = max(int(self.params.epoch), 1)

        def lr_lambda(epoch_index):
            current_epoch = epoch_index + 1
            if current_epoch <= warmup_epochs:
                return current_epoch / warmup_epochs

            progress = (current_epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _split_batch(self, batch):
        if len(batch) == 4:
            image, prior_shape, target, metadata = batch
            return image, prior_shape, target, metadata
        image, prior_shape, target = batch
        return image, prior_shape, target, None

    def _move_target_to_device(self, target):
        if isinstance(target, dict):
            return {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()
            }
        return target.to(self.device)

    def _ensure_batch_axis(self, image, prior_shape, target):
        if isinstance(image, torch.Tensor) and image.ndim == 3:
            image = image.unsqueeze(0)
        if isinstance(prior_shape, torch.Tensor) and prior_shape.ndim == 3:
            prior_shape = prior_shape.unsqueeze(0)

        if isinstance(target, dict):
            normalized = {}
            for key, value in target.items():
                if not isinstance(value, torch.Tensor):
                    normalized[key] = value
                    continue
                if key in {"mask", "sdf"} and value.ndim == 3:
                    normalized[key] = value.unsqueeze(0)
                elif key == "pose" and value.ndim == 1:
                    normalized[key] = value.unsqueeze(0)
                else:
                    normalized[key] = value
            return image, prior_shape, normalized

        if isinstance(target, torch.Tensor) and target.ndim == 3:
            target = target.unsqueeze(0)
        return image, prior_shape, target

    def _current_acdc_weights(self, epoch):
        stage_a_end = int(self.params.loss_params.stage_a_end)
        jacobian_max = float(self.params.loss_params.jacobian_weight_max)
        jacobian_warmup = int(self.params.loss_params.jacobian_warmup_epochs)

        if epoch <= stage_a_end:
            return {
                "dice": self.params.loss_params.dice_weight,
                "boundary": self.params.loss_params.boundary_weight,
                "smooth": 0.0,
                "jacobian": 0.0,
                "ring": 0.0,
                "pose": self.params.loss_params.pose_weight,
            }

        warmup_progress = min(max(epoch - stage_a_end, 0), max(jacobian_warmup - stage_a_end, 1))
        warmup_denominator = max(jacobian_warmup - stage_a_end, 1)
        jacobian_weight = jacobian_max * (warmup_progress / warmup_denominator)

        return {
            "dice": self.params.loss_params.dice_weight,
            "boundary": self.params.loss_params.boundary_weight,
            "smooth": self.params.loss_params.smooth_weight,
            "jacobian": jacobian_weight,
            "ring": self.params.loss_params.ring_weight,
            "pose": self.params.loss_params.pose_weight,
        }

    def perform_losses(self, target, output, epoch):
        if isinstance(output, dict):
            weights = self._current_acdc_weights(epoch)
            mask = target["mask"]
            sdf = target["sdf"]
            pose = target["pose"]

            total = output["final_mask"].new_tensor(0.0)
            components = {}

            components["dice"] = dice_loss().loss(mask, output["final_mask"], loss_mult=weights["dice"])
            components["boundary"] = self.boundary_loss.loss(sdf, output["final_sdf"], loss_mult=weights["boundary"])
            components["pose"] = self.pose_loss.loss(pose, output["pose_params"], loss_mult=weights["pose"])
            total = total + components["dice"] + components["boundary"] + components["pose"]

            if weights["smooth"] > 0.0:
                components["smooth"] = self.smooth_loss.loss(output["composed_flow"], loss_mult=weights["smooth"])
                total = total + components["smooth"]

            if weights["jacobian"] > 0.0:
                components["jacobian"] = self.jacobian_loss.loss(output["composed_flow"], loss_mult=weights["jacobian"])
                total = total + components["jacobian"]

            if weights["ring"] > 0.0:
                components["ring"] = self.ring_loss.loss(output["final_mask"], output["pose_params"], loss_mult=weights["ring"])
                total = total + components["ring"]

            return total, {key: float(value.item()) for key, value in components.items()}

        loss = output[0].new_tensor(0.0)
        for current_output, loss_name, weight in zip(
            output,
            self.params.loss_params.loss,
            self.params.loss_params.weight,
        ):
            if loss_name == "dice":
                current_loss = dice_loss().loss(target, current_output, loss_mult=weight)
            elif "grad" in loss_name:
                current_loss = grad_loss(self.params).loss(target, current_output, loss_mult=weight)
            else:
                raise ValueError(f"不支持的损失函数: {loss_name}")
            loss = loss + current_loss
        return loss, {}

    def _prediction_dice(self, target, output):
        if isinstance(output, dict):
            prediction = (output["final_mask"] > self.params.threshold).float()
            return 1.0 - dice_loss().np_loss(target["mask"], prediction)

        prediction = (output[0] > self.params.threshold).float()
        return 1.0 - dice_loss().np_loss(target, prediction)

    def _validation_threshold_candidates(self, epoch):
        if self.params.data != "ACDC":
            return [float(self.params.threshold)]

        if epoch >= int(self.params.training.threshold_sweep_start):
            return list(self.params.training.threshold_candidates)
        return [float(self.params.threshold)]

    def _validation_score_key(self, metrics):
        return (
            float(metrics.get("topology_keep_rate", 0.0)),
            -float(metrics.get("hd95_mean", float("inf"))),
            float(metrics.get("dice_mean", metrics.get("dice", 0.0))),
        )

    def train(self):
        best_validation_loss = float("inf")
        best_validation_key = None
        best_epoch = 0

        for epoch_index in range(self.params.epoch):
            epoch = epoch_index + 1
            self.net.train()
            training_losses = []
            training_dices = []

            for batch in self._iterate_subset(
                "train",
                max_batches=getattr(self.params, "max_train_batches", 0),
            ):
                image, prior_shape, target, _ = self._split_batch(batch)
                image, prior_shape, target = self._ensure_batch_axis(image, prior_shape, target)
                image = image.to(self.device)
                prior_shape = prior_shape.to(self.device)
                target = self._move_target_to_device(target)

                self.optimizer.zero_grad()
                output = self.net(image, prior_shape)
                loss, _ = self.perform_losses(target, output, epoch)
                loss.backward()
                self.optimizer.step()

                training_losses.append(loss.item())
                training_dices.append(self._prediction_dice(target, output))

            if self.scheduler is not None:
                self.scheduler.step()

            validation_metrics = self.validate(epoch)
            epoch_record = {
                "epoch": epoch,
                "train_loss": float(np.mean(training_losses)) if training_losses else float("nan"),
                "train_dice": float(np.mean(training_dices)) if training_dices else float("nan"),
                "validation_loss": validation_metrics["loss"],
                "validation_dice": validation_metrics["dice"],
                "validation_hd95": validation_metrics.get("hd95_mean"),
                "validation_topology_keep_rate": validation_metrics.get("topology_keep_rate"),
                "selected_threshold": validation_metrics.get("selected_threshold"),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }
            self.history["epochs"].append(epoch_record)

            self._save_checkpoint(self.last_checkpoint_path, epoch, is_best=False, validation_metrics=validation_metrics)
            current_key = self._validation_score_key(validation_metrics)
            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                self.history["best_validation_loss"] = best_validation_loss

            if best_validation_key is None or current_key > best_validation_key:
                best_validation_key = current_key
                best_epoch = epoch
                self.selected_validation_threshold = float(validation_metrics["selected_threshold"])
                self.history["best_validation_metrics"] = validation_metrics
                self._save_checkpoint(self.best_checkpoint_path, epoch, is_best=True, validation_metrics=validation_metrics)

            if self.params.checkpoint_freq and epoch % self.params.checkpoint_freq == 0:
                epoch_checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:03d}.pt")
                self._save_checkpoint(epoch_checkpoint_path, epoch, is_best=False, validation_metrics=validation_metrics)

            self._write_history()
            self._write_training_summary(best_epoch=best_epoch)
            print(
                f"[{epoch}] train_loss={epoch_record['train_loss']:.6f} "
                f"train_dice={epoch_record['train_dice']:.6f} "
                f"validation_loss={epoch_record['validation_loss']:.6f} "
                f"validation_dice={epoch_record['validation_dice']:.6f} "
                f"validation_hd95={validation_metrics.get('hd95_mean', float('nan')):.6f} "
                f"validation_topology={validation_metrics.get('topology_keep_rate', float('nan')):.6f} "
                f"threshold={validation_metrics.get('selected_threshold', self.params.threshold):.2f}"
            )

    def validate(self, epoch):
        self.net.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in self._iterate_subset(
                "validation",
                max_batches=getattr(self.params, "max_validation_batches", 0),
            ):
                image, prior_shape, target, _ = self._split_batch(batch)
                image, prior_shape, target = self._ensure_batch_axis(image, prior_shape, target)
                image = image.to(self.device)
                prior_shape = prior_shape.to(self.device)
                target = self._move_target_to_device(target)
                output = self.net(image, prior_shape)
                loss, _ = self.perform_losses(target, output, epoch)
                validation_losses.append(loss.item())

        if self.params.data != "ACDC":
            return {
                "loss": float(np.mean(validation_losses)) if validation_losses else float("nan"),
                "dice": float("nan"),
                "selected_threshold": float(self.params.threshold),
            }

        threshold_candidates = self._validation_threshold_candidates(epoch)
        summary_metrics, _, _ = evaluate_acdc_subset(
            model=self.net,
            dataloader=self.dataloader_dic["validation"],
            device=self.device,
            params=self.params,
            subset_name="validation",
            threshold_candidates=threshold_candidates,
        )
        return {
            "loss": float(np.mean(validation_losses)) if validation_losses else float("nan"),
            "dice": summary_metrics["dice_mean"],
            "dice_mean": summary_metrics["dice_mean"],
            "hd_mean": summary_metrics["hd_mean"],
            "hd95_mean": summary_metrics["hd95_mean"],
            "assd_mean": summary_metrics["assd_mean"],
            "topology_keep_rate": summary_metrics["topology_keep_rate"],
            "folding_ratio_mean": summary_metrics["folding_ratio_mean"],
            "selected_threshold": summary_metrics["selected_threshold"],
        }

    def _save_checkpoint(self, checkpoint_path, epoch, is_best, validation_metrics):
        payload = {
            "epoch": epoch,
            "is_best": is_best,
            "model_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "params": self.params.to_dict() if hasattr(self.params, "to_dict") else {},
            "history": self.history,
            "selected_validation_threshold": float(validation_metrics.get("selected_threshold", self.params.threshold)),
            "validation_metrics": validation_metrics,
        }
        torch.save(payload, checkpoint_path)

    def _write_history(self):
        with open(self.history_path, "w", encoding="utf-8") as handle:
            json.dump(self.history, handle, ensure_ascii=False, indent=2)

    def _write_training_summary(self, best_epoch):
        summary = {
            "run_name": self.params.run_name,
            "output_dir": self.output_dir,
            "best_checkpoint_path": self.best_checkpoint_path,
            "last_checkpoint_path": self.last_checkpoint_path,
            "best_epoch": int(best_epoch),
            "best_validation_loss": self.history.get("best_validation_loss"),
            "best_validation_metrics": self.history.get("best_validation_metrics"),
            "device": str(self.device),
            "train_command": self.command,
            "processed_data_path": getattr(self.params.dataset, "processed_data_path", None),
            "raw_data_path": getattr(self.params.dataset, "raw_data_path", None),
            "split_counts": getattr(self.params, "acdc_split_counts", {}),
            "preprocess_summary": getattr(self.params, "acdc_manifest_summary", {}),
        }
        with open(self.training_summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    def _load_best_checkpoint(self):
        if not os.path.isfile(self.best_checkpoint_path):
            raise FileNotFoundError(f"未找到最佳模型权重: {self.best_checkpoint_path}")
        checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state", checkpoint)
        self.net.load_state_dict(state_dict)
        self.selected_validation_threshold = float(
            checkpoint.get("selected_validation_threshold", self.params.threshold)
        )
        self.params.selected_validation_threshold = self.selected_validation_threshold

    def run(self):
        if not getattr(self.params, "evaluate_only", False):
            self.train()
        self._load_best_checkpoint()
        return self.evaluate()

    def evaluate(self):
        if self.params.data == "ACDC":
            return evaluate_acdc_model(
                model=self.net,
                dataloader=self.dataloader_dic["test"],
                device=self.device,
                params=self.params,
                run_output_dir=self.output_dir,
                project_root=self.project_root,
                command=self.command,
            )
        return self._evaluate_generic()

    def _evaluate_generic(self):
        self.net.eval()
        test_dice = []
        x = labels = prior_shape = output = None
        with torch.no_grad():
            for batch in self._iterate_subset(
                "test",
                max_batches=getattr(self.params, "max_test_batches", 0),
                progress=False,
            ):
                x, prior_shape, labels, _ = self._split_batch(batch)
                x, prior_shape, labels = self._ensure_batch_axis(x, prior_shape, labels)
                output = self.net(x.to(self.device), prior_shape.to(self.device))
                prediction = output["final_mask"] if isinstance(output, dict) else output[0]
                prediction = (prediction > self.params.threshold).int()
                test_dice.append(1.0 - dice_loss().np_loss(labels.to(self.device), prediction))

        if not test_dice:
            raise RuntimeError("未执行任何测试集评估。")

        metrics = {
            "dice_mean": float(np.mean(test_dice)),
            "dice_std": float(np.std(test_dice)),
        }
        print(" - -" * 10)
        print(f" Test Dice: {metrics['dice_mean']} +/- {metrics['dice_std']} ")
        print(" - -" * 10)
        return metrics
