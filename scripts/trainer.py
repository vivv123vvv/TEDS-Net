import json
import os

import numpy as np
import torch
from tqdm import tqdm

from utils.acdc_evaluator import evaluate_acdc_model
from utils.losses import dice_loss, grad_loss


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
        self.best_checkpoint_path = os.path.join(self.checkpoint_dir, "best.pt")
        self.last_checkpoint_path = os.path.join(self.checkpoint_dir, "last.pt")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.get_dataloader()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.params.lr)
        self.history = {
            "epochs": [],
            "best_validation_loss": None,
        }

    def _prepare_output_dir(self):
        if os.path.isabs(self.params.output_root):
            root_dir = self.params.output_root
        else:
            root_dir = os.path.join(self.project_root, self.params.output_root)
        output_dir = os.path.join(root_dir, self.params.run_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

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

    def perform_losses(self, labels, output):
        loss = 0.0
        for current_output, loss_name, weight in zip(
            output,
            self.params.loss_params.loss,
            self.params.loss_params.weight,
        ):
            if loss_name == "dice":
                current_loss = dice_loss().loss(labels, current_output, loss_mult=weight)
            elif "grad" in loss_name:
                current_loss = grad_loss(self.params).loss(labels, current_output, loss_mult=weight)
            else:
                raise ValueError(f"不支持的损失函数: {loss_name}")
            loss += current_loss
        return loss

    def _prediction_dice(self, labels, output):
        prediction = (output[0] > self.params.threshold).float()
        return 1.0 - dice_loss().np_loss(labels, prediction)

    def train(self):
        best_validation_loss = float("inf")
        for epoch in range(self.params.epoch):
            self.net.train()
            training_losses = []
            training_dices = []

            for image, prior_shape, labels in self._iterate_subset(
                "train",
                max_batches=getattr(self.params, "max_train_batches", 0),
            ):
                image = image.to(self.device)
                prior_shape = prior_shape.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                output = self.net(image, prior_shape)
                loss = self.perform_losses(labels, output)
                loss.backward()
                self.optimizer.step()

                training_losses.append(loss.item())
                training_dices.append(self._prediction_dice(labels, output))

            validation_metrics = self.validate()
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(training_losses)) if training_losses else float("nan"),
                "train_dice": float(np.mean(training_dices)) if training_dices else float("nan"),
                "validation_loss": validation_metrics["loss"],
                "validation_dice": validation_metrics["dice"],
            }
            self.history["epochs"].append(epoch_record)

            self._save_checkpoint(self.last_checkpoint_path, epoch + 1, is_best=False)
            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                self.history["best_validation_loss"] = best_validation_loss
                self._save_checkpoint(self.best_checkpoint_path, epoch + 1, is_best=True)

            if self.params.checkpoint_freq and (epoch + 1) % self.params.checkpoint_freq == 0:
                epoch_checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch + 1:03d}.pt")
                self._save_checkpoint(epoch_checkpoint_path, epoch + 1, is_best=False)

            self._write_history()
            print(
                f"[{epoch + 1}] training_loss={epoch_record['train_loss']:.6f} "
                f"training_dice={epoch_record['train_dice']:.6f} "
                f"validation_loss={epoch_record['validation_loss']:.6f} "
                f"validation_dice={epoch_record['validation_dice']:.6f}"
            )

    def validate(self):
        self.net.eval()
        validation_losses = []
        validation_dices = []
        with torch.no_grad():
            for image, prior_shape, labels in self._iterate_subset(
                "validation",
                max_batches=getattr(self.params, "max_validation_batches", 0),
            ):
                image = image.to(self.device)
                prior_shape = prior_shape.to(self.device)
                labels = labels.to(self.device)

                output = self.net(image, prior_shape)
                loss = self.perform_losses(labels, output)
                validation_losses.append(loss.item())
                validation_dices.append(self._prediction_dice(labels, output))

        return {
            "loss": float(np.mean(validation_losses)) if validation_losses else float("nan"),
            "dice": float(np.mean(validation_dices)) if validation_dices else float("nan"),
        }

    def _save_checkpoint(self, checkpoint_path, epoch, is_best):
        payload = {
            "epoch": epoch,
            "is_best": is_best,
            "model_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "params": self.params.to_dict() if hasattr(self.params, "to_dict") else {},
            "history": self.history,
        }
        torch.save(payload, checkpoint_path)

    def _write_history(self):
        with open(self.history_path, "w", encoding="utf-8") as handle:
            json.dump(self.history, handle, ensure_ascii=False, indent=2)

    def _load_best_checkpoint(self):
        if not os.path.isfile(self.best_checkpoint_path):
            raise FileNotFoundError(f"未找到最佳模型权重: {self.best_checkpoint_path}")
        checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state", checkpoint)
        self.net.load_state_dict(state_dict)

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
            for x, prior_shape, labels in self._iterate_subset(
                "test",
                max_batches=getattr(self.params, "max_test_batches", 0),
                progress=False,
            ):
                output = self.net(x.to(self.device), prior_shape.to(self.device))
                prediction = list(output)
                prediction[0] = (prediction[0] > self.params.threshold).int()
                test_dice.append(1.0 - dice_loss().np_loss(labels.to(self.device), prediction[0]))

        if not test_dice:
            raise RuntimeError("未执行任何测试集评估。")

        metrics = {
            "dice_mean": float(np.mean(test_dice)),
            "dice_std": float(np.std(test_dice)),
        }
        print(" - -" * 10)
        print(f" Test Dice: {metrics['dice_mean']} +/- {metrics['dice_std']} ")
        print(" - -" * 10)
        if getattr(self.params, "plot_predictions", True) and x is not None:
            self.view_prediction(x, labels, prior_shape, prediction)
        return metrics

    def view_prediction(self, x, labels, prior_shape, output):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = x[0, 0, :, :].cpu().numpy().astype(int)
        y = labels[0, 0, :, :].cpu().numpy().astype(int)
        y_hat = output[0][0, 0, :, :].cpu().numpy().astype(int)
        prior = prior_shape[0, 0, :, :].cpu().numpy().astype(int)

        figure_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(figure_dir, exist_ok=True)
        figure_path = os.path.join(figure_dir, "mnist_prediction.png")

        fig, axes = plt.subplots(ncols=3)
        for axis, seg, title, cmap in zip(
            axes,
            [y, prior, y_hat],
            ["Label", "Prior", "Prediction"],
            ["winter", "autumn", "summer"],
        ):
            axis.imshow(x, cmap="gray")
            masked = np.ma.masked_array(seg, seg == 0)
            axis.imshow(masked, cmap=cmap, alpha=0.6)
            axis.set_title(title)
            axis.axis("off")

        plt.savefig(figure_path, bbox_inches="tight")
        plt.close(fig)
