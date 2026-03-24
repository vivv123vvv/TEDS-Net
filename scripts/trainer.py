import os

import numpy as np
import torch
from tqdm import tqdm

from utils.losses import dice_loss, grad_loss


class Trainer:

    def __init__(self, params, device, net):

        """初始化训练与评估流程。"""

        self.params = params
        self.device = device
        self.net = net

        # 加载数据
        self.get_dataloader()
        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.params.lr)

    def _iterate_subset(self, subset, max_batches=0, progress=True):
        """按需截断 DataLoader，便于服务器 smoke test。"""

        iterator = self.dataloader_dic[subset]
        if progress:
            iterator = tqdm(iterator)

        for batch_index, batch in enumerate(iterator):
            if max_batches and batch_index >= max_batches:
                break
            yield batch

    def dothetraining(self):

        # 遍历所有训练 epoch
        for epoch in range(self.params.epoch):

            # 执行训练阶段
            self.net.train()
            self.epoch_loss = []
            for (x, prior_shape, labels) in self._iterate_subset(
                'train',
                max_batches=getattr(self.params, 'max_train_batches', 0),
            ):

                with torch.set_grad_enabled(True):
                    self.optimizer.zero_grad()

                    # 将输入送入网络
                    output = self.net(x.to(self.device), prior_shape.to(self.device))

                    # 计算损失
                    loss = self.perform_losses(labels.to(self.device), output)

                    # 更新模型参数
                    loss.backward()
                    self.optimizer.step()

            # 统计当前 epoch 的训练损失
            train_loss = float(np.mean(self.epoch_loss)) if self.epoch_loss else float('nan')

            # 执行验证阶段
            val_loss = self.do_validation()

            # 打印损失
            print("[{0}] {1}: {2:.6f}".format(epoch, 'training_loss', train_loss))
            print("[{0}] {1}: {2:.6f}".format(epoch, 'validation_loss', val_loss))

    def do_validation(self):
        """对当前模型执行逐 batch 验证。"""

        self.net.eval()
        self.epoch_loss = []
        for (x, prior_shape, labels) in self._iterate_subset(
            'validation',
            max_batches=getattr(self.params, 'max_validation_batches', 0),
        ):

            with torch.set_grad_enabled(False):

                # 将输入送入网络
                output = self.net(x.to(self.device), prior_shape.to(self.device))

                # 计算损失
                self.perform_losses(labels.to(self.device), output)

        return float(np.mean(self.epoch_loss)) if self.epoch_loss else float('nan')

    def get_dataloader(self):
        """加载当前任务所需的数据加载器。"""

        if self.params.data == "mnist":
            from dataloaders.setup import setup_mnist_dataloader as setup_dataloader
        elif self.params.data == "ACDC":
            from dataloaders.setup import setup_acdc_dataloader as setup_dataloader
        else:
            raise ValueError(f"不支持的数据集: {self.params.data}")

        self.dataloader_dic = setup_dataloader(self.params, ['train', 'validation', 'test'])

    def perform_losses(self, labels, output):
        """计算并汇总所有损失项。"""

        loss = 0
        for i, (loss_function, w) in enumerate(zip(self.params.loss_params.loss, self.params.loss_params.weight)):

            # 根据配置依次计算 Dice 与形变正则项
            if loss_function == "dice":
                curr_loss = dice_loss().loss(labels, output[i], loss_mult=w)
            elif "grad" in loss_function:
                curr_loss = grad_loss(self.params).loss(labels, output[i], loss_mult=w)
            else:
                raise ValueError(f"不支持的损失函数: {loss_function}")

            # 累加当前损失
            loss += curr_loss

        # 记录当前 batch 的损失
        self.epoch_loss.append(loss.item())

        return loss

    def do_evalutation(self):
        """评估训练后模型，并计算平均 Dice 指标。"""

        self.net.eval()
        test_dice = []
        x = labels = prior_shape = output = None
        for (x, prior_shape, labels) in self._iterate_subset(
            'test',
            max_batches=getattr(self.params, 'max_test_batches', 0),
            progress=False,
        ):

            # 前向推理
            with torch.no_grad():
                output = self.net(x.to(self.device), prior_shape.to(self.device))
                output = list(output)
                output[0] = (output[0] > self.params.threshold).int()

            # 计算 Dice 损失
            test_dice.append(dice_loss().np_loss(labels.to(self.device), output[0]))

        if not test_dice:
            print("未执行测试集评估：当前测试批次数为 0。")
            return

        print(" - -" * 10)
        print(f" Test Dice Loss: {1 - np.mean(test_dice)} +/- {np.std(test_dice)} ")
        print(" - -" * 10)
        if getattr(self.params, 'plot_predictions', True):
            self.ViewPrediction(x, labels, prior_shape, output)

    def ViewPrediction(self, x, labels, prior_shape, output):
        """可视化一组测试样本的标签、先验与预测结果。"""

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 取单个样本做展示
        x = x[0, 0, :, :].cpu().numpy().astype(int)  # 图像
        y = labels[0, 0, :, :].cpu().numpy().astype(int)  # 真实标注
        y_hat = output[0][0, 0, :, :].cpu().numpy().astype(int)  # 预测结果
        p = prior_shape[0, 0, :, :].cpu().numpy().astype(int)  # 先验形状

        # 绘制可视化图像
        fig, ax = plt.subplots(ncols=3)
        cmaps = 'winter', 'autumn', 'summer'
        for a, seg, title, cmap in zip(ax, [y, p, y_hat], ['Label', 'Prior', "Prediction"], cmaps):
            a.imshow(x, cmap='gray')
            mask_lab = np.ma.masked_array(seg, seg == 0)  # 掩掉背景区域
            a.imshow(mask_lab, cmap=cmap, alpha=0.6)
            a.set_title(title)
            a.axis('off')

        os.makedirs(self.params.data_path, exist_ok=True)
        plt.savefig(os.path.join(self.params.data_path, 'figure.png'), bbox_inches='tight')
        plt.close(fig)
