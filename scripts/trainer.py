import torch
import os
import numpy as np
from tqdm import tqdm
from utils.losses import dice_loss, grad_loss


class Trainer:

    def __init__(self, params, device, net):

        """初始化训练与评估流程。

        参数:
            params (dict): 训练参数配置。
            device (device): 模型训练所使用的设备。
            net (class): TEDS-Net 网络实例。
        """

        self.params = params
        self.device = device
        self.net = net

        # 加载数据
        self.get_dataloader()
        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.params.lr)

    def dothetraining(self):

        # 遍历所有训练 epoch
        for epoch in range(self.params.epoch):

            # 执行训练阶段
            self.net.train()
            subset = 'train'
            self.epoch_loss = []
            for (x, prior_shape, labels) in tqdm(self.dataloader_dic[subset]):

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
            train_loss = np.mean(self.epoch_loss)

            # 执行验证阶段
            val_loss = self.do_validation(epoch)

            # 打印损失
            print("[{0}] {1}: {2:.6f}".format(epoch, 'training_loss', train_loss))
            print("[{0}] {1}: {2:.6f}".format(epoch, 'validation_loss', val_loss))

    def do_validation(self, epoch):
        """对当前模型执行逐 batch 验证。"""
        self.net.eval()
        self.epoch_loss = []
        subset = 'validation'
        for (x, prior_shape, labels) in tqdm(self.dataloader_dic[subset]):

            with torch.set_grad_enabled(False):

                # 将输入送入网络
                output = self.net(x.to(self.device), prior_shape.to(self.device))

                # 计算损失
                self.perform_losses(labels.to(self.device), output)

        return np.mean(self.epoch_loss)

    def get_dataloader(self):
        """加载当前任务所需的数据加载器。"""

        if self.params.data == "mnist":
            from dataloaders.setup import setup_mnist_dataloader as setup_dataloader
        elif self.params.data == "ACDC":
            from dataloaders.setup import setup_acdc_dataloader as setup_dataloader

        self.dataloader_dic = setup_dataloader(self.params, ['train', 'validation', 'test'])

    def perform_losses(self, labels, output):
        """计算并汇总所有损失项。

        参数:
            labels (tensor): 真实标注。
            output (list): `output[0]` 为预测结果，`output[1]` 为 bulk 场，
                `output[2]` 为 fine-tune 场。
        """

        loss = 0
        for i, (loss_function, w) in enumerate(zip(self.params.loss_params.loss, self.params.loss_params.weight)):

            # 根据配置依次计算 Dice 与形变正则项
            if loss_function == "dice":
                curr_loss = dice_loss().loss(labels, output[i], loss_mult=w)
            elif "grad" in loss_function:
                curr_loss = grad_loss(self.params).loss(labels, output[i], loss_mult=w)

            # 累加当前损失
            loss += curr_loss

        # 记录当前 batch 的损失
        self.epoch_loss.append(loss.item())

        return loss

    def do_evalutation(self):
        """评估训练后模型，并计算平均 Dice 指标。"""

        self.params.batch = 1
        test_dice = []
        for (x, prior_shape, labels) in self.dataloader_dic['test']:

            # 前向推理
            output = self.net(x.to(self.device), prior_shape.to(self.device))
            output = list(output)
            output[0] = (output[0] > self.params.threshold).int()

            # 计算 Dice 损失
            test_dice.append(dice_loss().np_loss(labels.to(self.device), output[0]))

        print(" - -" * 10)
        print(f" Test Dice Loss: {1 - np.mean(test_dice)} +/- {np.std(test_dice)} ")
        print(" - -" * 10)
        self.ViewPrediction(x, labels, prior_shape, output)

    def ViewPrediction(self, x, labels, prior_shape, output):
        """可视化一组测试样本的标签、先验与预测结果。"""
        import matplotlib.pyplot as plt

        # 取单个样本做展示
        x = x[0, 0, :, :].cpu().numpy().astype(int)  # 图像
        y = labels[0, 0, :, :].cpu().numpy().astype(int)  # 真实标注
        y_hat = output[0][0, 0, :, :].cpu().numpy().astype(int)  # 预测结果
        p = prior_shape[0, 0, :, :].cpu().numpy().astype(int)  # 先验形状

        # 绘制可视化图像
        fig, ax = plt.subplots(ncols=3)
        cmaps = 'winter', 'autumn', 'summer'
        for i, (a, seg, t) in enumerate(zip(ax, [y, p, y_hat], ['Label', 'Prior', "Prediction"])):
            a.imshow(x, cmap='gray')
            mask_lab = np.ma.masked_array(seg, seg == 0)  # 掩掉背景区域
            a.imshow(mask_lab, cmap=cmaps[i], alpha=0.6)
            a.set_title(t)
            a.axis('off')

        plt.savefig(os.path.join(self.params.data_path, 'figure'))
