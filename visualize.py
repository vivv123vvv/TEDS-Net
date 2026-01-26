import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

# 导入你的模块
from parameters.acdc_parameters import Parameters
from network.TEDS_Net import TEDS_Net
from dataloaders.acdc_npz import ACDCNpzDataset


def visualize():
    # ---------------------------------------------------------
    # 1. 配置
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # 模型权重路径 (请确保这个文件存在)
    model_path = "checkpoints/best_teds_net.pth"
    if not os.path.exists(model_path):
        # 如果没有 best, 尝试找 epoch_xx
        print(f"⚠️ 找不到 {model_path}，尝试查找最近的 checkpoint...")
        # 这里你可以手动修改为你有的 .pth 文件名
        # model_path = "checkpoints/teds_net_epoch_10.pth"
        return

    # 数据路径
    data_dir = os.path.join("Resources", "database", "processed_2d")

    # ---------------------------------------------------------
    # 2. 加载数据 & 模型
    # ---------------------------------------------------------
    params = Parameters()

    # 准备验证集数据
    full_dataset = ACDCNpzDataset(data_dir)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    _, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 取一个 Batch 进行可视化
    val_loader = DataLoader(val_set, batch_size=4, shuffle=True)  # 一次看4张图

    # 初始化模型
    model = TEDS_Net(params).to(device)

    # 加载权重
    print(f"📥 正在加载模型权重: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # ---------------------------------------------------------
    # 3. 预测与绘图
    # ---------------------------------------------------------
    # 获取一批数据
    images, priors, labels = next(iter(val_loader))
    images, priors, labels = images.to(device), priors.to(device), labels.to(device)

    with torch.no_grad():
        # 模型预测
        outputs = model(images, priors)
        # 根据你的 forward 返回值，通常第一个是最终分割图
        if isinstance(outputs, tuple):
            pred_seg = outputs[0]
        else:
            pred_seg = outputs

    # 转回 CPU 方便绘图
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    pred_seg = pred_seg.cpu().numpy()
    priors = priors.cpu().numpy()

    # 开始绘图
    batch_size = images.shape[0]
    fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))

    # 如果只有1张图，axes不是列表，需要处理一下
    if batch_size == 1:
        axes = [axes]

    for i in range(batch_size):
        # 1. 原始图像
        ax_img = axes[i][0]
        ax_img.imshow(images[i, 0], cmap='gray')
        ax_img.set_title("Input Image")
        ax_img.axis('off')

        # 2. 初始先验 (Prior)
        ax_prior = axes[i][1]
        ax_prior.imshow(images[i, 0], cmap='gray')
        # 叠加先验轮廓 (蓝色)
        ax_prior.contour(priors[i, 0], levels=[0.5], colors='blue', linewidths=2)
        ax_prior.set_title("Initial Prior (Blue)")
        ax_prior.axis('off')

        # 3. 预测结果 (Prediction)
        ax_pred = axes[i][2]
        ax_pred.imshow(images[i, 0], cmap='gray')
        # 叠加预测轮廓 (红色)
        # 预测值通常是 0-1 之间的概率，取 0.5 为阈值
        ax_pred.contour(pred_seg[i, 0], levels=[0.5], colors='red', linewidths=2)
        ax_pred.set_title("Prediction (Red)")
        ax_pred.axis('off')

        # 4. 金标准对比 (GT vs Pred)
        ax_gt = axes[i][3]
        ax_gt.imshow(images[i, 0], cmap='gray')
        # GT (绿色) vs 预测 (红色)
        ax_gt.contour(labels[i, 0], levels=[0.5], colors='lime', linewidths=2, linestyles='--')
        ax_gt.contour(pred_seg[i, 0], levels=[0.5], colors='red', linewidths=2)
        ax_gt.set_title("GT (Green) vs Pred (Red)")
        ax_gt.axis('off')

    plt.tight_layout()
    # 保存图片
    save_path = "visualization_result.png"
    plt.savefig(save_path)
    print(f"✅ 可视化结果已保存为: {save_path}")

    # 如果你在本地运行支持 GUI，可以取消下面这行的注释
    # plt.show()


if __name__ == "__main__":
    visualize()