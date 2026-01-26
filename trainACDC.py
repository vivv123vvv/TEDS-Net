import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# --- 导入自定义模块 (基于你的目录结构) ---
from parameters.acdc_parameters import Parameters
from network.TEDS_Net import TEDS_Net
from dataloaders.acdc_npz import ACDCNpzDataset
from utils.losses import dice_loss, grad_loss


def train():
    # ---------------------------------------------------------
    # 1. 配置与初始化
    # ---------------------------------------------------------
    # 检查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在使用计算设备: {device}")

    # 加载参数
    params = Parameters()

    # 强制覆盖数据路径 (确保指向你的 .npz 文件夹)
    # 根据你的截图，路径应该是 Resources/database/processed_2d
    # 如果你的预处理数据在别的地方，请修改这里
    data_dir = os.path.join("Resources", "database", "processed_2d")

    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"❌ 错误: 找不到数据目录: {data_dir}")
        print("请检查 preprocess_acdc.py 是否运行成功，或者修改 trainACDC.py 中的 data_dir 路径。")
        return

    # ---------------------------------------------------------
    # 2. 准备数据 (Data Pipeline)
    # ---------------------------------------------------------
    print(f"📂 正在加载数据从: {data_dir} ...")
    full_dataset = ACDCNpzDataset(data_dir)

    # 划分训练集和验证集 (80% 训练, 20% 验证)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=params.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=params.batch, shuffle=False, num_workers=0)

    print(f"✅ 数据加载完毕: 训练集 {len(train_set)} 张, 验证集 {len(val_set)} 张")

    # ---------------------------------------------------------
    # 3. 初始化模型与优化器
    # ---------------------------------------------------------
    model = TEDS_Net(params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.lr)

    # 初始化损失函数
    calc_dice = dice_loss()
    calc_grad = grad_loss(params)

    # ---------------------------------------------------------
    # 4. 训练循环
    # ---------------------------------------------------------
    print("🔥 开始训练...")

    best_val_loss = float('inf')

    for epoch in range(params.epoch):
        model.train()
        epoch_loss = 0.0

        # 使用 tqdm 显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{params.epoch}")

        for image, prior, label in pbar:
            # 数据移至 GPU
            image = image.to(device)
            prior = prior.to(device)
            label = label.to(device)  # 形状通常为 [B, 1, H, W]

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播 (Forward)
            # TEDS_Net 返回: (ft_sampled, flow_bulk_upsamp, flow_ft_upsamp)
            outputs = model(image, prior)

            # 解析输出
            # 根据 TEDS_Net.py 的逻辑 (no_branches == 2)
            if len(outputs) == 3:
                pred_seg, flow_bulk, flow_ft = outputs

                # 计算平滑度损失 (Gradient Loss)
                loss_reg = calc_grad.loss(None, flow_bulk) + calc_grad.loss(None, flow_ft)
            else:
                # 单分支情况 (备用)
                pred_seg, flow = outputs
                loss_reg = calc_grad.loss(None, flow)

            # 计算分割损失 (Dice Loss)
            # pred_seg 已经是变形后的 Prior，需要和 Ground Truth (label) 比较
            loss_dice = calc_dice.loss(label, pred_seg)

            # 总损失 (加权求和)
            # 权重参考 acdc_parameters.py: dice=1, grad=10000
            total_loss = 1.0 * loss_dice + 10000.0 * loss_reg

            # 反向传播与更新
            total_loss.backward()
            optimizer.step()

            # 记录损失
            epoch_loss += total_loss.item()

            # 更新进度条显示
            pbar.set_postfix({'Loss': total_loss.item(), 'Dice': (1 - loss_dice.item())})

        # --- 每个 Epoch 结束后进行验证 ---
        avg_train_loss = epoch_loss / len(train_loader)

        # 简单验证过程
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v_img, v_prior, v_lbl in val_loader:
                v_img, v_prior, v_lbl = v_img.to(device), v_prior.to(device), v_lbl.to(device)

                v_out = model(v_img, v_prior)
                v_pred = v_out[0] if isinstance(v_out, tuple) else v_out

                # 只计算 Dice Loss 作为验证指标
                val_loss += calc_dice.loss(v_lbl, v_pred).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = 1.0 - avg_val_loss  # 近似 Dice 分数

        print(f"📊 Epoch {epoch + 1} 结束 | Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

        # --- 保存模型 ---
        # 如果是最佳模型则保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_teds_net.pth")
            print("💾 最佳模型已保存!")

        # 定期保存 (例如每10轮)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/teds_net_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    train()