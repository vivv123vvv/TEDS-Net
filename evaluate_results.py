import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

# 引入你的模型和数据集类
from network.TEDS_Net import TEDS_Net


# 假设你有一个 dataset.py 或类似的数据加载器
# from dataloader import ACDC_Dataset
# 暂时用伪代码代替数据加载，你需要替换成你 trainACDC.py 里用的那个 dataset
import torch
import numpy as np


# --- 缺失的辅助函数 START ---

def dice_score(pred, target):
    """
    计算 Dice 系数
    pred, target: [Batch, H, W] (0/1 binary masks or probabilities)
    """
    smooth = 1e-5

    # 展平
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()

    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def compute_jacobian_determinant_2d(flow):
    """
    计算 2D 变形场的 Jacobian 行列式。
    flow shape: [Batch, 2, H, W]
    """
    B, C, H, W = flow.size()
    # 转换为 Numpy 处理梯度比较方便
    flow_np = flow.detach().cpu().numpy()
    jacobians = []

    for b in range(B):
        # flow[:, 0] 是 y 方向位移, flow[:, 1] 是 x 方向位移
        u_b = flow_np[b, 0]
        v_b = flow_np[b, 1]

        # np.gradient 返回 [gradient_axis_0 (y), gradient_axis_1 (x)]
        dy_u, dx_u = np.gradient(u_b)
        dy_v, dx_v = np.gradient(v_b)

        # Determinant |J| = (1 + du/dx)(1 + dv/dy) - (du/dy)(dv/dx)
        det_J = (1 + dx_u) * (1 + dy_v) - (dy_u * dx_v)
        jacobians.append(det_J)

    return np.array(jacobians)


def count_folding_pixels(jacobian_det):
    """
    统计 Jacobian < 0 的像素比例
    """
    num_negative = np.sum(jacobian_det < 0)
    total_pixels = jacobian_det.size
    ratio = num_negative / total_pixels
    return ratio, num_negative


# --- 缺失的辅助函数 END ---
# --- 1. 配置参数 ---
class Params:
    def __init__(self):
        # 必须与训练时的参数严格保持一致
        self.network_params = type('obj', (object,), {
            'in_chan': 1,
            'out_chan': 1,
            'fi': 12,  # <--- 这里必须改成 12 (原来是 32)
            'net_depth': 4,  # 确认深度也是 4
            'dropout': 0.0
        })

        self.network = type('obj', (object,), {
            # 这里的 dec_depth 也要和训练时一致，ACDC通常是 [4] 或 [4, 2] (如果用了两分支)
            # 如果你的模型有两分支 (STN_bulk 和 STN_ft)，这里应该是 [4, 2] 之类
            # 如果是单分支，则是 [4]
            # 根据 TEDS-Net 默认配置，ACDC 往往是双分支
            'dec_depth': [4, 2],  # 请确认这里是否需要改为 [4, 2] ?
            'diffeo_int': 7,
            'guas_smooth': False,
            'Guas_kernel': 3,
            'sigma': 1,
            'act': 'tanh',
            'mega_P': 1
        })
        self.dataset = type('obj', (object,), {
            'ndims': 2,
            'inshape': [288, 416]  # 确认尺寸是否正确
        })


def evaluate(model_path, test_loader, device='cuda'):
    # 初始化参数
    params = Params()

    # --- 2. 加载模型 ---
    print(f"Loading model from {model_path}...")
    model = TEDS_Net(params).to(device)

    # 加载权重 (处理可能的 'module.' 前缀)
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)  # strict=False 以防有些无关层不匹配
    model.eval()

    total_dice = []
    total_folding_ratio = []

    print("Starting Evaluation...")

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            images = batch[0].float().to(device)
            priors = batch[1].float().to(device)
            gt_masks = batch[2].float().to(device)

            # 前向传播
            # returns: flow_field, flow_upsamp, sampled_prior
            # --- 修复代码开始 ---
            outputs = model(images, priors)

            if len(outputs) == 2:
                # 单分支模式 (Single Branch): 返回 (预测图, 形变场)
                pred_warped, flow_upsamp = outputs
            elif len(outputs) == 3:
                # 双分支模式 (Dual Branch): 返回 (精细预测图, 粗糙场, 精细场)
                # 我们通常需要精细预测图 (Index 0) 和 精细形变场 (Index 2)
                pred_warped, _, flow_upsamp = outputs
            else:
                raise ValueError(f"Unexpected number of outputs from model: {len(outputs)}")
            # --- 修复代码结束 ---

            # --- A. 计算 Dice ---
            # 对 pred_warped 进行二值化 (通常 threshold=0.5)
            pred_mask = (pred_warped > 0.5).float()

            # 如果 GT 是 Channel-first [B, 1, H, W], 去掉 Channel 维
            if len(gt_masks.shape) == 4:
                gt_masks = gt_masks.squeeze(1)
            if len(pred_mask.shape) == 4:
                pred_mask = pred_mask.squeeze(1)

                # --- 尺寸对齐修复 START ---
                # pred_mask 的尺寸通常是 [B, 1, H, W] 或 [B, H, W]
                # gt_masks 的尺寸通常是 [B, H, W]

                # 1. 确保维度一致 (都变成 4D: B, C, H, W) 以便插值
                if len(pred_mask.shape) == 3:
                    pred_mask = pred_mask.unsqueeze(1)
                if len(gt_masks.shape) == 3:
                    gt_masks = gt_masks.unsqueeze(1)

                # 2. 检查尺寸是否匹配
                if pred_mask.shape[-2:] != gt_masks.shape[-2:]:
                    # 如果不匹配，强制将 GT (标签) 缩放到 预测图 (Pred) 的尺寸
                    # 使用 'nearest' 最近邻插值，因为标签是 0/1 整数，不能由小数
                    gt_masks = torch.nn.functional.interpolate(
                        gt_masks,
                        size=pred_mask.shape[-2:],
                        mode='nearest'
                    )

                # --- 尺寸对齐修复 END ---

                # 现在可以安全计算 Dice 了
                batch_dice = dice_score(pred_mask, gt_masks)
            total_dice.append(batch_dice.item())

            # --- B. 计算拓扑折叠率 (Jacobian) ---
            # flow_upsamp 是 [B, 2, H, W]
            # 重要：Jacobian 计算可能对数值范围敏感。
            # 如果 flow 是归一化坐标 (-1~1)，需要 rescale 到像素坐标才能计算准确的物理折叠
            # 如果 flow 是 TEDS-Net 输出的 pixel displacement，直接用即可。
            # 假设是 pixel displacement (大部分网络输出):
            j_det = compute_jacobian_determinant_2d(flow_upsamp)

            ratio, _ = count_folding_pixels(j_det)
            total_folding_ratio.append(ratio)

    # --- 3. 打印统计结果 ---
    avg_dice = np.mean(total_dice)
    avg_folding = np.mean(total_folding_ratio)

    print("\n" + "=" * 30)
    print("📊 Evaluation Report")
    print("=" * 30)
    print(f"✅ Average Test Dice:       {avg_dice:.4f}")
    print(f"🥨 Average Folding Ratio:   {avg_folding * 100:.6f} % (Jacobian < 0)")
    print("=" * 30)

    if avg_folding < 0.001:
        print("🌟 拓扑保持性极佳！(R2Net 模块起作用了)")
    elif avg_folding < 0.01:
        print("⚠️ 存在少量折叠，可能在边界处。")
    else:
        print("❌ 折叠较多，请检查 Lipschitz 约束 (Scaling) 是否生效。")


if __name__ == '__main__':
    # 配置这里
    MODEL_PATH = 'checkpoints/best_teds_net.pth'  # 修改为你的模型路径
    from dataloaders.acdc_npz import ACDCNpzDataset

    # 3. 实例化数据集 (参考 trainACDC.py 里的参数)
    val_dataset = ACDCNpzDataset(
        data_dir="Resources/database/processed_2d",
        mode='val'  # 或者 'test'
    )

    # 4. 创建 Loader
    test_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,  # 评估通常用 batch_size=1
        shuffle=False,  # 千万不要 shuffle
        num_workers=0  # Windows下如果报错可以设为0
    )

    # 5. 运行评估
    evaluate(MODEL_PATH, test_loader)