import os
import glob
import numpy as np
import nibabel as nib
import cv2
from skimage import measure
from tqdm import tqdm

# 引用 acdc_parameters.py 中的配置 (或者直接在此处硬编码以避免导入路径问题)
CROP_H, CROP_W = 144, 208  # 对应 acdc_parameters.py 中的 inshape
PRIOR_RADIUS = 35  # 对应 acdc_parameters.py 中的 ps_meas[0]
PRIOR_THICKNESS = 7  # 对应 acdc_parameters.py 中的 ps_meas[1]
MYO_LABEL = 2  # ACDC数据集中，2 代表心肌


def calculate_betti_numbers(binary_mask):
    """
    计算 Betti 数用于拓扑筛选。
    b0: 连通分量数, b1: 孔洞数
    我们需要 b0=1, b1=1 (标准的环形心肌)
    """
    binary_mask = (binary_mask > 0).astype(int)
    if np.sum(binary_mask) == 0:
        return 0, 0

    # 计算 b0 (连通分量)
    labeled_image, num_components = measure.label(binary_mask, return_num=True, connectivity=1)
    b0 = num_components

    # 计算欧拉示性数 chi = b0 - b1 -> b1 = b0 - chi
    chi = measure.euler_number(binary_mask, connectivity=1)
    b1 = b0 - chi
    return b0, b1


def create_prior(shape, radius, thickness):
    """
    生成 TEDS-Net 所需的 Prior Shape (圆环)
    """
    h, w = shape
    prior = np.zeros(shape, dtype=np.float32)
    center = (w // 2, h // 2)  # CV2 使用 (x, y)

    # 绘制圆环：厚度设为 thickness
    cv2.circle(prior, center, radius, 1, thickness)
    return prior


def crop_center(img, target_h, target_w):
    """中心裁剪"""
    h, w = img.shape
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2

    # 防止负索引
    if start_h < 0 or start_w < 0:
        # 如果图像比目标小，可能需要 Padding（这里简化处理，假设图像足够大）
        # 实际 ACDC 图像通常大于 144x208
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    return img[start_h:start_h + target_h, start_w:start_w + target_w]


def normalize_image(img):
    """Min-Max 归一化到 [0, 1]"""
    min_val = img.min()
    max_val = img.max()
    if max_val - min_val > 0:
        return (img - min_val) / (max_val - min_val)
    return img


def process_acdc(raw_data_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 获取所有 patient 文件夹
    patient_dirs = glob.glob(os.path.join(raw_data_path, "patient*"))

    print(f"Found {len(patient_dirs)} patients. Starting processing...")

    for patient_dir in tqdm(patient_dirs):
        patient_id = os.path.basename(patient_dir)

        # 查找该患者文件夹下的所有 nii.gz 文件
        # ACDC 命名规则: patientXXX_frameXX.nii.gz 和 patientXXX_frameXX_gt.nii.gz
        all_files = glob.glob(os.path.join(patient_dir, "*.nii.gz"))
        img_files = [f for f in all_files if "_gt" not in f and "4d" not in f]

        for img_path in img_files:
            # 构建对应的 GT 路径
            gt_path = img_path.replace(".nii.gz", "_gt.nii.gz")
            if not os.path.exists(gt_path):
                continue

            # 提取帧 ID (例如 frame01)
            frame_id = os.path.basename(img_path).split("_")[1].split(".")[0]

            # 加载数据
            img_obj = nib.load(img_path)
            gt_obj = nib.load(gt_path)
            img_vol = img_obj.get_fdata()
            gt_vol = gt_obj.get_fdata()

            # 遍历切片
            num_slices = img_vol.shape[2]
            for z in range(num_slices):
                img_slice = img_vol[:, :, z]
                gt_slice = gt_vol[:, :, z]

                # 提取心肌掩码 (Label 2)
                myo_mask = (gt_slice == MYO_LABEL).astype(np.uint8)

                # --- 拓扑筛选 ---
                b0, b1 = calculate_betti_numbers(myo_mask)
                # TEDS-Net 要求先验和数据的拓扑一致 (单连通，单孔)
                if b0 != 1 or b1 != 1:
                    continue

                    # --- 裁剪与归一化 ---
                img_crop = crop_center(img_slice, CROP_H, CROP_W)
                gt_crop = crop_center(gt_slice, CROP_H, CROP_W)  # 此时 GT 包含所有标签 (0,1,2,3)
                myo_crop = crop_center(myo_mask, CROP_H, CROP_W)  # 仅心肌

                img_norm = normalize_image(img_crop)

                # --- 生成 Prior ---
                # 注意：Prior 必须和裁剪后的图像尺寸一致
                prior = create_prior((CROP_H, CROP_W), PRIOR_RADIUS, PRIOR_THICKNESS)

                # --- 保存 ---
                save_name = f"{patient_id}_{frame_id}_slice{z:03d}.npz"
                np.savez(os.path.join(save_path, save_name),
                         image=img_norm,
                         label=gt_crop,
                         prior=prior)


if __name__ == "__main__":
    # 请根据你的实际路径修改这里
    # 根据你的截图，你的数据似乎在 TEDS-Net/Resources/database/training
    RAW_PATH = "./Resources/database/training"
    SAVE_PATH = "./Resources/database/processed_2d"

    process_acdc(RAW_PATH, SAVE_PATH)