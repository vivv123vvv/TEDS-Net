import os
import numpy as np
import raster_geometry as rg
import torch
from torch.utils.data import Dataset
from glob import glob
import nibabel as nib


class ACDC_dataclass(Dataset):
    ''' ACDC Dataclass

    Using the ACDC dataset, available at: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
    This task only using the myocaridum segmentation (label 2)

    NEW USERS TO-DO:
    1) Add list of IDS Path
    2) Check datapath stored in Parameter file: e.g. params.data_path

    '''

    def __init__(self,
                 params,
                 subset,

                 ):
        self.params = params

        assert subset in ['Train', 'Test']

        # 根据子集获取对应的患者数据
        if subset == 'Train':
            data_dir = os.path.join(params.dataset.datapath, "training")
        else:
            # 检查测试目录是否存在，如果不存在则使用训练数据的一部分
            test_dir = os.path.join(params.dataset.datapath, "testing")
            if os.path.exists(test_dir):
                data_dir = test_dir
            else:
                data_dir = os.path.join(params.dataset.datapath, "training")

        # 获取所有患者文件夹
        patient_dirs = glob(os.path.join(data_dir, "patient*"))
        print(f"数据目录: {data_dir}")
        print(f"找到的患者目录数量: {len(patient_dirs)}")
        self.list_IDS = [os.path.basename(p) for p in patient_dirs]
        
        # 如果是测试子集且测试目录不存在，则使用训练数据的前5个患者作为测试数据
        if subset == 'Test' and not os.path.exists(os.path.join(params.dataset.datapath, "testing")):
            self.list_IDS = self.list_IDS[:5]  # 使用前5个患者作为测试数据
        elif subset == 'Train' and not os.path.exists(os.path.join(params.dataset.datapath, "testing")):
            self.list_IDS = self.list_IDS[5:]  # 剩余的作为训练数据
        
        print(f"{subset} 子集的患者数量: {len(self.list_IDS)}")
        self.subset = subset

        # --- Generate Prior Shape ---
        rad, thick = params.dataset.ps_meas
        M, N = params.dataset.inshape
        self.prior = rg.circle((M, N), radius=rad).astype(int) - rg.circle((M, N), radius=(rad - thick)).astype(int)

    def __len__(self):
        # Return the volumes in that data subet
        return len(self.list_IDS)

    def __getitem__(self, idx):

        ID = self.list_IDS[idx]

        # 构建数据路径
        if self.subset == 'Train' or not os.path.exists(os.path.join(self.params.dataset.datapath, "testing")):
            patient_path = os.path.join(self.params.dataset.datapath, "training", ID)
        else:
            patient_path = os.path.join(self.params.dataset.datapath, "testing", ID)

        # 获取所有nii.gz文件
        nii_files = glob(os.path.join(patient_path, "*.nii.gz"))

        # 排除4D文件，只处理帧文件
        frame_files = [f for f in nii_files if "_4d.nii.gz" not in f]

        if not frame_files:
            raise FileNotFoundError(f"No valid image files found for patient {ID}")

        # 分离图像文件和标签文件
        image_files = [f for f in frame_files if "_gt" not in f]
        gt_files = [f for f in frame_files if "_gt" in f]

        if not image_files:
            raise FileNotFoundError(f"No image file found for patient {ID}")

        # 选择第一个图像文件
        image_file = image_files[0]
        
        # 尝试找到对应的标签文件
        corresponding_gt = image_file.replace(".nii.gz", "_gt.nii.gz")
        if corresponding_gt in gt_files:
            gt_file = corresponding_gt
        elif gt_files:
            gt_file = gt_files[0]
        else:
            # 如果没有标签文件，使用图像文件作为替代（仅用于测试集）
            gt_file = image_file

        # Load in volume and segmentation:
        img_nii = nib.load(image_file)
        x = img_nii.get_fdata()

        seg_nii = nib.load(gt_file)
        y_seg = seg_nii.get_fdata()

        # 处理4D数据，如果存在
        if len(x.shape) == 4:
            x = x[:, :, :, 0]  # 取第一帧
        if len(y_seg.shape) == 4:
            y_seg = y_seg[:, :, :, 0]  # 取第一帧

        # 选择中间切片
        mid_slice = x.shape[2] // 2
        x = x[:, :, mid_slice]
        y_seg = y_seg[:, :, mid_slice]

        # 只保留心肌标签（标签2），如果存在标签文件
        if image_file != gt_file:  # 有独立的标签文件
            y_seg = (y_seg == 2).astype(np.float32)  # 心肌标签为2
        else:  # 没有独立标签文件，测试集情况
            y_seg = np.zeros_like(y_seg, dtype=np.float32)

        # 调整图像尺寸以匹配预期输入尺寸
        target_shape = tuple(self.params.dataset.inshape)
        if x.shape != target_shape:
            # 使用线性插值调整图像尺寸
            x_resized = np.zeros(target_shape, dtype=np.float32)
            y_seg_resized = np.zeros(target_shape, dtype=np.float32)

            # 简单的重采样方法（最近邻）
            x_ratio = x.shape[0] / target_shape[0]
            y_ratio = x.shape[1] / target_shape[1]

            for i in range(target_shape[0]):
                for j in range(target_shape[1]):
                    x_src = min(int(i * x_ratio), x.shape[0] - 1)
                    y_src = min(int(j * y_ratio), x.shape[1] - 1)
                    x_resized[i, j] = x[x_src, y_src]
                    y_seg_resized[i, j] = y_seg[x_src, y_src]

            x = x_resized
            y_seg = y_seg_resized

        # Get Data in Torch Convention:
        x = np.expand_dims(x, axis=0).copy()  # 添加.copy()确保数组是可写的
        x = torch.from_numpy(x.astype(np.float32))
        y_seg = np.expand_dims(y_seg, axis=0).copy()  # 添加.copy()确保数组是可写的
        y_seg = torch.from_numpy(y_seg.astype(np.float32))
        prior_shape = np.expand_dims(self.prior, 0).copy()  # 添加.copy()确保数组是可写的
        prior_shape = torch.from_numpy(prior_shape.astype(np.float32))

        return x, prior_shape, y_seg