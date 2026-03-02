import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class LC_ResNet_Block_KerasStyle(nn.Module):
    """
    基于你提供的 network.py (Keras版) 复现的 PyTorch 版 Block
    结构: ConvSN(3x3) -> LeakyReLU -> ConvSN(1x1) -> Tanh -> Scale -> Add
    """

    def __init__(self, channels):
        super(LC_ResNet_Block_KerasStyle, self).__init__()

        # 1. 第一层: 3x3x3 卷积 + 谱归一化
        self.conv1 = spectral_norm(nn.Conv3d(channels, channels, kernel_size=3, padding=1, stride=1, bias=False))
        self.relu = nn.LeakyReLU(0.2)

        # 2. 第二层: 1x1x1 卷积 + 谱归一化 (参考 network.py 中的 kernel_size=1)
        self.conv2 = spectral_norm(nn.Conv3d(channels, channels, kernel_size=1, padding=0, stride=1, bias=False))
        self.tanh = nn.Tanh()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)

        residual = self.conv2(residual)
        residual = self.tanh(residual)

        # 3. Scale: network.py 中使用了 Lambda(lambda x: x / 2.0)
        residual = residual / 2.0

        # 4. Add: v_next = v_current + residual
        return x + residual


class R2Net_Integrator(nn.Module):
    """
    R2Net 积分器：包含初始缩放和7个残差块
    """

    def __init__(self, channels, n_blocks=7):
        super(R2Net_Integrator, self).__init__()

        # 定义7个堆叠的块 (对应 out1 到 out7)
        self.blocks = nn.ModuleList([
            LC_ResNet_Block_KerasStyle(channels) for _ in range(n_blocks)
        ])

    def forward(self, flow0):
        # 对应 network.py 中的: flow0 = Lambda(lambda x: x / 2.0)(flow0)
        # 初始速度场先除以2
        v = flow0 / 2.0

        # 依次通过7个残差块
        for block in self.blocks:
            v = block(v)

        return v