
import torch
import numpy as np


class dice_loss:
    """Dice 损失函数。"""

    def loss(self, y_true, y_pred, loss_mult=None):
        smooth = 1.
        iflat = y_pred.view(-1)
        tflat = y_true.view(-1)

        intersection = (iflat * tflat).sum()
        dice = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

        if loss_mult is not None:
            dice *= loss_mult

        return dice

    def np_loss(self, y_true, y_pred, loss_mult=None):

        return self.loss(y_true, y_pred, loss_mult).item()


class grad_loss:
    """梯度正则损失。

    使用绝对差分形式的梯度损失，并将网格项纳入计算。
    参考实现改编自：https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, params, penalty='l2'):

        self.penalty = penalty
        self.ndims = params.dataset.ndims

    def loss(self, _, y_pred, loss_mult=None):

        """
        使用 PyTorch grid 的坐标格式，即取值范围通常位于 -1 到 1 之间。
        """

        size = np.shape(y_pred)[2:]
        vectors = [torch.linspace(-1, 1, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to('cuda:0')

        flow_feild = torch.zeros(y_pred.size(), device='cuda:0')
        for i in range(y_pred.size()[0]):
            flow_feild[i] = y_pred[i] + grid

        dy = torch.abs(flow_feild[:, :, 1:, :] - flow_feild[:, :, :-1, :])
        dx = torch.abs(flow_feild[:, :, :, 1:] - flow_feild[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if loss_mult is not None:
            grad *= loss_mult

        return grad

    def np_loss(self, _, y_pred, loss_mult=None):

        return self.loss(_, y_pred, loss_mult).item()
