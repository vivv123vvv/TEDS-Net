import torch
import argparse
from trainer import Trainer


class Train_Runner:

    def __init__(self, args):

        # 1) 初始化参数
        self.setup_params(args)

        # 2) 选择设备
        device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")

        # 3) 构建模型
        from network.TEDS_Net import TEDS_Net as net
        net = net(self.params)
        net.to(device)

        # 4) 训练并评估模型
        trainer = Trainer(self.params, device, net)
        trainer.dothetraining()
        trainer.do_evalutation()

    def setup_params(self, args):
        """根据命令行参数加载默认训练配置。

        参数:
            args (module): 训练运行时传入的设置。
        """

        if args.dataset == "mnist":
            from parameters.mnist_parameters import Parameters
        elif args.dataset == "ACDC":
            from parameters.acdc_parameters import Parameters

        self.params = Parameters.from_dict({'data': args.dataset})


if __name__ == '__main__':
    """TEDS-Net 训练入口。

    ACDC 心肌分割相关的论文配置保存在 `parameters/acdc_parameters.py` 中，
    先验形状生成逻辑位于 dataloader 目录。

    如果只是验证环境是否正确，建议先运行简化的 MNIST 示例。
    该示例会自动下载数据到 `tmp` 目录。
    """

    parser = argparse.ArgumentParser(
        description='运行 TEDS-Net 分割训练')

    parser.add_argument('--dataset',
                        help='选择要使用的数据集',
                        choices=['ACDC', 'mnist'],
                        default='mnist')

    args = parser.parse_args()

    # 执行训练流程
    Train_Runner(args)
