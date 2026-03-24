import argparse

import torch

from trainer import Trainer


class Train_Runner:

    def __init__(self, args):

        # 1) 初始化参数
        self.setup_params(args)

        # 2) 选择设备
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"当前训练设备: {device}")

        # 3) 构建模型
        from network.TEDS_Net import TEDS_Net as net
        net = net(self.params)
        net.to(device)

        # 4) 训练并评估模型
        trainer = Trainer(self.params, device, net)
        trainer.dothetraining()
        trainer.do_evalutation()

    def setup_params(self, args):
        """根据命令行参数加载默认训练配置。"""

        if args.dataset == "mnist":
            from parameters.mnist_parameters import Parameters
        elif args.dataset == "ACDC":
            from parameters.acdc_parameters import Parameters
        else:
            raise ValueError(f"不支持的数据集: {args.dataset}")

        self.params = Parameters.from_dict({'data': args.dataset})

        if args.epochs is not None:
            self.params.epoch = args.epochs
        if args.batch_size is not None:
            self.params.batch = args.batch_size
        if args.num_workers is not None:
            self.params.num_workers = args.num_workers
        if args.data_path is not None:
            self.params.data_path = args.data_path
        if args.max_train_batches is not None:
            self.params.max_train_batches = args.max_train_batches
        if args.max_validation_batches is not None:
            self.params.max_validation_batches = args.max_validation_batches
        if args.max_test_batches is not None:
            self.params.max_test_batches = args.max_test_batches
        if args.skip_plot:
            self.params.plot_predictions = False


if __name__ == '__main__':
    """TEDS-Net 训练入口。"""

    parser = argparse.ArgumentParser(
        description='运行 TEDS-Net 分割训练')

    parser.add_argument('--dataset',
                        help='选择要使用的数据集',
                        choices=['ACDC', 'mnist'],
                        default='mnist')
    parser.add_argument('--epochs',
                        type=int,
                        help='覆盖默认训练轮数')
    parser.add_argument('--batch-size',
                        type=int,
                        dest='batch_size',
                        help='覆盖默认 batch size')
    parser.add_argument('--num-workers',
                        type=int,
                        help='覆盖 DataLoader worker 数')
    parser.add_argument('--data-path',
                        help='覆盖数据缓存或输出目录')
    parser.add_argument('--max-train-batches',
                        type=int,
                        help='每个 epoch 最多执行多少个训练 batch，0 表示不截断')
    parser.add_argument('--max-validation-batches',
                        type=int,
                        help='验证阶段最多执行多少个 batch，0 表示不截断')
    parser.add_argument('--max-test-batches',
                        type=int,
                        help='测试阶段最多执行多少个 batch，0 表示不截断')
    parser.add_argument('--skip-plot',
                        action='store_true',
                        help='跳过最终预测图保存，适合纯 smoke test')

    args = parser.parse_args()

    # 执行训练流程
    Train_Runner(args)
