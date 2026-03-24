## TEDS-Net 实现说明 ##

本仓库实现了 MICCAI 2021 论文《TEDS-Net: Enforcing Diffeomorphisms in Spatial Transformers to Guarantee Topology Preservation in Segmentations》中描述的 TEDS-Net 架构。

2023 年 1 月更新后的代码额外提供了一个简化训练脚本，可基于模拟 MNIST 数据集完成数字 `0` 的分割训练。同时仓库中也包含了 ACDC 训练使用的参数文件，先验形状生成逻辑位于数据加载器中。如果你要运行 ACDC 示例，请先修改超参数文件和数据加载器中的数据路径。

--------------- 模拟示例 ---------------

如果你想快速验证 TEDS-Net 是否安装正确，推荐先运行数字 `0` 分割的模拟示例。由于输入图像尺寸较小，该示例使用的是更小版本的 TEDS-Net。

运行：

```bash
python scripts/train_runner.py
```

将训练 TEDS-Net 共 20 个 epoch，通常不会超过 1 分钟。

训练结束后，终端会输出类似下面的最终测试 Dice 结果：

```text
- - - - - - - - - - - - - - - - - - - ----------------------
Test Dice Loss: 0.9272134661674499 +/- 0.004152349107265217
- - - - - - - - - - - - - - - - - - - -----------------------
```

![MNIST 示例结果](https://github.com/mwyburd/TEDS-Net/blob/main/MNIST_0_Example.png "MNIST 示例结果")

上图展示了 MNIST 图像、先验形状 `P`，以及 TEDS-Net 在训练 20 个 epoch 后得到的分割结果。

--------------- 服务器运行 ---------------

如果只需要在服务器上验证“项目可以启动运行”，推荐直接使用仓库内的最小 smoke run 脚本：

```bash
bash server/bootstrap_and_run_mnist.sh
```

该脚本会执行以下操作：

- 先加载系统预装的 `anaconda/3.10`
- 仅在你自己的新 conda 环境中安装依赖，不改动已有环境
- 检查 `nvidia-smi` 和 `torch.cuda.is_available()`
- 以 `mnist` 数据集执行 1 个训练 batch、1 个验证 batch、1 个测试 batch 的最小运行

常用可覆盖变量：

```bash
ENV_NAME=tedsnet_py39 DATA_DIR=/path/to/tmp bash server/bootstrap_and_run_mnist.sh
```

如果你想直接运行训练入口，也可以手动执行：

```bash
python scripts/train_runner.py --dataset mnist --epochs 1 --batch-size 64 --num-workers 0 --max-train-batches 1 --max-validation-batches 1 --max-test-batches 1 --skip-plot
```
