## TEDS-Net 实现说明 ##

本仓库实现了 MICCAI 2021 论文《TEDS-Net: Enforcing Diffeomorphisms in Spatial Transformers to Guarantee Topology Preservation in Segmentations》中描述的 TEDS-Net 架构。

2023 年 1 月更新后的代码额外提供了一个简化训练脚本，可基于模拟 MNIST 数据集完成数字 `0` 的分割训练。同时仓库中也包含了 ACDC 训练使用的参数文件，先验形状生成逻辑位于数据加载器中。如果你要运行 ACDC 示例，请先修改超参数文件和数据加载器中的数据路径。

--------------- 模拟示例 ---------------

如果你想快速验证 TEDS-Net 是否安装正确，推荐先运行数字 `0` 分割的模拟示例。由于输入图像尺寸较小，该示例使用的是更小版本的 TEDS-Net。

运行：

>> train_runner.py

将训练 TEDS-Net 共 20 个 epoch，通常不会超过 1 分钟。

训练结束后，终端会输出类似下面的最终测试 Dice 结果：

 >> - - - - - - - - - - - - - - - - - - - ----------------------
 >> Test Dice Loss: 0.9272134661674499 +/- 0.004152349107265217
 >> - - - - - - - - - - - - - - - - - - - -----------------------

![MNIST 示例结果](https://github.com/mwyburd/TEDS-Net/blob/main/MNIST_0_Example.png "MNIST 示例结果")

上图展示了 MNIST 图像、先验形状 `P`，以及 TEDS-Net 在训练 20 个 epoch 后得到的分割结果。
