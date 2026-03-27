## TEDS-Net 实现说明 ##

本仓库实现了 MICCAI 2021 论文《TEDS-Net: Enforcing Diffeomorphisms in Spatial Transformers to Guarantee Topology Preservation in Segmentations》中描述的 TEDS-Net 架构。

当前分支保留了原始 `preprocess_acdc.py`、`trainACDC.py`、`evaluate_results.py` 与 `visualize.py` 这一条 ACDC 运行链路，并补齐了官方数据划分、批量训练、完整评估指标、报告导出与可视化导出能力。

--------------- 模拟示例 ---------------

如果你想快速验证 TEDS-Net 是否安装正确，推荐先运行数字 `0` 分割的模拟示例。由于输入图像尺寸较小，该示例使用的是更小版本的 TEDS-Net。

运行：

```bash
python scripts/train_runner.py --dataset mnist
```

训练结束后，终端会输出最终测试 Dice 结果。

![MNIST 示例结果](https://github.com/mwyburd/TEDS-Net/blob/main/MNIST_0_Example.png "MNIST 示例结果")

--------------- ACDC 运行 ---------------

1. 预处理官方 ACDC 数据集：

```bash
python preprocess_acdc.py ^
  --raw-data-path Resources ^
  --processed-data-path results/preprocessed/acdc_ring_144x208
```

2. 训练 ACDC 模型，例如 `batch=200`：

```bash
python trainACDC.py ^
  --processed-data-path results/preprocessed/acdc_ring_144x208 ^
  --run-name acdc_batch200 ^
  --epochs 200 ^
  --batch-size 200 ^
  --num-workers 0
```

3. 评估并导出报告：

```bash
python evaluate_results.py ^
  --processed-data-path results/preprocessed/acdc_ring_144x208 ^
  --run-name acdc_batch200
```

4. 如需单独重导可视化：

```bash
python visualize.py ^
  --processed-data-path results/preprocessed/acdc_ring_144x208 ^
  --run-name acdc_batch200
```

评估后会生成以下产物：

- 运行目录：`results/acdc/<run_name>/`
- 汇总指标：`results/acdc/<run_name>/metrics/summary_metrics.json`
- 逐切片指标：`results/acdc/<run_name>/metrics/per_slice_metrics.json`
- Markdown 报告：`docs/experiments/acdc_batch200_report.md`
- 精选可视化：`docs/experiments/assets/acdc_batch200/*.png`
