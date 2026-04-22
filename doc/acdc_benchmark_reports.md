# ACDC Benchmark 报告使用说明

## 默认位置

- 固定数据划分：`parameters/acdc_split.json`
- 训练与评估报告：`reports/benchmarks/<run_name>/`
- 每个 run 的最佳 checkpoint：`checkpoints/<run_name>/best_teds_net.pth`

## 指标范围

当前 benchmark 会自动记录以下指标：

- 训练阶段：每个 epoch 的 `epoch_sec`、`epoch_min`、平均 batch 时间、峰值 GPU 显存、验证 Dice。
- 评估阶段：每个样本和每个病例的前向时间、Dice、HD、Correct topology、Jacobian `< 0` 比例。
- 汇总阶段：`comparison.csv` 和 `comparison.md` 会聚合 `Dice`、`HD`、`Correct topology`、`Time per epoch [min]`、`# Parameters [×10^5]`、显存和 Jacobian 指标。

其中 `HD` 的单位为 pixel；`Correct topology` 通过预测 mask 与 GT mask 的前景连通分量数、孔洞数是否一致来统计。

## 训练并自动评估

在仓库根目录运行：

```powershell
C:\ProgramData\Anaconda3\envs\TEDS-Net\python.exe trainACDC.py --run-name acdc-baseline
```

训练阶段会写出：

- `train_epochs.csv`
- `train_summary.json`

最佳 checkpoint 保存后，训练脚本会自动在配置的 split 上运行评估，并写出：

- `eval_per_sample.csv`
- `eval_per_case.csv`
- `eval_summary.json`

同时会刷新：

- `reports/benchmarks/comparison.csv`
- `reports/benchmarks/comparison.md`

## Smoke 运行

如果只想快速检查链路是否可跑，可以运行：

```powershell
C:\ProgramData\Anaconda3\envs\TEDS-Net\python.exe trainACDC.py ^
  --run-name smoke-acdc ^
  --epochs 2 ^
  --max-train-batches 2 ^
  --max-val-batches 2 ^
  --eval-max-samples 8
```

## 单独评估

```powershell
C:\ProgramData\Anaconda3\envs\TEDS-Net\python.exe evaluate_results.py ^
  --checkpoint checkpoints\acdc-baseline\best_teds_net.pth ^
  --run-name acdc-baseline
```

如果省略 `--checkpoint`，且 `checkpoints\best_teds_net.pth` 不存在，评估脚本会回退到最近更新的 `checkpoints\<run_name>\best_teds_net.pth`。

## 手动刷新对比表

```powershell
C:\ProgramData\Anaconda3\envs\TEDS-Net\python.exe scripts\compare_benchmarks.py
```

## 正式 baseline

正式 run name：`acdc-formal-20260417`

| 指标 | 数值 |
| --- | ---: |
| 最佳验证 Dice | `0.8819` |
| 测试 Dice | `0.8649` |
| HD | `3.3776 px` |
| Correct topology | `94.41%` |
| Jacobian `< 0` 比例 | `0.0` |
| 平均 epoch 时间 | `0.1246 min` / `7.476 s` |
| 平均前向时间 | `6.04 ms` |
| P50 / P95 前向时间 | `5.49 ms` / `9.53 ms` |
| 参数量 | `709,488` / `7.09488 × 10^5` |
| 训练峰值 GPU 显存 | `222.04 MB` |
| 评估峰值 GPU 显存 | `28.73 MB` |

该 run 的本地产物位于：

- `reports/benchmarks/acdc-formal-20260417/`
- `checkpoints/acdc-formal-20260417/best_teds_net.pth`
