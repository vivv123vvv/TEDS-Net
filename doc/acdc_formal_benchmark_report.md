# ACDC 正式 Benchmark 报告

## 概览

本报告总结了本 PR 中新增的可复现 benchmark 流程对应的 ACDC 正式训练与评估结果。

- Run name：`acdc-formal-20260417`
- 运行日期：`2026-04-17`
- 计算设备：`NVIDIA GeForce RTX 5060`
- 数据划分文件：`parameters/acdc_split.json`
- 训练样本数：`1247`
- 验证样本数：`356`
- 测试样本数：`179`
- 训练 epoch：`200`
- Batch size：`5`
- Learning rate：`0.0001`

benchmark 产物已在本地生成，并且按约定不提交到 Git。本地输出目录为：

```text
reports/benchmarks/acdc-formal-20260417/
```

最佳 checkpoint 保存在：

```text
checkpoints/acdc-formal-20260417/best_teds_net.pth
```

## 指标汇总

| 指标 | 数值 |
| --- | ---: |
| 最佳验证 Dice | `0.8819` |
| 测试 Dice | `0.8649` |
| Jacobian `< 0` 比例 | `0.0` |
| 平均 epoch 时间 | `7.476 s` |
| 平均前向时间 | `5.16 ms` |
| P50 前向时间 | `4.68 ms` |
| P95 前向时间 | `9.02 ms` |
| 训练峰值 GPU 显存 | `222.04 MB` |
| 评估峰值 GPU 显存 | `48.72 MB` |

## 结果解读

这次正式 run 为后续 ACDC 替换实验建立了可复现 baseline。最佳验证 Dice 达到 `0.8819`，固定测试集 Dice 达到 `0.8649`，可作为后续比较分割质量的主要指标。

形变正则性结果较稳定：固定测试集上的 Jacobian `< 0` 比例为 `0.0`，说明当前评估没有检测到折叠。后续如果替换积分器或形变模块，应尽量保持该指标，或者明确说明 Dice、速度、显存与折叠比例之间的取舍。

速度与显存开销较轻：训练平均每个 epoch 为 `7.476 s`，训练峰值显存为 `222.04 MB`；评估平均前向时间为 `5.16 ms`，评估峰值显存为 `48.72 MB`。这些结果可以作为后续 integrator 或 architecture 对比实验的 baseline 行。

## 本地产物

正式 run 已生成以下本地文件：

- `reports/benchmarks/acdc-formal-20260417/train_epochs.csv`
- `reports/benchmarks/acdc-formal-20260417/train_summary.json`
- `reports/benchmarks/acdc-formal-20260417/eval_per_sample.csv`
- `reports/benchmarks/acdc-formal-20260417/eval_per_case.csv`
- `reports/benchmarks/acdc-formal-20260417/eval_summary.json`
- `reports/benchmarks/comparison.csv`
- `reports/benchmarks/comparison.md`

这些文件属于本地实验输出，不纳入提交。本 PR 只提交 benchmark pipeline、固定 split manifest 和结论文档。

## 复现命令

完整训练并自动评估：

```powershell
C:\ProgramData\Anaconda3\envs\TEDS-Net\python.exe trainACDC.py --run-name acdc-formal-20260417
```

使用最佳 checkpoint 单独评估：

```powershell
C:\ProgramData\Anaconda3\envs\TEDS-Net\python.exe evaluate_results.py ^
  --checkpoint checkpoints\acdc-formal-20260417\best_teds_net.pth ^
  --run-name acdc-formal-20260417
```

刷新对比表：

```powershell
C:\ProgramData\Anaconda3\envs\TEDS-Net\python.exe scripts\compare_benchmarks.py
```

## PR 结论

本 PR 已具备作为 baseline benchmark 基础设施的条件。后续实验可以继续在本地生成 `reports/benchmarks/<run_name>/`，再通过 `scripts/compare_benchmarks.py` 与 `acdc-formal-20260417` 进行统一对比。
