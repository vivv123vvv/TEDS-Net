# ACDC 正式 Benchmark 报告

## 概览

本报告总结本 PR 中 ACDC 正式训练与评估结果。当前分支使用 R2Net / LC-ResNet 风格积分器替换单纯 TEDS-Net 中原有的 scaling-and-squaring 形变积分流程，并补齐可复现实验报告产物。

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

最佳 checkpoint 保存位置为：

```text
checkpoints/acdc-formal-20260417/best_teds_net.pth
```

## 指标汇总

### 论文表格风格指标

| Case | Dice ↑ | HD ↓ | Correct topology ↑ | Time per epoch [min] ↓ | # Parameters [×10^5] |
| --- | ---: | ---: | ---: | ---: | ---: |
| R2Net-integrated TEDS-Net | `0.8649` | `3.3776` | `94.41%` | `0.1246` | `7.0949` |

### 补充运行指标

| 指标 | 数值 |
| --- | ---: |
| 最佳验证 Dice | `0.8819` |
| Jacobian `< 0` 比例 | `0.0` |
| 平均前向时间 | `6.04 ms` |
| P50 前向时间 | `5.49 ms` |
| P95 前向时间 | `9.53 ms` |
| 训练峰值 GPU 显存 | `222.04 MB` |
| 评估峰值 GPU 显存 | `28.73 MB` |
| 测试样本数 | `179` |

## 结果解读

这次正式 run 为后续 ACDC 替换实验建立了可复现 baseline。固定测试集 Dice 为 `0.8649`，HD 为 `3.3776 px`，说明分割质量可以作为后续结构替换实验的基础参照。

形变与拓扑稳定性目前表现较稳：固定测试集上的 Jacobian `< 0` 比例为 `0.0`，没有检测到折叠；Correct topology 为 `94.41%`，说明大多数样本的前景连通分量数和孔洞数与 GT 一致。后续如果继续替换积分器、形变模块或网络结构，应同时观察 Dice、HD、拓扑正确率和 Jacobian 指标，而不能只看 Dice。

速度与资源开销也已经落盘为可复现实验产物：平均每个 epoch 为 `0.1246 min`，平均前向时间为 `6.04 ms`，训练峰值显存为 `222.04 MB`，评估峰值显存为 `28.73 MB`。参数量为 `7.09488 × 10^5`，与原 TEDS-Net 同量级，因此当前结果不是靠显著增加参数量换来的。

## 替换理由

相较于单纯 TEDS-Net，当前替换的核心理由是把最影响速度、显存和形变稳定性的积分环节单独拿出来做可控实验。原始 TEDS-Net 的 scaling-and-squaring 会反复进行形变场组合和空间采样；当前 R2Net / LC-ResNet 积分器在低分辨率 velocity field 上用谱归一化残差块进行积分，再上采样到 prior warp 所需尺寸，因此保留了 `raw_velocity -> flow_field -> flow_upsamp -> warp prior` 的外部接口，同时减少了显式迭代组合带来的额外开销。

这个替换不是为了改变整个网络定义，而是为了控制变量：编码器、解码器、prior warp 和训练/评估 split 保持一致，主要变化集中在形变积分器。这样后续可以更清楚地回答一个问题：学习式积分器能否在不明显增加参数量的情况下，维持 Dice 和拓扑稳定性，同时改善速度或显存。

从本次正式 run 看，替换版本达到了 `Dice=0.8649`、`HD=3.3776 px`、`Correct topology=94.41%`、`Jacobian < 0=0.0`，说明它已经具备作为后续对照实验 baseline 的条件。不过这里还不应直接宣称“优于单纯 TEDS-Net”：严格结论仍需要在同一 split、同一硬件、同一训练预算下重新跑原版 TEDS-Net，并用本 PR 新增的对比脚本生成同字段对照表。

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

本 PR 已具备作为 ACDC baseline benchmark 基础设施的条件。后续实验可以继续在本地生成 `reports/benchmarks/<run_name>/`，再通过 `scripts/compare_benchmarks.py` 与 `acdc-formal-20260417` 做统一对比。若要形成“替换优于单纯 TEDS-Net”的正式实验结论，下一步应在同环境下补跑原版 TEDS-Net，并对齐本报告中的全部指标。
