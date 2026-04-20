# PR 说明：ACDC benchmark 与本地报告自动化

## 建议标题

新增 ACDC 可复现实验报告与自动对比流程

## 摘要

本 PR 将 ACDC 训练和评估流程整理为一套可复现的 benchmark pipeline。它固定数据划分，记录每个 epoch 的训练速度和峰值显存，记录每个样本和每个病例的评估指标，并将本地 benchmark 报告写入 `reports/benchmarks/<run_name>/`。

同时，PR 新增了自动对比脚本，后续替换积分器、网络模块或训练配置时，可以直接与同一套 baseline 字段进行对比。

## 主要改动

- 新增固定 split manifest：`parameters/acdc_split.json`
- 扩展 `ACDCNpzDataset`，支持固定文件列表和元信息输出：`sample_id`、`case_id`
- 新增共享 benchmark 工具：`utils/acdc_benchmark.py`
- 更新 `trainACDC.py`：
  - 支持 benchmark 相关 CLI 参数
  - 写出 `train_epochs.csv` 和 `train_summary.json`
  - 保存 run 专属 checkpoint
  - 训练结束后自动执行评估
  - 训练结束后自动刷新对比输出
- 更新 `evaluate_results.py`：
  - 支持 benchmark 相关 CLI 参数
  - 写出 `eval_per_sample.csv`、`eval_per_case.csv` 和 `eval_summary.json`
  - 支持 warmup batch 和受限样本评估
- 新增 `scripts/compare_benchmarks.py`，用于聚合多个 run 的 benchmark 结果
- 新增 benchmark 流程文档和正式实验报告
- 更新 `.gitignore`，排除本地报告、checkpoint、数据缓存和原始数据目录

## 验证结果

- smoke 训练和训练后自动评估已完成
- 独立评估脚本已完成验证
- 多 run 对比聚合已完成验证
- ACDC 正式 baseline 训练已在 `2026-04-17` 完成

## 正式 baseline 结论

Run name：`acdc-formal-20260417`

- 最佳验证 Dice：`0.8819`
- 测试 Dice：`0.8649`
- 平均 epoch 时间：`7.476 s`
- 平均前向时间：`5.16 ms`
- P50 / P95 前向时间：`4.68 ms / 9.02 ms`
- 训练 / 评估峰值 GPU 显存：`222.04 MB / 48.72 MB`
- Jacobian `< 0` 比例：`0.0`

## 注意事项

- 完整 benchmark 产物已在本地生成，但不会提交到 Git。
- 正式 run 的本地产物位于 `reports/benchmarks/acdc-formal-20260417/`。
- 最佳 checkpoint 位于 `checkpoints/acdc-formal-20260417/best_teds_net.pth`。
