# ACDC Batch=200 实验报告

当前文件为自动生成报告的占位路径。

正式运行 `scripts/preprocess_acdc.py` 与 `scripts/train_runner.py --dataset ACDC ...` 后，评估流程会自动覆盖本文件，并写入：

- Dice
- HD95
- ASSD
- 拓扑保持率
- Jacobian folding 比率
- 代表性可视化

如果此文件仍是占位内容，表示当前分支尚未在目标服务器上完成正式 ACDC 训练与评估。
