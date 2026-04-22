# AGENT.md

## 适用范围

本文件适用于仓库根目录及所有子目录。后续自动化代理、协作者或脚本在本仓库内修改代码、整理实验结论、撰写 PR 或生成 Markdown 报告时，应优先遵守这里的约定。

## 语言规范

- PR 标题、PR 正文、PR 评论和变更说明默认使用中文书写。
- Markdown 报告、实验结论、benchmark 总结和复现实验说明默认使用中文书写。
- 代码标识、文件路径、命令、参数名、指标字段名和日志字段可以保留英文，例如 `Dice`、`Jacobian < 0`、`train_summary.json`、`--run-name`。
- 如需引用论文名、模型名、函数名或第三方工具名，应保留原文，并在必要时补充中文解释。

## PR 规范

- PR 标题应简洁说明用户可感知的改动，例如“新增 ACDC 可复现实验报告与自动对比脚本”。
- PR 正文建议包含 `摘要`、`主要改动`、`验证结果`、`实验结论` 和 `注意事项`。
- 如果 PR 包含 benchmark，应写清 run name、日期、数据 split、checkpoint 路径和本地报告目录。
- benchmark 结论至少应覆盖 `Dice`、`HD`、`Correct topology`、`Jacobian < 0`、前向时间、每 epoch 时间、峰值 GPU 显存和参数量。
- 不要把大型本地产物、训练 checkpoint、原始医学影像数据或本地 IDE 配置提交进 PR。

## Markdown 报告规范

- 报告正文使用中文，结构优先采用 `概览`、`指标汇总`、`结果解读`、`替换理由`、`本地产物`、`复现命令`、`PR 结论`。
- 指标表格应包含数值和单位，例如 `0.125 min`、`6.04 ms`、`222.04 MB`、`7.09488 × 10^5`。
- 如报告讨论 R2Net / LC-ResNet 积分器替换，应说明它相较于单纯 TEDS-Net 的动机、收益边界和仍需同环境对照验证的事项。
- 正式实验报告应说明本地产物没有提交到 Git，只提交可复现实验脚本、固定 split 和结论文档。
- 复现命令应使用 fenced code block，并保留原始命令格式。

## 实验与产物约定

- ACDC benchmark 默认使用 `parameters/acdc_split.json` 作为固定划分。
- 训练和评估报告默认输出到 `reports/benchmarks/<run_name>/`。
- 训练 checkpoint 默认输出到 `checkpoints/<run_name>/`。
- `reports/`、`checkpoints/`、原始数据目录和本地缓存应保持 Git ignore。
- 代码提交前应至少确认相关 Python 文件可以通过语法检查；如运行了 smoke 或正式训练，应在 PR 文档中记录结果。
