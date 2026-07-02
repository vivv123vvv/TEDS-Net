# 实验日志：mnms2-3d-baseline-original-teds-s42-e200-b1

- 状态：completed
- 开始时间：2026-06-30T16:56:26
- 结束时间：2026-06-30T19:26:22
- 数据集：M&Ms-2 3D (`mnms2_3d`)

## 研究目的

训练并评估 TEDS-Net 在 M&Ms-2 3D 数据集上的分割与拓扑保持表现。

## 核心产出

- run_dir: `reports\benchmarks\mnms2_3d\mnms2-3d-baseline-original-teds-s42-e200-b1`
- train_summary: `reports\benchmarks\mnms2_3d\mnms2-3d-baseline-original-teds-s42-e200-b1\train_summary.json`
- train_epochs: `reports\benchmarks\mnms2_3d\mnms2-3d-baseline-original-teds-s42-e200-b1\train_epochs.csv`
- eval_summary: `reports\benchmarks\mnms2_3d\mnms2-3d-baseline-original-teds-s42-e200-b1\eval_summary.json`
- best_checkpoint: `checkpoints\mnms2_3d\mnms2-3d-baseline-original-teds-s42-e200-b1\best_teds_net.pth`
- comparison_csv: `reports\benchmarks\mnms2_3d\comparison.csv`
- comparison_md: `reports\benchmarks\mnms2_3d\comparison.md`

## 实验结果

- Best Val Dice: 0.345941
- Mean Epoch Time: 0.740184 min
- Peak GPU Mem Train: 1297.688477 MB
- Parameter Count: 2027958
- Mean Dice: 0.345555
- Mean HD: 14.706829 pixel
- Correct Topology: 68.75%
- Jacobian < 0: 0.000000
- Mean Forward: 21.367734 ms
- Peak GPU Mem Eval: 824.084961 MB

## 数据集隔离

- data_dir: `Resources\database\mnms2_stratified_seed42_20260615_preprocess_v2_3d`
- split_manifest: `parameters\mnms2_stratified_seed42_20260615_preprocess_v2_3d_split.json`
- reports_dir: `reports\benchmarks\mnms2_3d`
- checkpoint_root: `checkpoints\mnms2_3d`
- experiment_log_dir: `logs\experiments\mnms2_3d`
- loader: `teds_npz`
- target_rule: `label_equals_value` / label_value: `2`

## 模型/代码隔离标注

- artifact_scope: reports, checkpoints, and experiment logs are separated under the mnms2_3d dataset_id
- code_scope: M&Ms-2 3D uses the shared NPZ loader with label_value=2 for MYO and D,H,W tensors
- model_change_scope: 3D behavior is selected through the dataset registry ndims/inshape fields.
- model_config: `{"dataset": {"ndims": 3, "inshape": [16, 144, 208], "ps_meas": [35, 7], "betti": [1, 1, 0, 0]}}`

## 复现命令

```powershell
trainACDC.py --dataset mnms2_3d --run-name mnms2-3d-baseline-original-teds-s42-e200-b1 --integrator original_teds --epochs 200 --batch-size 1 --device cuda
```

## 备注

无
