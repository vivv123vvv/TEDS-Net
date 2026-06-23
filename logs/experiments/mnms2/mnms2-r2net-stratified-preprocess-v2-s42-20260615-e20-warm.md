# 实验日志：mnms2-r2net-stratified-preprocess-v2-s42-20260615-e20-warm

- 状态：completed
- 开始时间：2026-06-15T11:14:42
- 结束时间：2026-06-15T11:42:23
- 数据集：M&Ms-2 (`mnms2`)

## 研究目的

训练并评估 TEDS-Net 在 M&Ms-2 数据集上的分割与拓扑保持表现。

## 核心产出

- run_dir: `reports\benchmarks\mnms2\mnms2-r2net-stratified-preprocess-v2-s42-20260615-e20-warm`
- train_summary: `reports\benchmarks\mnms2\mnms2-r2net-stratified-preprocess-v2-s42-20260615-e20-warm\train_summary.json`
- train_epochs: `reports\benchmarks\mnms2\mnms2-r2net-stratified-preprocess-v2-s42-20260615-e20-warm\train_epochs.csv`
- best_checkpoint: `checkpoints\mnms2\mnms2-r2net-stratified-preprocess-v2-s42-20260615-e20-warm\best_teds_net.pth`

## 实验结果

- Best Val Dice: 0.845256
- Mean Epoch Time: 1.382971 min
- Peak GPU Mem Train: 388.945312 MB
- Parameter Count: 709488

## 数据集隔离

- data_dir: `Resources\database\mnms2_processed_2d`
- split_manifest: `parameters\mnms2_split.json`
- reports_dir: `reports\benchmarks\mnms2`
- checkpoint_root: `checkpoints\mnms2`
- experiment_log_dir: `logs\experiments\mnms2`
- loader: `teds_npz`
- target_rule: `label_equals_value` / label_value: `2`

## 模型/代码隔离标注

- artifact_scope: reports, checkpoints, and experiment logs are separated under the mnms2 dataset_id
- code_scope: M&Ms-2 uses the shared NPZ loader with label_value=2 for MYO
- model_change_scope: No M&Ms-2-specific model file change is required; model differences are selected by --integrator.
- model_config: `{"dataset": {"ndims": 2, "inshape": [144, 208], "ps_meas": [35, 7], "betti": [1, 1, 0, 0]}}`

## 复现命令

```powershell
trainACDC.py --dataset mnms2 --data-dir 'Resources\database\mnms2_stratified_seed42_20260615_preprocess_v2_2d' --split-manifest 'parameters\mnms2_stratified_seed42_20260615_preprocess_v2_split.json' --run-name mnms2-r2net-stratified-preprocess-v2-s42-20260615-e20-warm --epochs 20 --lr 0.00002 --integrator r2net --init-checkpoint 'checkpoints\mnms2\mnms2-r2net-s42-e200-finetune-smooth-hdtopo-e5\best_teds_net.pth' --flow-smooth-weight 2500 --flow-smooth-penalty l2 --boundary-distance-weight 0.2 --boundary-distance-max 20 --boundary-distance-min-weight 1 --cldice-weight 0.1 --cldice-iterations 10 --best-checkpoint-metric total --device cuda --skip-final-eval --experiment-notes 'Warm-start R2Net smooth/hd-topology on stratified seed42 preprocess-v2 split.'
```

## 备注

Warm-start R2Net smooth/hd-topology on stratified seed42 preprocess-v2 split.
