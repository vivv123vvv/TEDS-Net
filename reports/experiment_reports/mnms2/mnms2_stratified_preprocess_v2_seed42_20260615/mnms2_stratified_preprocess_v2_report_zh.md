# M&Ms-2 stratified preprocess-v2 实验报告

## 实验目的

本实验修正原顺序切分导致的 pathology/vendor/field 分布偏移，使用 patient-level stratified split 和更透明的预处理记录，公平比较 original TEDS baseline 与 R2Net-TEDS。

## 新旧 split 差异

- 旧 split: readme 顺序切分 160/40/160，之前发现 train 中缺 RV/TRI，而 test 中集中出现 RV/TRI。
- 新 split: seed=42，train/val/test=216/72/72；按 pathology 硬配额，并平衡 vendor 与 field strength。
- split overview: `Resources\database\mnms2_stratified_seed42_20260615_preprocess_v2_2d\split_stats\split_overview.csv`
- split distributions: `Resources\database\mnms2_stratified_seed42_20260615_preprocess_v2_2d\split_stats\split_distributions.csv`
- 旧/新 split 对照汇总: `reports\experiment_reports\mnms2\mnms2_stratified_preprocess_v2_seed42_20260615\tables\old_vs_new_split_summary.csv`
- 旧/新 split 对照明细: `reports\experiment_reports\mnms2\mnms2_stratified_preprocess_v2_seed42_20260615\tables\old_vs_new_split_distributions.csv`
- 关键差异：旧 train 缺少 ARR/CIA/RV/TRI，旧 test 集中 ARR/CIA/RV/TRI；新 split 中 RV/TRI 均为 train=18、val=6、test=6，field=3T 为 train=10、val=3、test=4。

## 预处理修改

- patient-level split 后再生成 processed dataset。
- 任务仍为 SA ED/ES myocardium segmentation，MYO label=2。
- 图像使用 per-slice robust percentile normalization，spacing 不重采样但完整记录。
- 无 MYO slice 默认不进入训练样本，但写入 slice_records；topology abnormal slice 不再过滤，只记录。
- processed dataset: `Resources\database\mnms2_stratified_seed42_20260615_preprocess_v2_2d`
- split manifest: `parameters\mnms2_stratified_seed42_20260615_preprocess_v2_split.json`

## 训练配置

- R2Net: warm start 自旧最佳 smooth/hd-topology checkpoint，20 epoch，lr=2e-5，flow smooth=2500，boundary=0.2，clDice=0.1。
- baseline: original TEDS integrator，warm start 自旧 original TEDS checkpoint，20 epoch，lr=2e-5，原始 dice+grad loss。
- R2Net checkpoint: `checkpoints\mnms2\mnms2-r2net-stratified-preprocess-v2-s42-20260615-e20-warm\best_teds_net.pth`
- baseline checkpoint: `checkpoints\mnms2\mnms2-original-teds-stratified-preprocess-v2-s42-20260615-e20-warm\best_teds_net.pth`
- R2Net log: `logs/mnms2_r2net_stratified_preprocess_v2_seed42_20260615.out.log`
- baseline log: `logs/mnms2_baseline_stratified_preprocess_v2_seed42_20260615.out.log`

## 总体结果

| case | threshold | postprocess | dice | iou | hd | hd95 | assd | precision | recall | topology_success_rate | topology_failure_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_no_post | 0.5800 | none | 0.8547 | 0.7641 | 3.3330 | 2.2580 | 0.7768 | 0.8529 | 0.8670 | 0.8617 | 170 |
| baseline_post | 0.5800 | closing_r1_lcc_fill_extra_holes_preserve_largest_hole | 0.8548 | 0.7643 | 3.3328 | 2.2605 | 0.7772 | 0.8522 | 0.8680 | 0.8617 | 170 |
| r2net_no_post | 0.7800 | none | 0.8572 | 0.7683 | 3.4055 | 2.3021 | 0.7638 | 0.8735 | 0.8515 | 0.8560 | 177 |
| r2net_post | 0.7800 | closing_r1_lcc_fill_extra_holes_preserve_largest_hole | 0.8576 | 0.7687 | 3.4027 | 2.3024 | 0.7630 | 0.8731 | 0.8525 | 0.8625 | 169 |

## 后处理

- 策略: `closing_r1_lcc_fill_extra_holes_preserve_largest_hole`，closing radius=1，keep largest connected component，填充额外小孔但保留最大内腔。
- 四组结果均报告：baseline/new model 的无后处理与有后处理。

## 分组结果

- disease/pathology CSV: `reports\experiment_reports\mnms2\mnms2_stratified_preprocess_v2_seed42_20260615\tables\r2net_no_post_by_pathology.csv` 等同目录文件。
- vendor CSV: `reports\experiment_reports\mnms2\mnms2_stratified_preprocess_v2_seed42_20260615\tables\r2net_no_post_by_vendor.csv` 等同目录文件。
- field CSV: `reports\experiment_reports\mnms2\mnms2_stratified_preprocess_v2_seed42_20260615\tables\r2net_no_post_by_field_strength.csv` 等同目录文件。
- phase CSV: `reports\experiment_reports\mnms2\mnms2_stratified_preprocess_v2_seed42_20260615\tables\r2net_no_post_by_phase.csv` 等同目录文件。

## HD 与 topology failure 分析

- R2Net no-post HD top-30: `reports\experiment_reports\mnms2\mnms2_stratified_preprocess_v2_seed42_20260615\failures\r2net_no_post_hd_top30.csv`
- R2Net no-post topology failures: `reports\experiment_reports\mnms2\mnms2_stratified_preprocess_v2_seed42_20260615\failures\r2net_no_post_topology_failures.csv`
- R2Net post remaining failures: `reports\experiment_reports\mnms2\mnms2_stratified_preprocess_v2_seed42_20260615\failures\r2net_post_topology_failures.csv`
- 失败类型已拆分为 extra components、missing holes、extra holes、remote false positives 和 basal/apical 标记。

## 可视化

![new no post topology failures](reports\experiment_reports\mnms2\mnms2_stratified_preprocess_v2_seed42_20260615\figures\r2net_no_post_topology_failure_overview.png)

![new post topology failures](reports\experiment_reports\mnms2\mnms2_stratified_preprocess_v2_seed42_20260615\figures\r2net_post_remaining_failure_overview.png)

![baseline vs r2net HD outliers](reports\experiment_reports\mnms2\mnms2_stratified_preprocess_v2_seed42_20260615\figures\baseline_vs_r2net_hd_outlier_comparison.png)

## 是否超过 baseline

新模型在 Dice/IoU 上略高于 baseline，但 HD/HD95 和拓扑成功率未超过 baseline。

## 下一步建议

- 对 R2Net 在新 split 上从头训练 200 epoch，或至少延长 warm-start 至 50-100 epoch。
- 将 HD/topology loss 的训练目标与评估失败类型对齐，重点抑制远端假阳性和多连通域。
- 尝试 spacing-aware crop/resample 或记录物理单位 HD，减少 vendor/field spacing 差异的解释偏差。
- 对 basal/apical slice 单独建模或降低其对 topology ring 的硬约束。
