# M&Ms-2 错误样本可视化与训练结果汇报

生成日期：2026-06-16  
实验目录：`reports/experiment_reports/mnms2/mnms2_stratified_preprocess_v2_seed42_20260615`

## 1. 实验设置回顾

本轮实验在新的 patient-level stratified split 和 `preprocess-v2` 数据上完成，目标是缓解旧切分中 RV/TRI 只集中在 test 的问题，并避免预处理阶段静默丢弃 topology abnormal slice。新切分保持 patient-level 无泄漏，train/val/test 为 216/72/72 例，RV、TRI、ARR、CIA、FALL、HCM、LV、NOR 都覆盖到三个 split。

预处理侧改为先按 patient 切分再生成 slice，采用 robust intensity normalization，保留 patient_id、phase、slice_id、pathology、vendor、field_strength、spacing、split 等 metadata；无 MYO slice 和 topology abnormal slice 被记录到统计表，不再作为默认静默过滤对象。

本轮对比包含两个模型：

| 模型 | integrator | checkpoint | val threshold |
| --- | --- | --- | --- |
| baseline | original_teds | `checkpoints/mnms2/mnms2-original-teds-stratified-preprocess-v2-s42-20260615-e20-warm/best_teds_net.pth` | 0.58 |
| new model | r2net | `checkpoints/mnms2/mnms2-r2net-stratified-preprocess-v2-s42-20260615-e20-warm/best_teds_net.pth` | 0.78 |

后处理对 baseline 和 R2Net 完全一致：`closing_r1_lcc_fill_extra_holes_preserve_largest_hole`，即 closing radius=1、保留最大连通域、填补额外孔洞但保留最大内部孔洞。

## 2. 总体结果

| case | Dice | IoU | HD | HD95 | ASSD | Precision | Recall | Topology success | Topology failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline no post | 0.8547 | 0.7641 | 3.3330 | 2.2580 | 0.7768 | 0.8529 | 0.8670 | 0.8617 | 170 |
| baseline post | 0.8548 | 0.7643 | 3.3328 | 2.2605 | 0.7772 | 0.8522 | 0.8680 | 0.8617 | 170 |
| R2Net no post | 0.8572 | 0.7683 | 3.4055 | 2.3021 | 0.7638 | 0.8735 | 0.8515 | 0.8560 | 177 |
| R2Net post | 0.8576 | 0.7687 | 3.4027 | 2.3024 | 0.7630 | 0.8731 | 0.8525 | 0.8625 | 169 |

R2Net 在 Dice、IoU、ASSD 上优于 baseline，说明整体重叠质量和平均表面距离略有收益；但 HD 和 HD95 仍高于 baseline，说明长尾边界错误没有解决。后处理后 R2Net topology failure 从 177 降到 169，略好于 baseline post 的 170，但优势非常小。

因此本轮结论是：R2Net 在平均分割质量上超过 baseline，但在 HD/HD95 和 topology 鲁棒性上还不能称为全面超过 baseline。

## 3. 可视化错误样本总结

关键可视化文件：

- `figures/r2net_no_post_topology_failure_overview.png`
- `figures/r2net_post_remaining_failure_overview.png`
- `figures/baseline_vs_r2net_hd_outlier_comparison.png`

### 3.1 R2Net no-post topology failure overview

该图展示的是 R2Net 未后处理时的 topology failure 样本。整体观察不是“大面积完全分错”，而是集中在心肌环局部结构错误：

1. 额外孔洞是主要 topology failure 来源。R2Net no post 的 177 个 topology failure 中，176 个被统计为 extra holes；post 后仍有 168 个 extra holes。
2. basal/apical slice 占比较高。R2Net no post topology failure 中 basal/apical=1 的样本有 96/177，post 后仍有 95/169。这类 slice 心肌面积小、边界弱、形态不完整，少量轮廓断裂就会破坏拓扑。
3. 远端假阳性导致 HD 被显著放大。R2Net no post topology failure 中 remote false positive 为 10 个；HD top-30 中 remote false positive 达到 12 个。
4. 多连通域不是主要问题。本轮所有四组评估的 extra components count 都是 0，说明 keep largest connected component 对总体指标影响有限，当前问题主要是孔洞/环结构和远端假阳性。

### 3.2 R2Net post remaining failure overview

后处理修复了 8 个 topology failure，且没有引入新的 topology failure。被修复样本主要来自 CIA 和 ES phase：

| 类型 | 数量 |
| --- | ---: |
| 修复 topology failure | 8 |
| 新引入 topology failure | 0 |
| 修复样本 pathology | CIA 5, HCM 1, FALL 1, TRI 1 |
| 修复样本 phase | ES 7, ED 1 |

但 post 后剩余 failure 的视觉形态与 no-post 图高度相似，说明简单 closing + LCC + hole filling 只能处理局部小孔洞，无法修复以下问题：

- 预测整体位置偏移；
- basal/apical slice 上心肌环本身不完整；
- 阈值过高导致细薄区域断裂；
- 远端假阳性离 GT 太远，直接拉高 HD；
- GT 与预测的孔洞定义不一致。

### 3.3 Baseline vs R2Net HD outlier comparison

HD outlier 对比图里最突出的样本是 patient 295：

| sample | pathology | phase | R2Net HD | baseline HD | 主要错误 |
| --- | --- | --- | ---: | ---: | --- |
| 295_SA_ES_slice001 | FALL | ES | 80.53 | 14.56 | 大面积远端假阳性，post 基本无法修复 |
| 295_SA_ED_slice001 | FALL | ED | 62.94 | 8.06 | 大面积远端假阳性，R2Net 明显差于 baseline |
| 069_SA_ED_slice009 | LV | ED | 33.42 | 10.00 | basal/apical 小目标，远端误检和轮廓错位 |
| 086_SA_ES_slice001 | HCM | ES | 24.35 | 23.60 | 两者都困难，局部环结构和边界偏移 |
| 105_SA_ED_slice003 | ARR | ED | 23.77 | 11.05 | 3T 病例，边界偏移并伴远端误检 |
| 241_SA_ES_slice010 | LV | ES | 23.32 | 23.09 | 两者都差，心肌环偏移/变形 |
| 241_SA_ES_slice009 | LV | ES | 22.47 | 18.60 | LV 病例连续切片困难，边界偏移 |
| 241_SA_ED_slice011 | LV | ED | 21.95 | 18.03 | basal/apical slice，局部假阳性拉高 HD |
| 163_SA_ED_slice000 | NOR | ED | 21.59 | 21.59 | 两者相近，典型 apical/basal 难例 |
| 105_SA_ES_slice004 | ARR | ES | 21.02 | 11.18 | 3T 病例，R2Net 边界误差更大 |

这些 outlier 说明：R2Net 的总体 Dice 更高，但高 HD 样本并非随机噪声，而是集中出现在少量病人/连续 slice 上，尤其是 FALL/LV/HCM/ARR 的 basal-apical 或边界弱样本。patient 295 是当前 R2Net HD 被拉高的最明显病例，baseline 在该病例上明显更稳。

## 4. 分组结果解读

### 4.1 By pathology

R2Net post 相比 baseline post：

| pathology | Dice delta | HD delta | topology delta | 解读 |
| --- | ---: | ---: | ---: | --- |
| ARR | +0.0033 | +0.0124 | +0.0000 | Dice 小幅提升，HD 基本持平 |
| CIA | -0.0024 | +0.0492 | +0.0000 | 略差，但 topology 稳定 |
| FALL | -0.0007 | +1.1838 | +0.0000 | HD 明显变差，主要受 patient 295 远端假阳性影响 |
| HCM | +0.0098 | -0.0637 | +0.0000 | R2Net 在该组收益较清楚 |
| LV | +0.0046 | -0.1660 | +0.0000 | Dice 和 HD 都优于 baseline，但 topology failure 仍多 |
| NOR | +0.0040 | -0.1132 | +0.0000 | 整体略优 |
| RV | -0.0160 | +0.5549 | +0.0000 | R2Net 在 RV 上明显退化 |
| TRI | +0.0115 | -0.4313 | +0.0109 | R2Net 在 TRI 上收益最好之一 |

重点结论：HCM、LV、NOR、TRI 是 R2Net 的主要收益来源；FALL 和 RV 是主要风险来源。FALL 的 HD 退化尤其需要单独处理。

### 4.2 By vendor / field strength / phase

R2Net post 在 GE 上明显优于 baseline post：Dice +0.0172，HD -0.2311。Philips 和 SIEMENS 上 Dice 基本持平，但 HD 变差，说明 vendor domain 上仍有边界长尾问题。

field strength 方面，1.5T Dice 小幅提升但 HD 略差；3T Dice 和 HD 都略差。由于 3T test 样本只有 73 个，结论需要谨慎，但 ARR 3T outlier 已经出现在 top HD 中，后续应重点检查 3T 强度归一化和边界质量。

phase 方面，ES 的 Dice 提升较明显（+0.0064），ED 基本持平；但 ED/ES 的 HD 都略差。这说明 R2Net 对主体区域更自信，尤其 ES，但边界长尾并未被同步压低。

## 5. 为什么 Dice 提升但 topology/HD 不稳定

R2Net 的验证集最优阈值为 0.78，而 baseline 为 0.58。较高阈值会让预测更保守，通常能提升 precision 和局部重叠质量，但也更容易让细薄心肌区域断裂，或在弱边界处形成孔洞错误。本轮 R2Net 的 Precision 为 0.8731，高于 baseline post 的 0.8522；Recall 为 0.8525，低于 baseline post 的 0.8680，正好符合这一现象。

同时，HD 对远端假阳性极其敏感。少数如 patient 295 的远端误检会把 HD 从十几推到六十甚至八十以上，即使总体 Dice 在多数样本上更好，也会让平均 HD/HD95 落后。

## 6. 结论

本轮新 split 和 preprocess-v2 后的实验更公平，也更接近真实难度。R2Net 在总体 Dice、IoU、ASSD 上超过 original TEDS baseline，post 后 topology failure 也从 177 降至 169，略低于 baseline post 的 170。

但 R2Net 仍没有在 HD/HD95 上超过 baseline，主要瓶颈是远端假阳性、basal/apical slice 上的环结构破坏，以及 extra holes 主导的 topology failure。当前结果更适合表述为：R2Net 改善了平均重叠质量，但长尾拓扑和边界鲁棒性还没有真正解决。

## 7. 下一步建议

1. 对 patient 295、069、241、105 等 top HD 病例做 hard case replay，检查输入强度、GT、预测概率图和阈值敏感性。
2. 针对 remote false positive 增加训练或推理约束，例如 ROI 限制、距离惩罚、组件级 false positive penalty，或在后处理中加入与 LV/MYO 合理位置相关的约束。
3. 对 R2Net 单独做 threshold sensitivity 分析，不只按 mean HD 选阈值，也同时约束 topology success 和 recall。
4. 对 basal/apical slice 单独评估，可考虑 phase/slice-aware sampling 或对小目标 slice 加权。
5. 检查 topology target 的孔洞定义，确认 extra holes 是否与 MYO ring 的理论拓扑一致，必要时调整 topology metric 或修复策略。

## 8. 关键文件路径

- 总体对比：`reports/experiment_reports/mnms2/mnms2_stratified_preprocess_v2_seed42_20260615/comparison.md`
- 汇总 JSON：`reports/experiment_reports/mnms2/mnms2_stratified_preprocess_v2_seed42_20260615/summary.json`
- R2Net no-post topology failures：`reports/experiment_reports/mnms2/mnms2_stratified_preprocess_v2_seed42_20260615/failures/r2net_no_post_topology_failures.csv`
- R2Net post topology failures：`reports/experiment_reports/mnms2/mnms2_stratified_preprocess_v2_seed42_20260615/failures/r2net_post_topology_failures.csv`
- R2Net HD top-30：`reports/experiment_reports/mnms2/mnms2_stratified_preprocess_v2_seed42_20260615/failures/r2net_no_post_hd_top30.csv`
- HD outlier 对比图：`reports/experiment_reports/mnms2/mnms2_stratified_preprocess_v2_seed42_20260615/figures/baseline_vs_r2net_hd_outlier_comparison.png`
