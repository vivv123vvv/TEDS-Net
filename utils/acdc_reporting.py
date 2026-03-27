import json
import os
from datetime import datetime


def _format_mean_std(mean_value, std_value=None, unit=""):
    if std_value is None:
        return f"{mean_value:.4f}{unit}"
    return f"{mean_value:.4f}{unit} +/- {std_value:.4f}{unit}"


def _format_percent(value):
    return f"{value * 100.0:.2f}%"


def _format_parameter_count(parameter_count):
    return f"{parameter_count / 1e5:.2f}"


def write_acdc_report(report_path, context, summary_metrics, visualization_entries):
    """生成中文 Markdown 实验报告。"""

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    split_counts = context.get("split_counts", {})

    lines = [
        "# ACDC Batch=200 实验报告",
        "",
        "> 本文件由评估流程自动生成；若重复运行，会被新的实验结果覆盖。",
        "",
        "## 实验概览",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 分支：`{context['git_branch']}`",
        f"- 提交：`{context['git_commit']}`",
        f"- 运行名：`{context['run_name']}`",
        f"- 原始数据目录：`{context['raw_data_path']}`",
        f"- 预处理缓存目录：`{context['processed_data_path']}`",
        f"- 结果目录：`{context['run_output_dir']}`",
        f"- 最佳模型：`{context['best_checkpoint_path']}`",
        f"- 训练命令：`{context['train_command']}`",
        f"- 评估命令：`{context['eval_command']}`",
        f"- 设备：`{context['device']}`",
        "",
        "## 数据划分与训练配置",
        "",
        "- 数据划分策略：官方 `training/testing` 划分；仅在 `training` 内按病人切分训练集与验证集。",
        "- 数据泄漏风险说明：训练、验证、测试之间不共享病人；不再使用随机切片混拆。",
        f"- epoch：{context['epochs']}",
        f"- batch size：{context['batch_size']}",
        f"- 最佳 epoch：{context['best_epoch']}",
        f"- 训练病人数：{split_counts.get('train_patients', 'unknown')}",
        f"- 验证病人数：{split_counts.get('validation_patients', 'unknown')}",
        f"- 测试病人数：{split_counts.get('test_patients', 'unknown')}",
        f"- 训练切片数：{split_counts.get('train_slices', 'unknown')}",
        f"- 验证切片数：{split_counts.get('validation_slices', 'unknown')}",
        f"- 测试切片数：{split_counts.get('test_slices', 'unknown')}",
        "",
        "## 预处理摘要",
        "",
        "```json",
        json.dumps(context["preprocess_summary"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 主结果表",
        "",
        "| Dice ↑ | HD ↓ | Correct topology ↑ | Time per epoch [min] ↓ | # Parameters [x10^5] |",
        "| --- | --- | --- | --- | --- |",
        (
            f"| {_format_mean_std(summary_metrics['dice_mean'], summary_metrics['dice_std'])} "
            f"| {_format_mean_std(summary_metrics['hd_mean'], summary_metrics['hd_std'])} "
            f"| {_format_percent(summary_metrics['topology_keep_rate'])} "
            f"| {context['epoch_time_minutes_mean']:.2f} "
            f"| {_format_parameter_count(context['parameter_count'])} |"
        ),
        "",
        "## 补充指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| HD95 | {_format_mean_std(summary_metrics['hd95_mean'], summary_metrics['hd95_std'])} |",
        f"| ASSD | {_format_mean_std(summary_metrics['assd_mean'], summary_metrics['assd_std'])} |",
        f"| Jacobian folding 比率 | {_format_mean_std(summary_metrics['folding_ratio_mean'], summary_metrics['folding_ratio_std'])} |",
        "",
        "## 可视化样例",
        "",
        "下列四组样例分别对应 Dice 最好、中位、最差，以及固定随机种子的随机样本。",
        "",
    ]

    for entry in visualization_entries:
        lines.extend(
            [
                f"### {entry['label']}",
                "",
                f"- 样本：`{entry['sample_id']}`",
                f"- Dice：{entry['dice']:.4f}",
                f"- HD：{entry['hd']:.4f}",
                f"- HD95：{entry['hd95']:.4f}",
                f"- ASSD：{entry['assd']:.4f}",
                f"- GT Betti：`{tuple(entry['gt_betti'])}`",
                f"- Pred Betti：`{tuple(entry['pred_betti'])}`",
                f"- Folding Ratio：{entry['folding_ratio']:.6f}",
                "",
                f"![{entry['label']}]({entry['report_image_path']})",
                "",
            ]
        )

    lines.extend(
        [
            "## 结论",
            "",
            "- 主表按照图表口径给出 Dice、HD、拓扑保持率、单 epoch 耗时和参数量，便于与基线直接横向对比。",
            "- 报告同时补充 HD95、ASSD 与 Jacobian folding 比率，帮助判断精度、边界误差与形变稳定性是否同步变化。",
            "",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
