import json
import os
from datetime import datetime


def _format_metric(mean_value, std_value=None, unit=""):
    if std_value is None:
        return f"{mean_value:.4f}{unit}"
    return f"{mean_value:.4f}{unit} +/- {std_value:.4f}{unit}"


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
        f"- 训练命令：`{context['command']}`",
        f"- 设备：`{context['device']}`",
        "",
        "## 训练配置",
        "",
        f"- epoch：{context['epochs']}",
        f"- batch size：{context['batch_size']}",
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
        "## 测试集指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| Dice | {_format_metric(summary_metrics['dice_mean'], summary_metrics['dice_std'])} |",
        f"| HD95 | {_format_metric(summary_metrics['hd95_mean'], summary_metrics['hd95_std'])} |",
        f"| ASSD | {_format_metric(summary_metrics['assd_mean'], summary_metrics['assd_std'])} |",
        f"| 拓扑保持率 | {_format_metric(summary_metrics['topology_keep_rate'])} |",
        f"| Jacobian folding 比率 | {_format_metric(summary_metrics['folding_ratio_mean'], summary_metrics['folding_ratio_std'])} |",
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
            "- 本报告同时给出分割精度、距离误差、结果拓扑与形变 folding 两条稳定性证据。",
            "- 如果后续继续对积分器或先验形状做实验，应优先对比本报告中的 Dice/HD95/ASSD 与拓扑保持率是否同步变化。",
            "",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
