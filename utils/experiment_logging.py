import json
import shlex
import sys
from datetime import datetime
from pathlib import Path

from utils.acdc_benchmark import ensure_dir


def format_command(argv=None):
    argv = list(sys.argv if argv is None else argv)
    return " ".join(shlex.quote(str(part)) for part in argv)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _format_value(value, unit=None, percent=False):
    if value is None:
        return "未产生"
    if percent:
        return f"{float(value) * 100.0:.2f}%"
    if isinstance(value, float):
        text = f"{value:.6f}"
    else:
        text = str(value)
    return f"{text} {unit}" if unit else text


def _summary_metric_lines(train_summary, eval_summary):
    lines = []
    if train_summary:
        lines.extend(
            [
                f"- Best Val Dice: {_format_value(train_summary.get('best_val_dice'))}",
                f"- Mean Epoch Time: {_format_value(train_summary.get('mean_epoch_min'), 'min')}",
                f"- Peak GPU Mem Train: {_format_value(train_summary.get('max_peak_gpu_mem_mb'), 'MB')}",
                f"- Parameter Count: {_format_value(train_summary.get('parameter_count'))}",
            ]
        )
    if eval_summary:
        lines.extend(
            [
                f"- Mean Dice: {_format_value(eval_summary.get('mean_dice'))}",
                f"- Mean HD: {_format_value(eval_summary.get('mean_hd'), eval_summary.get('hd_unit', 'pixel'))}",
                f"- Correct Topology: {_format_value(eval_summary.get('correct_topology_rate'), percent=True)}",
                f"- Jacobian < 0: {_format_value(eval_summary.get('mean_jacobian_neg_ratio'))}",
                f"- Mean Forward: {_format_value(eval_summary.get('mean_forward_ms'), 'ms')}",
                f"- Peak GPU Mem Eval: {_format_value(eval_summary.get('peak_gpu_mem_mb'), 'MB')}",
            ]
        )
    return lines or ["- 本轮未产生可汇总指标。"]


def _artifact_lines(artifact_paths, train_summary, eval_summary, comparison):
    rows = []
    artifact_paths = artifact_paths or {}
    for label, path in artifact_paths.items():
        if path:
            rows.append(f"- {label}: `{path}`")
    if train_summary:
        checkpoint_path = train_summary.get("checkpoint_path")
        if checkpoint_path:
            rows.append(f"- best_checkpoint: `{checkpoint_path}`")
    if eval_summary:
        eval_checkpoint = eval_summary.get("checkpoint_path")
        if eval_checkpoint and not any(eval_checkpoint in row for row in rows):
            rows.append(f"- eval_checkpoint: `{eval_checkpoint}`")
    if comparison:
        if comparison.get("csv_path"):
            rows.append(f"- comparison_csv: `{comparison['csv_path']}`")
        if comparison.get("md_path"):
            rows.append(f"- comparison_md: `{comparison['md_path']}`")
    return rows or ["- 本轮没有写出本地产物。"]


def experiment_log_paths(log_dir, run_name):
    log_dir = ensure_dir(log_dir)
    safe_run_name = str(run_name).replace("/", "_").replace("\\", "_")
    return {
        "md_path": log_dir / f"{safe_run_name}.md",
        "json_path": log_dir / f"{safe_run_name}.json",
    }


def write_experiment_log(
    log_dir,
    run_name,
    dataset_spec,
    research_purpose,
    status,
    train_summary=None,
    eval_summary=None,
    comparison=None,
    artifact_paths=None,
    command=None,
    notes=None,
    error=None,
    started_at=None,
    finished_at=None,
):
    finished_at = finished_at or datetime.now().isoformat(timespec="seconds")
    paths = experiment_log_paths(log_dir, run_name)
    dataset_payload = dataset_spec.to_dict() if hasattr(dataset_spec, "to_dict") else dataset_spec
    payload = {
        "run_name": run_name,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "dataset": dataset_payload,
        "research_purpose": research_purpose,
        "train_summary": train_summary,
        "eval_summary": eval_summary,
        "comparison": comparison,
        "artifact_paths": artifact_paths or {},
        "command": command,
        "notes": notes,
        "error": error,
    }

    with paths["json_path"].open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False)

    isolation = dataset_payload.get("isolation", {}) if isinstance(dataset_payload, dict) else {}
    model_payload = dataset_payload.get("model", {}) if isinstance(dataset_payload, dict) else {}
    lines = [
        f"# 实验日志：{run_name}",
        "",
        f"- 状态：{status}",
        f"- 开始时间：{started_at or '未记录'}",
        f"- 结束时间：{finished_at}",
        f"- 数据集：{dataset_payload.get('display_name', 'unknown')} (`{dataset_payload.get('dataset_id', 'unknown')}`)",
        "",
        "## 研究目的",
        "",
        research_purpose or "未填写，本轮仅按默认训练/评估流程执行。",
        "",
        "## 核心产出",
        "",
        *_artifact_lines(artifact_paths, train_summary, eval_summary, comparison),
        "",
        "## 实验结果",
        "",
        *_summary_metric_lines(train_summary, eval_summary),
        "",
        "## 数据集隔离",
        "",
        f"- data_dir: `{dataset_payload.get('data_dir')}`",
        f"- split_manifest: `{dataset_payload.get('split_manifest')}`",
        f"- reports_dir: `{dataset_payload.get('reports_dir')}`",
        f"- checkpoint_root: `{dataset_payload.get('checkpoint_root')}`",
        f"- experiment_log_dir: `{dataset_payload.get('experiment_log_dir')}`",
        f"- loader: `{dataset_payload.get('loader')}`",
        f"- target_rule: `{dataset_payload.get('target_rule')}` / label_value: `{dataset_payload.get('label_value')}`",
        "",
        "## 模型/代码隔离标注",
        "",
        f"- artifact_scope: {isolation.get('artifact_scope', '未标注')}",
        f"- code_scope: {isolation.get('code_scope', '未标注')}",
        f"- model_change_scope: {isolation.get('model_change_scope', '未标注')}",
        f"- model_config: `{json.dumps(_json_safe(model_payload), ensure_ascii=False)}`",
        "",
        "## 复现命令",
        "",
        "```powershell",
        command or "未记录",
        "```",
        "",
        "## 备注",
        "",
        notes or "无",
    ]
    if error:
        lines.extend(["", "## 失败信息", "", f"```text\n{error}\n```"])

    paths["md_path"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths
