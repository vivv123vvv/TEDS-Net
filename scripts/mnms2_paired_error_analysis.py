import argparse
import csv
from collections import Counter
from pathlib import Path


METRICS = ["dice", "iou", "hd", "hd95", "assd", "precision", "recall", "correct_topology"]
TRACKED_CASES = ["062", "100", "102"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align M&Ms-2 per-sample CSVs and export paired HD/topology error analysis."
    )
    parser.add_argument(
        "--old-r2net-csv",
        default=str(
            Path("reports")
            / "benchmarks"
            / "mnms2"
            / "threshold_sweep"
            / "r2net_test_thr0p80_none_eval_per_sample.csv"
        ),
    )
    parser.add_argument(
        "--smooth-csv",
        default=str(
            Path("reports")
            / "benchmarks"
            / "mnms2"
            / "threshold_sweep_finetune_smooth_hdtopo_e5"
            / "r2net_test_thr0p76_none_eval_per_sample.csv"
        ),
    )
    parser.add_argument(
        "--hardmine-csv",
        default=str(
            Path("reports")
            / "benchmarks"
            / "mnms2"
            / "threshold_sweep_hardmine_hdtopo_e10"
            / "r2net_test_thr0p74_none_eval_per_sample.csv"
        ),
    )
    parser.add_argument(
        "--baseline-csv",
        default=str(
            Path("reports")
            / "benchmarks"
            / "mnms2"
            / "threshold_sweep_hardmine_hdtopo_e10"
            / "baseline_test_thr0p48_none_eval_per_sample.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("reports") / "benchmarks" / "mnms2" / "paired_error_analysis_hardmine_vs_smooth"),
    )
    parser.add_argument("--top-n", type=int, default=30)
    return parser.parse_args()


def read_rows(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["sample_id"]: row for row in rows}


def maybe_float(row, key):
    if row is None:
        return None
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def topology_ok(row):
    value = maybe_float(row, "correct_topology")
    return value is not None and value >= 1.0


def case_group(sample_id):
    return str(sample_id).split("_", 1)[0]


def row_case_id(*rows):
    for row in rows:
        if row and row.get("case_id"):
            return row["case_id"]
    return ""


def build_paired_rows(old_r2net, smooth, hardmine, baseline):
    sample_ids = sorted(set(old_r2net) & set(smooth) & set(hardmine) & set(baseline))
    paired_rows = []
    for sample_id in sample_ids:
        old_row = old_r2net[sample_id]
        smooth_row = smooth[sample_id]
        hardmine_row = hardmine[sample_id]
        baseline_row = baseline[sample_id]
        paired = {
            "sample_id": sample_id,
            "case_group": case_group(sample_id),
            "case_id": row_case_id(hardmine_row, smooth_row, old_row, baseline_row),
        }
        for label, source in [
            ("old_r2net", old_row),
            ("smooth_e5", smooth_row),
            ("hardmine", hardmine_row),
            ("baseline", baseline_row),
        ]:
            for metric in METRICS:
                value = maybe_float(source, metric)
                paired[f"{label}_{metric}"] = "" if value is None else value
            paired[f"{label}_pred_components"] = source.get("pred_components", "")
            paired[f"{label}_pred_holes"] = source.get("pred_holes", "")
            paired[f"{label}_target_components"] = source.get("target_components", "")
            paired[f"{label}_target_holes"] = source.get("target_holes", "")

        for metric in ["dice", "iou", "hd", "hd95", "assd", "precision", "recall", "correct_topology"]:
            smooth_value = maybe_float(smooth_row, metric)
            hardmine_value = maybe_float(hardmine_row, metric)
            if smooth_value is None or hardmine_value is None:
                paired[f"delta_hardmine_minus_smooth_{metric}"] = ""
            else:
                paired[f"delta_hardmine_minus_smooth_{metric}"] = hardmine_value - smooth_value

        paired["hardmine_fixed_smooth_topology_failure"] = int(
            (not topology_ok(smooth_row)) and topology_ok(hardmine_row)
        )
        paired["hardmine_still_topology_failure"] = int(not topology_ok(hardmine_row))
        paired["baseline_success_old_r2net_failure"] = int(
            topology_ok(baseline_row) and not topology_ok(old_row)
        )
        paired["baseline_success_smooth_failure"] = int(
            topology_ok(baseline_row) and not topology_ok(smooth_row)
        )
        paired["baseline_success_hardmine_failure"] = int(
            topology_ok(baseline_row) and not topology_ok(hardmine_row)
        )
        paired_rows.append(paired)
    return paired_rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return sum(values) / len(values) if values else None


def fmt(value, digits=4):
    if value is None or value == "":
        return "-"
    return f"{float(value):.{digits}f}"


def concentration(rows):
    counts = Counter(row["case_group"] for row in rows)
    total = len(rows)
    tracked = {case: counts.get(case, 0) for case in TRACKED_CASES}
    tracked_total = sum(tracked.values())
    top_cases = counts.most_common(10)
    return {
        "total": total,
        "tracked": tracked,
        "tracked_total": tracked_total,
        "tracked_rate": (tracked_total / total) if total else 0.0,
        "top_cases": top_cases,
    }


def markdown_case_concentration(name, rows):
    info = concentration(rows)
    top_text = ", ".join(f"{case}:{count}" for case, count in info["top_cases"]) or "-"
    tracked_text = ", ".join(f"{case}:{count}" for case, count in info["tracked"].items())
    return (
        f"| {name} | {info['total']} | {tracked_text} | "
        f"{info['tracked_rate']:.2%} | {top_text} |"
    )


def markdown_table(rows, columns, limit):
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows[:limit]:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = fmt(value, 4)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_markdown(path, args, paired_rows, hd_worse_rows, fixed_rows, still_failed_rows, baseline_failure_sets):
    metric_rows = []
    for label in ["old_r2net", "smooth_e5", "hardmine", "baseline"]:
        metric_rows.append(
            {
                "model": label,
                "Dice": mean(paired_rows, f"{label}_dice"),
                "IoU": mean(paired_rows, f"{label}_iou"),
                "HD": mean(paired_rows, f"{label}_hd"),
                "HD95": mean(paired_rows, f"{label}_hd95"),
                "ASSD": mean(paired_rows, f"{label}_assd"),
                "Precision": mean(paired_rows, f"{label}_precision"),
                "Recall": mean(paired_rows, f"{label}_recall"),
                "Topology failures": sum(1 for row in paired_rows if float(row[f"{label}_correct_topology"]) < 1.0),
            }
        )

    lines = [
        "# M&Ms-2 R2Net Paired Error Analysis",
        "",
        "## Inputs",
        "",
        f"- old_r2net: `{args.old_r2net_csv}`",
        f"- smooth_e5: `{args.smooth_csv}`",
        f"- hardmine: `{args.hardmine_csv}`",
        f"- baseline: `{args.baseline_csv}`",
        f"- aligned samples: {len(paired_rows)}",
        "",
        "## Metric Means",
        "",
        "| model | Dice | IoU | HD | HD95 | ASSD | Precision | Recall | Topology failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metric_rows:
        lines.append(
            "| {model} | {dice} | {iou} | {hd} | {hd95} | {assd} | {precision} | {recall} | {failures} |".format(
                model=row["model"],
                dice=fmt(row["Dice"]),
                iou=fmt(row["IoU"]),
                hd=fmt(row["HD"]),
                hd95=fmt(row["HD95"]),
                assd=fmt(row["ASSD"]),
                precision=fmt(row["Precision"]),
                recall=fmt(row["Recall"]),
                failures=row["Topology failures"],
            )
        )

    hd_delta = mean(paired_rows, "delta_hardmine_minus_smooth_hd")
    hd95_delta = mean(paired_rows, "delta_hardmine_minus_smooth_hd95")
    dice_delta = mean(paired_rows, "delta_hardmine_minus_smooth_dice")
    hd_worse_count = sum(1 for row in paired_rows if float(row["delta_hardmine_minus_smooth_hd"]) > 0)
    hd_better_count = sum(1 for row in paired_rows if float(row["delta_hardmine_minus_smooth_hd"]) < 0)
    lines.extend(
        [
            "",
            "## Hardmine vs Smooth",
            "",
            f"- Mean Dice delta: {fmt(dice_delta, 6)}",
            f"- Mean HD delta: {fmt(hd_delta, 6)}",
            f"- Mean HD95 delta: {fmt(hd95_delta, 6)}",
            f"- HD worsened on {hd_worse_count} samples and improved on {hd_better_count} samples.",
            f"- Hardmine fixed {len(fixed_rows)} smooth topology failures and still failed topology on {len(still_failed_rows)} samples.",
            "",
            "## Case Concentration",
            "",
            "| subset | samples | tracked 062/100/102 counts | tracked share | top case groups |",
            "| --- | ---: | --- | ---: | --- |",
            markdown_case_concentration("Top HD regressions", hd_worse_rows[: args.top_n]),
            markdown_case_concentration("Hardmine topology fixed", fixed_rows),
            markdown_case_concentration("Hardmine still failed", still_failed_rows),
        ]
    )
    for name, rows in baseline_failure_sets.items():
        lines.append(markdown_case_concentration(name, rows))

    lines.extend(
        [
            "",
            "## Top Hardmine HD Regressions",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            hd_worse_rows,
            [
                "sample_id",
                "case_id",
                "smooth_e5_hd",
                "hardmine_hd",
                "delta_hardmine_minus_smooth_hd",
                "smooth_e5_correct_topology",
                "hardmine_correct_topology",
            ],
            args.top_n,
        )
    )

    lines.extend(["", "## Hardmine Fixed Topology", ""])
    lines.extend(
        markdown_table(
            fixed_rows,
            [
                "sample_id",
                "case_id",
                "smooth_e5_hd",
                "hardmine_hd",
                "delta_hardmine_minus_smooth_hd",
                "smooth_e5_pred_holes",
                "hardmine_pred_holes",
            ],
            args.top_n,
        )
    )

    lines.extend(["", "## Hardmine Still Failed Topology", ""])
    lines.extend(
        markdown_table(
            still_failed_rows,
            [
                "sample_id",
                "case_id",
                "hardmine_hd",
                "hardmine_hd95",
                "hardmine_pred_components",
                "hardmine_pred_holes",
                "hardmine_target_components",
                "hardmine_target_holes",
            ],
            args.top_n,
        )
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    old_r2net = read_rows(args.old_r2net_csv)
    smooth = read_rows(args.smooth_csv)
    hardmine = read_rows(args.hardmine_csv)
    baseline = read_rows(args.baseline_csv)
    paired_rows = build_paired_rows(old_r2net, smooth, hardmine, baseline)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hd_worse_rows = sorted(
        [row for row in paired_rows if float(row["delta_hardmine_minus_smooth_hd"]) > 0],
        key=lambda row: float(row["delta_hardmine_minus_smooth_hd"]),
        reverse=True,
    )
    fixed_rows = [row for row in paired_rows if int(row["hardmine_fixed_smooth_topology_failure"]) == 1]
    still_failed_rows = [row for row in paired_rows if int(row["hardmine_still_topology_failure"]) == 1]
    baseline_failure_sets = {
        "Baseline success, old r2net failed": [
            row for row in paired_rows if int(row["baseline_success_old_r2net_failure"]) == 1
        ],
        "Baseline success, smooth failed": [
            row for row in paired_rows if int(row["baseline_success_smooth_failure"]) == 1
        ],
        "Baseline success, hardmine failed": [
            row for row in paired_rows if int(row["baseline_success_hardmine_failure"]) == 1
        ],
    }

    write_csv(output_dir / "paired_rows.csv", paired_rows)
    write_csv(output_dir / "hardmine_hd_worse_top.csv", hd_worse_rows[: args.top_n])
    write_csv(output_dir / "hardmine_topology_fixed.csv", fixed_rows)
    write_csv(output_dir / "hardmine_topology_still_failed.csv", still_failed_rows)
    for name, rows in baseline_failure_sets.items():
        file_name = name.lower().replace(", ", "_").replace(" ", "_") + ".csv"
        write_csv(output_dir / file_name, rows)

    write_markdown(
        output_dir / "paired_error_analysis.md",
        args,
        paired_rows,
        hd_worse_rows,
        fixed_rows,
        still_failed_rows,
        baseline_failure_sets,
    )

    print(f"Wrote paired analysis for {len(paired_rows)} aligned samples to {output_dir}")
    print(
        "Hardmine vs smooth: HD worsened on {worse}, fixed topology on {fixed}, still failed topology on {failed}".format(
            worse=len(hd_worse_rows),
            fixed=len(fixed_rows),
            failed=len(still_failed_rows),
        )
    )


if __name__ == "__main__":
    main()
