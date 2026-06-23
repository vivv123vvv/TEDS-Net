import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a hard-sample manifest from per-sample evaluation CSV files."
    )
    parser.add_argument("--input-csv", nargs="+", required=True)
    parser.add_argument(
        "--cutoff-reference-csv",
        nargs="*",
        default=None,
        help="Optional CSV(s) used only to compute top-percent HD/HD95 cutoffs.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="mnms2")
    parser.add_argument("--split-manifest", default=str(Path("parameters") / "mnms2_split.json"))
    parser.add_argument("--top-percent", type=float, default=10.0)
    parser.add_argument("--hd-threshold", type=float, default=None)
    parser.add_argument("--hd95-threshold", type=float, default=None)
    parser.add_argument(
        "--filter-to-split",
        default=None,
        help="Optional split name to keep in the emitted hard sample ids, e.g. train.",
    )
    parser.add_argument(
        "--reference-split",
        default=None,
        help="Optional split name to keep when computing top-percent cutoffs, e.g. val.",
    )
    return parser.parse_args()


def read_csv_rows(paths):
    rows = []
    for csv_path in paths:
        csv_path = Path(csv_path)
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["_source_csv"] = str(csv_path)
                rows.append(row)
    return rows


def load_sample_split_map(split_manifest_path):
    split_manifest_path = Path(split_manifest_path)
    if not split_manifest_path.exists():
        return {}
    with split_manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    splits = payload.get("splits", payload)
    sample_to_split = {}
    for split_name, file_names in splits.items():
        for file_name in file_names:
            sample_to_split[Path(file_name).stem] = split_name
    return sample_to_split


def as_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value):
    number = as_float(value)
    if number is None:
        return None
    return int(number)


def percentile(values, q):
    clean_values = sorted(float(value) for value in values if value is not None)
    if not clean_values:
        return None
    if len(clean_values) == 1:
        return clean_values[0]
    position = (len(clean_values) - 1) * (float(q) / 100.0)
    low = int(position)
    high = min(low + 1, len(clean_values) - 1)
    fraction = position - low
    return clean_values[low] * (1.0 - fraction) + clean_values[high] * fraction


def normalize_row(row, sample_to_split):
    sample_id = row.get("sample_id")
    if not sample_id:
        raise ValueError(f"Missing sample_id in row from {row.get('_source_csv')}")
    split = row.get("split") or sample_to_split.get(sample_id)
    return {
        "sample_id": sample_id,
        "case_id": row.get("case_id", ""),
        "split": split,
        "source_csv": row.get("_source_csv"),
        "threshold": as_float(row.get("threshold")),
        "postprocess": row.get("postprocess", ""),
        "dice": as_float(row.get("dice")),
        "iou": as_float(row.get("iou")),
        "hd": as_float(row.get("hd")),
        "hd95": as_float(row.get("hd95")),
        "assd": as_float(row.get("assd")),
        "precision": as_float(row.get("precision")),
        "recall": as_float(row.get("recall")),
        "correct_topology": as_float(row.get("correct_topology")),
        "pred_components": as_int(row.get("pred_components")),
        "pred_holes": as_int(row.get("pred_holes")),
        "target_components": as_int(row.get("target_components")),
        "target_holes": as_int(row.get("target_holes")),
    }


def hard_reasons(row, hd_cutoff, hd95_cutoff, hd_threshold, hd95_threshold, top_percent):
    reasons = []
    correct_topology = row["correct_topology"]
    if correct_topology is not None and correct_topology < 1.0:
        reasons.append("topology_failure")

    if (
        row["pred_components"] is not None
        and row["target_components"] is not None
        and row["pred_components"] != row["target_components"]
    ):
        reasons.append("component_mismatch")

    if (
        row["pred_holes"] is not None
        and row["target_holes"] is not None
        and row["pred_holes"] != row["target_holes"]
    ):
        reasons.append("hole_mismatch")

    hd = row["hd"]
    if hd is not None and hd_cutoff is not None and hd >= hd_cutoff:
        reasons.append(f"hd_top_{top_percent:g}_percent")
    if hd is not None and hd_threshold is not None and hd >= hd_threshold:
        reasons.append("hd_threshold")

    hd95 = row["hd95"]
    if hd95 is not None and hd95_cutoff is not None and hd95 >= hd95_cutoff:
        reasons.append(f"hd95_top_{top_percent:g}_percent")
    if hd95 is not None and hd95_threshold is not None and hd95 >= hd95_threshold:
        reasons.append("hd95_threshold")

    return reasons


def build_manifest(args):
    sample_to_split = load_sample_split_map(args.split_manifest)
    raw_rows = read_csv_rows(args.input_csv)
    rows = [normalize_row(row, sample_to_split) for row in raw_rows]
    source_split_counts = Counter(row["split"] or "unknown" for row in rows)

    if args.filter_to_split:
        candidate_rows = [row for row in rows if row["split"] == args.filter_to_split]
    else:
        candidate_rows = rows

    if args.cutoff_reference_csv:
        reference_raw_rows = read_csv_rows(args.cutoff_reference_csv)
        reference_rows = [normalize_row(row, sample_to_split) for row in reference_raw_rows]
        if args.reference_split:
            reference_rows = [row for row in reference_rows if row["split"] == args.reference_split]
    else:
        reference_rows = candidate_rows

    reference_split_counts = Counter(row["split"] or "unknown" for row in reference_rows)
    hd_cutoff = percentile([row["hd"] for row in reference_rows], 100.0 - args.top_percent)
    hd95_cutoff = percentile([row["hd95"] for row in reference_rows], 100.0 - args.top_percent)

    grouped = {}
    evidence_counts = defaultdict(Counter)
    for row in candidate_rows:
        reasons = hard_reasons(
            row,
            hd_cutoff,
            hd95_cutoff,
            args.hd_threshold,
            args.hd95_threshold,
            args.top_percent,
        )
        if not reasons:
            continue

        sample_id = row["sample_id"]
        if sample_id not in grouped:
            grouped[sample_id] = {
                "sample_id": sample_id,
                "case_id": row["case_id"],
                "split": row["split"],
                "reasons": [],
                "metrics": {
                    "dice": row["dice"],
                    "iou": row["iou"],
                    "hd": row["hd"],
                    "hd95": row["hd95"],
                    "assd": row["assd"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "correct_topology": row["correct_topology"],
                    "pred_components": row["pred_components"],
                    "pred_holes": row["pred_holes"],
                    "target_components": row["target_components"],
                    "target_holes": row["target_holes"],
                },
                "evidence": [],
            }
        item = grouped[sample_id]
        item["reasons"] = sorted(set(item["reasons"]) | set(reasons))
        item["evidence"].append(
            {
                "source_csv": row["source_csv"],
                "threshold": row["threshold"],
                "postprocess": row["postprocess"],
                "reasons": reasons,
            }
        )
        for reason in reasons:
            evidence_counts[sample_id][reason] += 1

    hard_samples = sorted(grouped.values(), key=lambda item: item["sample_id"])
    hard_sample_ids = [item["sample_id"] for item in hard_samples]
    hard_split_counts = Counter(item["split"] or "unknown" for item in hard_samples)
    reason_counts = Counter(reason for item in hard_samples for reason in item["reasons"])

    return {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "source_csvs": [str(Path(path)) for path in args.input_csv],
        "cutoff_reference_csvs": [str(Path(path)) for path in args.cutoff_reference_csv or []],
        "split_manifest": str(Path(args.split_manifest)),
        "selection": {
            "filter_to_split": args.filter_to_split,
            "reference_split": args.reference_split,
            "top_percent": float(args.top_percent),
            "hd_cutoff": hd_cutoff,
            "hd95_cutoff": hd95_cutoff,
            "hd_threshold": args.hd_threshold,
            "hd95_threshold": args.hd95_threshold,
            "rules": [
                "correct_topology == 0",
                "pred_components != target_components",
                "pred_holes != target_holes",
                "hd >= validation/source split top-percent cutoff or hd threshold",
                "hd95 >= validation/source split top-percent cutoff or hd95 threshold",
            ],
        },
        "counts": {
            "source_rows": len(rows),
            "candidate_rows": len(candidate_rows),
            "reference_rows": len(reference_rows),
            "hard_sample_count": len(hard_samples),
            "source_split_counts": dict(sorted(source_split_counts.items())),
            "reference_split_counts": dict(sorted(reference_split_counts.items())),
            "hard_split_counts": dict(sorted(hard_split_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "hard_sample_ids": hard_sample_ids,
        "hard_samples": hard_samples,
    }


def main():
    args = parse_args()
    manifest = build_manifest(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    counts = manifest["counts"]
    print(
        "Wrote {path} | hard={hard}/{candidates} | splits={splits}".format(
            path=output_path,
            hard=counts["hard_sample_count"],
            candidates=counts["candidate_rows"],
            splits=counts["hard_split_counts"],
        )
    )


if __name__ == "__main__":
    main()
