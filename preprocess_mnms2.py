import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from skimage import measure
from tqdm import tqdm


DEFAULT_RAW_ROOT = (
    Path("Resources")
    / "database"
    / "kagglehub_cache"
    / "datasets"
    / "tailength"
    / "m-and-m2-dataset"
    / "versions"
    / "1"
    / "MnM2"
)
DEFAULT_OUTPUT_DIR = Path("Resources") / "database" / "mnms2_processed_2d"
DEFAULT_SPLIT_JSON = Path("parameters") / "mnms2_split.json"
DEFAULT_SPLIT_RATIOS = {"train": 0.60, "val": 0.20, "test": 0.20}

TARGET_SHAPE = (144, 208)
PRIOR_RADIUS = 35
PRIOR_THICKNESS = 7
MYO_LABEL = 2
PHASES = ("ED", "ES")
OFFICIAL_SPLIT_FIELDS = ("split", "subset", "partition", "set", "challenge_split", "data_split")
METADATA_FIELDS = {
    "patient_id": ("SUBJECT_CODE_PADDED", "SUBJECT_CODE", "patient_id"),
    "pathology": ("DISEASE", "Disease", "disease", "pathology"),
    "vendor": ("VENDOR", "Vendor", "vendor"),
    "scanner": ("SCANNER", "Scanner", "scanner"),
    "field_strength": ("FIELD", "Field", "field", "field_strength"),
}


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv_rows(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def markdown_table(rows, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines


def write_markdown(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def counter_to_prefixed_rows(split_name, category, counter):
    return [
        {
            "split": split_name,
            "category": category,
            "value": value,
            "count": count,
        }
        for value, count in sorted(counter.items())
    ]


def normalize_split_name(value):
    normalized = str(value).strip().lower()
    if normalized in {"train", "training", "tr"}:
        return "train"
    if normalized in {"val", "valid", "validation", "dev"}:
        return "val"
    if normalized in {"test", "testing", "ts"}:
        return "test"
    return None


def read_metadata(metadata_path):
    rows = []
    with Path(metadata_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject = str(row.get("SUBJECT_CODE", "")).strip()
            if not subject:
                continue
            row = {key: value for key, value in row.items()}
            row["SUBJECT_CODE_PADDED"] = f"{int(float(subject)):03d}"
            rows.append(row)
    return rows


def metadata_value(row, logical_name, default="UNKNOWN"):
    for key in METADATA_FIELDS[logical_name]:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def normalized_metadata(row):
    return {
        "patient_id": metadata_value(row, "patient_id"),
        "pathology": metadata_value(row, "pathology"),
        "vendor": metadata_value(row, "vendor"),
        "scanner": metadata_value(row, "scanner"),
        "field_strength": metadata_value(row, "field_strength"),
    }


def metadata_by_case(metadata_rows):
    return {
        row["SUBJECT_CODE_PADDED"]: normalized_metadata(row)
        for row in metadata_rows
    }


def official_case_splits(metadata_rows):
    if not metadata_rows:
        return None, None

    lower_to_actual = {key.lower(): key for key in metadata_rows[0].keys() if key}
    for field in OFFICIAL_SPLIT_FIELDS:
        actual_field = lower_to_actual.get(field)
        if not actual_field:
            continue

        splits = {"train": [], "val": [], "test": []}
        recognized = 0
        for row in metadata_rows:
            split_name = normalize_split_name(row.get(actual_field, ""))
            if split_name is None:
                continue
            splits[split_name].append(row["SUBJECT_CODE_PADDED"])
            recognized += 1

        if recognized == len(metadata_rows) and all(splits[name] for name in splits):
            return splits, actual_field
    return None, None


def sequential_case_splits(metadata_rows):
    case_ids = [row["SUBJECT_CODE_PADDED"] for row in metadata_rows]
    return {
        "train": case_ids[:160],
        "val": case_ids[160:200],
        "test": case_ids[200:360],
    }


def split_sizes(total_count, ratios):
    normalized = {name: float(ratios[name]) for name in ("train", "val", "test")}
    ratio_sum = sum(normalized.values())
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    normalized = {name: value / ratio_sum for name, value in normalized.items()}

    raw_sizes = {name: total_count * normalized[name] for name in normalized}
    sizes = {name: int(raw_sizes[name]) for name in normalized}
    remainder = total_count - sum(sizes.values())
    by_fraction = sorted(
        normalized,
        key=lambda name: (raw_sizes[name] - sizes[name], normalized[name]),
        reverse=True,
    )
    for idx in range(remainder):
        sizes[by_fraction[idx % len(by_fraction)]] += 1
    return sizes, normalized


def stratified_split_score(assignments, rows_by_case, target_sizes, ratios):
    attr_keys = ("pathology", "vendor", "field_strength")
    totals = {key: Counter() for key in attr_keys}
    split_counts = {split_name: 0 for split_name in target_sizes}
    split_attr_counts = {
        split_name: {key: Counter() for key in attr_keys}
        for split_name in target_sizes
    }

    for case_id, split_name in assignments.items():
        meta = normalized_metadata(rows_by_case[case_id])
        split_counts[split_name] += 1
        for key in attr_keys:
            totals[key][meta[key]] += 1
            split_attr_counts[split_name][key][meta[key]] += 1

    score = 0.0
    for split_name, target_size in target_sizes.items():
        score += 5.0 * ((split_counts[split_name] - target_size) ** 2)

    for split_name in target_sizes:
        ratio = ratios[split_name]
        for key in attr_keys:
            for value, total in totals[key].items():
                target = total * ratio
                actual = split_attr_counts[split_name][key][value]
                score += ((actual - target) ** 2) / max(target, 1.0)
                if total >= len(target_sizes) and actual == 0:
                    score += 1000.0
    return score


def balance_error(actual, target):
    return ((actual - target) ** 2) / max(target, 1.0)


def make_stratified_case_splits(metadata_rows, ratios=None, seed=42, attempts=256):
    ratios = ratios or DEFAULT_SPLIT_RATIOS
    rows_by_case = {row["SUBJECT_CODE_PADDED"]: row for row in metadata_rows}
    target_sizes, normalized_ratios = split_sizes(len(metadata_rows), ratios)
    attr_keys = ("pathology", "vendor", "field_strength")
    metadata_items = []
    total_attr_counts = {key: Counter() for key in attr_keys}
    combo_counts = Counter()
    pathology_groups = defaultdict(list)

    for row in metadata_rows:
        case_id = row["SUBJECT_CODE_PADDED"]
        meta = normalized_metadata(row)
        combo = tuple(meta[key] for key in attr_keys)
        combo_counts[combo] += 1
        for key in attr_keys:
            total_attr_counts[key][meta[key]] += 1
        metadata_items.append((case_id, meta, combo))
        pathology_groups[meta["pathology"]].append((case_id, meta, combo))

    pathology_targets = {
        pathology: split_sizes(len(items), normalized_ratios)[0]
        for pathology, items in pathology_groups.items()
    }

    best_assignments = None
    best_score = float("inf")
    split_names = ("train", "val", "test")

    for attempt_idx in range(max(1, int(attempts))):
        rng = random.Random(int(seed) + attempt_idx)
        assignments = {}
        split_case_counts = Counter()
        split_attr_counts = {
            split_name: {key: Counter() for key in attr_keys}
            for split_name in split_names
        }

        pathologies = list(pathology_groups.keys())
        rng.shuffle(pathologies)
        pathologies.sort(key=lambda pathology: (len(pathology_groups[pathology]), rng.random()))

        for pathology in pathologies:
            items = list(pathology_groups[pathology])
            rng.shuffle(items)
            items.sort(
                key=lambda item: (
                    total_attr_counts["field_strength"][item[1]["field_strength"]],
                    total_attr_counts["vendor"][item[1]["vendor"]],
                    combo_counts[item[2]],
                    rng.random(),
                )
            )
            remaining_pathology_quota = Counter(pathology_targets[pathology])

            for case_id, meta, _ in items:
                scores = []
                for split_name in split_names:
                    if remaining_pathology_quota[split_name] <= 0:
                        continue
                    if split_case_counts[split_name] >= target_sizes[split_name]:
                        continue

                    current_size = split_case_counts[split_name]
                    projected_size = current_size + 1
                    score = 8.0 * (
                        balance_error(projected_size, target_sizes[split_name])
                        - balance_error(current_size, target_sizes[split_name])
                    )
                    ratio = normalized_ratios[split_name]
                    for key, weight in (("vendor", 2.0), ("field_strength", 4.0)):
                        value = meta[key]
                        target = total_attr_counts[key][value] * ratio
                        current = split_attr_counts[split_name][key][value]
                        actual = current + 1
                        score += weight * (
                            balance_error(actual, target) - balance_error(current, target)
                        )
                    combo_target = combo_counts[tuple(meta[key] for key in attr_keys)] * ratio
                    current_combo = sum(
                        1
                        for assigned_case, assigned_split in assignments.items()
                        if assigned_split == split_name
                        and tuple(normalized_metadata(rows_by_case[assigned_case])[key] for key in attr_keys)
                        == tuple(meta[key] for key in attr_keys)
                    )
                    score += 0.5 * (
                        balance_error(current_combo + 1, combo_target)
                        - balance_error(current_combo, combo_target)
                    )
                    scores.append((score + rng.random() * 1e-6, split_name))

                if not scores:
                    raise RuntimeError(
                        f"Could not assign case {case_id} while building stratified split."
                    )

                _, chosen_split = min(scores)
                assignments[case_id] = chosen_split
                remaining_pathology_quota[chosen_split] -= 1
                split_case_counts[chosen_split] += 1
                for key in attr_keys:
                    split_attr_counts[chosen_split][key][meta[key]] += 1

        score = stratified_split_score(assignments, rows_by_case, target_sizes, normalized_ratios)
        if score < best_score:
            best_score = score
            best_assignments = assignments

    splits = {split_name: [] for split_name in split_names}
    for case_id, split_name in sorted(best_assignments.items()):
        splits[split_name].append(case_id)
    return splits, {
        "algorithm": "greedy_multifactor_stratification",
        "seed": int(seed),
        "attempts": int(attempts),
        "ratios": normalized_ratios,
        "target_case_counts": target_sizes,
        "pathology_target_counts": pathology_targets,
        "objective_score": best_score,
        "stratification_fields": list(attr_keys),
    }


def load_case_splits(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "case_splits" in payload:
        return {
            split_name: list(payload["case_splits"].get(split_name, []))
            for split_name in ("train", "val", "test")
        }
    return {
        split_name: list(payload.get(split_name, []))
        for split_name in ("train", "val", "test")
    }


def assert_no_patient_leakage(case_splits):
    seen = {}
    leakage = []
    for split_name, case_ids in case_splits.items():
        for case_id in case_ids:
            if case_id in seen:
                leakage.append({"case_id": case_id, "splits": [seen[case_id], split_name]})
            seen[case_id] = split_name
    if leakage:
        raise ValueError(f"Patient leakage detected across splits: {leakage[:10]}")


def build_split_stats(case_splits, case_metadata_map, sample_records, slice_records):
    stats = {}
    sample_by_split = defaultdict(list)
    slice_by_split = defaultdict(list)
    for record in sample_records:
        sample_by_split[record["split"]].append(record)
    for record in slice_records:
        slice_by_split[record["split"]].append(record)

    for split_name in ("train", "val", "test"):
        case_ids = list(case_splits.get(split_name, []))
        split_case_meta = [case_metadata_map[case_id] for case_id in case_ids]
        split_samples = sample_by_split[split_name]
        split_slices = slice_by_split[split_name]
        phase_saved_counts = Counter(record["phase"] for record in split_samples)
        phase_seen_counts = Counter(record["phase"] for record in split_slices)
        stats[split_name] = {
            "case_count": len(case_ids),
            "sample_count": len(split_samples),
            "slice_seen_count": len(split_slices),
            "pathology": dict(sorted(Counter(meta["pathology"] for meta in split_case_meta).items())),
            "vendor": dict(sorted(Counter(meta["vendor"] for meta in split_case_meta).items())),
            "field_strength": dict(sorted(Counter(meta["field_strength"] for meta in split_case_meta).items())),
            "ed_saved_slices": int(phase_saved_counts.get("ED", 0)),
            "es_saved_slices": int(phase_saved_counts.get("ES", 0)),
            "ed_seen_slices": int(phase_seen_counts.get("ED", 0)),
            "es_seen_slices": int(phase_seen_counts.get("ES", 0)),
            "myo_effective_slices": int(sum(1 for record in split_samples if int(record["has_myo"]) == 1)),
            "empty_myo_slices_seen": int(sum(1 for record in split_slices if int(record["has_myo"]) == 0)),
            "topology_abnormal_slices_seen": int(
                sum(
                    1
                    for record in split_slices
                    if int(record["has_myo"]) == 1 and int(record["target_topology_ok"]) == 0
                )
            ),
            "topology_filtered_slices": int(
                sum(1 for record in split_slices if record.get("skip_reason") == "topology_filter")
            ),
        }
    return stats


def split_overview_rows(split_stats):
    rows = []
    for split_name in ("train", "val", "test"):
        stats = split_stats[split_name]
        rows.append(
            {
                "split": split_name,
                "case_count": stats["case_count"],
                "sample_count": stats["sample_count"],
                "slice_seen_count": stats["slice_seen_count"],
                "ed_saved_slices": stats["ed_saved_slices"],
                "es_saved_slices": stats["es_saved_slices"],
                "myo_effective_slices": stats["myo_effective_slices"],
                "empty_myo_slices_seen": stats["empty_myo_slices_seen"],
                "topology_abnormal_slices_seen": stats["topology_abnormal_slices_seen"],
                "topology_filtered_slices": stats["topology_filtered_slices"],
            }
        )
    return rows


def split_distribution_rows(split_stats):
    rows = []
    for split_name in ("train", "val", "test"):
        stats = split_stats[split_name]
        rows.extend(counter_to_prefixed_rows(split_name, "pathology", stats["pathology"]))
        rows.extend(counter_to_prefixed_rows(split_name, "vendor", stats["vendor"]))
        rows.extend(counter_to_prefixed_rows(split_name, "field_strength", stats["field_strength"]))
    return rows


def case_split_rows(case_splits, case_metadata_map):
    rows = []
    for split_name in ("train", "val", "test"):
        for case_id in case_splits.get(split_name, []):
            meta = case_metadata_map[case_id]
            rows.append(
                {
                    "split": split_name,
                    "patient_id": case_id,
                    "pathology": meta["pathology"],
                    "vendor": meta["vendor"],
                    "scanner": meta["scanner"],
                    "field_strength": meta["field_strength"],
                }
            )
    return rows


def write_split_artifacts(
    output_dir,
    split_json,
    case_splits,
    case_metadata_map,
    sample_records,
    slice_records,
    split_stats,
    case_split_csv=None,
    slice_records_csv=None,
):
    output_dir = Path(output_dir)
    stats_dir = output_dir / "split_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    overview_rows = split_overview_rows(split_stats)
    distribution_rows = split_distribution_rows(split_stats)
    case_rows = case_split_rows(case_splits, case_metadata_map)

    overview_fields = [
        "split",
        "case_count",
        "sample_count",
        "slice_seen_count",
        "ed_saved_slices",
        "es_saved_slices",
        "myo_effective_slices",
        "empty_myo_slices_seen",
        "topology_abnormal_slices_seen",
        "topology_filtered_slices",
    ]
    distribution_fields = ["split", "category", "value", "count"]
    case_fields = ["split", "patient_id", "pathology", "vendor", "scanner", "field_strength"]
    sample_fields = [
        "split",
        "patient_id",
        "phase",
        "slice_id",
        "sample_id",
        "file_name",
        "pathology",
        "vendor",
        "scanner",
        "field_strength",
        "spacing_x",
        "spacing_y",
        "spacing_z",
        "source_height",
        "source_width",
        "source_slices",
        "has_myo",
        "myo_pixels",
        "target_components",
        "target_holes",
        "target_topology_ok",
        "saved",
        "skip_reason",
    ]

    write_csv_rows(stats_dir / "split_overview.csv", overview_fields, overview_rows)
    write_csv_rows(stats_dir / "split_distributions.csv", distribution_fields, distribution_rows)
    write_csv_rows(stats_dir / "case_split.csv", case_fields, case_rows)
    write_csv_rows(stats_dir / "sample_records.csv", sample_fields, sample_records)
    write_csv_rows(stats_dir / "slice_records.csv", sample_fields, slice_records)
    if case_split_csv:
        write_csv_rows(case_split_csv, case_fields, case_rows)
    if slice_records_csv:
        write_csv_rows(slice_records_csv, sample_fields, slice_records)

    lines = [
        "# M&Ms-2 Split / Preprocess Summary",
        "",
        f"- split manifest: `{Path(split_json)}`",
        f"- processed dataset: `{output_dir}`",
        "",
        "## Overview",
        "",
        *markdown_table(overview_rows, overview_fields),
        "",
        "## Distributions",
        "",
        *markdown_table(distribution_rows, distribution_fields),
        "",
        "## Files",
        "",
        f"- overview csv: `{stats_dir / 'split_overview.csv'}`",
        f"- distribution csv: `{stats_dir / 'split_distributions.csv'}`",
        f"- case split csv: `{stats_dir / 'case_split.csv'}`",
        f"- sample records csv: `{stats_dir / 'sample_records.csv'}`",
        f"- slice records csv: `{stats_dir / 'slice_records.csv'}`",
    ]
    write_markdown(stats_dir / "split_stats.md", lines)


def select_case_subset(case_splits, max_cases=None, max_cases_per_split=None):
    if max_cases_per_split is not None:
        return {
            split_name: case_ids[:max_cases_per_split]
            for split_name, case_ids in case_splits.items()
        }

    if max_cases is None:
        return {split_name: list(case_ids) for split_name, case_ids in case_splits.items()}

    selected = set()
    remaining = int(max_cases)
    for split_name in ("train", "val", "test"):
        for case_id in case_splits.get(split_name, []):
            if remaining <= 0:
                break
            selected.add(case_id)
            remaining -= 1
        if remaining <= 0:
            break

    return {
        split_name: [case_id for case_id in case_ids if case_id in selected]
        for split_name, case_ids in case_splits.items()
    }


def create_prior(shape, radius, thickness):
    h, w = shape
    prior = np.zeros(shape, dtype=np.float32)
    center = (w // 2, h // 2)
    cv2.circle(prior, center, radius, 1, thickness)
    return prior


def crop_or_pad_center(array, target_shape, pad_value=0):
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D slice, got shape {array.shape}")

    target_h, target_w = target_shape
    result = np.full((target_h, target_w), pad_value, dtype=array.dtype)
    src_slices = []
    dst_slices = []

    for src_size, target_size in zip(array.shape, target_shape):
        if src_size >= target_size:
            src_start = (src_size - target_size) // 2
            src_end = src_start + target_size
            dst_start = 0
            dst_end = target_size
        else:
            src_start = 0
            src_end = src_size
            dst_start = (target_size - src_size) // 2
            dst_end = dst_start + src_size
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    result[tuple(dst_slices)] = array[tuple(src_slices)]
    return result


def normalize_image(image):
    image = np.nan_to_num(image.astype(np.float32), copy=False)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)
    return image.astype(np.float32, copy=False)


def normalize_image_robust(image, lower_percentile=1.0, upper_percentile=99.0):
    image = np.nan_to_num(image.astype(np.float32), copy=False)
    finite_values = image[np.isfinite(image)]
    if finite_values.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low = float(np.percentile(finite_values, lower_percentile))
    high = float(np.percentile(finite_values, upper_percentile))
    if high <= low:
        return normalize_image(image)
    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    return image.astype(np.float32, copy=False)


def normalize_image_by_mode(image, mode):
    mode = str(mode).strip().lower()
    if mode == "minmax":
        return normalize_image(image)
    if mode == "robust":
        return normalize_image_robust(image)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def calculate_betti_numbers(binary_mask):
    binary_mask = np.asarray(binary_mask > 0, dtype=bool)
    if not binary_mask.any():
        return 0, 0

    _, component_count = measure.label(binary_mask, return_num=True, connectivity=1)
    euler_number = measure.euler_number(binary_mask, connectivity=1)
    holes = component_count - euler_number
    return int(component_count), int(holes)


def expected_phase_paths(case_dir, case_id, phase):
    return (
        case_dir / f"{case_id}_SA_{phase}.nii",
        case_dir / f"{case_id}_SA_{phase}_gt.nii",
    )


def process_phase(
    case_id,
    split_name,
    phase,
    raw_dataset_dir,
    output_dir,
    prior,
    target_shape,
    myo_label,
    use_topology_filter,
    include_empty_slices,
    normalization,
    case_metadata,
    split_files,
    sample_records,
    slice_records,
    summary,
):
    case_dir = raw_dataset_dir / case_id
    image_path, label_path = expected_phase_paths(case_dir, case_id, phase)
    if not image_path.exists() or not label_path.exists():
        summary["missing_pairs"].append(
            {
                "case_id": case_id,
                "phase": phase,
                "image_path": str(image_path),
                "label_path": str(label_path),
            }
        )
        return

    image_nii = nib.load(str(image_path))
    label_nii = nib.load(str(label_path))
    image_volume = image_nii.get_fdata(dtype=np.float32)
    label_volume = label_nii.get_fdata(dtype=np.float32)
    if image_volume.shape != label_volume.shape:
        raise ValueError(f"Image/label shape mismatch for {case_id} {phase}: {image_volume.shape} vs {label_volume.shape}")
    if image_volume.ndim != 3:
        raise ValueError(f"Expected 3D SA volume for {case_id} {phase}, got {image_volume.shape}")

    summary["source_shapes"][f"{case_id}_SA_{phase}"] = list(image_volume.shape)
    spacing = tuple(float(value) for value in image_nii.header.get_zooms()[:3])
    summary["source_spacings"][f"{case_id}_SA_{phase}"] = list(spacing)
    summary["source_label_values"].update(int(value) for value in np.unique(label_volume))

    for slice_idx in range(image_volume.shape[2]):
        summary["total_slices_seen"] += 1
        image_slice = crop_or_pad_center(image_volume[:, :, slice_idx], target_shape, pad_value=0)
        label_slice = crop_or_pad_center(label_volume[:, :, slice_idx], target_shape, pad_value=0)
        label_slice = np.rint(label_slice).astype(np.uint8, copy=False)
        myo_mask = label_slice == myo_label
        b0, b1 = calculate_betti_numbers(myo_mask)
        topology_ok = (b0, b1) == (1, 1)
        record = {
            "split": split_name,
            "patient_id": case_id,
            "phase": phase,
            "slice_id": int(slice_idx),
            "sample_id": f"{case_id}_SA_{phase}_slice{slice_idx:03d}",
            "file_name": f"{case_id}_SA_{phase}_slice{slice_idx:03d}.npz",
            "pathology": case_metadata["pathology"],
            "vendor": case_metadata["vendor"],
            "scanner": case_metadata["scanner"],
            "field_strength": case_metadata["field_strength"],
            "spacing_x": spacing[0],
            "spacing_y": spacing[1],
            "spacing_z": spacing[2],
            "source_height": int(image_volume.shape[0]),
            "source_width": int(image_volume.shape[1]),
            "source_slices": int(image_volume.shape[2]),
            "has_myo": int(bool(myo_mask.any())),
            "myo_pixels": int(myo_mask.sum()),
            "target_components": int(b0),
            "target_holes": int(b1),
            "target_topology_ok": int(topology_ok),
            "saved": 0,
            "skip_reason": "",
        }

        if not myo_mask.any():
            summary["empty_myo_slices"] += 1
            if not include_empty_slices:
                record["skip_reason"] = "empty_myo"
                slice_records.append(record)
                continue

        if use_topology_filter:
            if (b0, b1) != (1, 1):
                summary["topology_filtered_slices"] += 1
                record["skip_reason"] = "topology_filter"
                slice_records.append(record)
                continue
        elif not topology_ok and myo_mask.any():
            summary["topology_abnormal_slices"] += 1

        image_slice = normalize_image_by_mode(image_slice, normalization)
        output_name = record["file_name"]
        np.savez(
            output_dir / output_name,
            image=image_slice,
            label=label_slice,
            prior=prior,
            patient_id=np.asarray(case_id),
            phase=np.asarray(phase),
            slice_id=np.asarray(slice_idx, dtype=np.int16),
            pathology=np.asarray(case_metadata["pathology"]),
            vendor=np.asarray(case_metadata["vendor"]),
            scanner=np.asarray(case_metadata["scanner"]),
            field_strength=np.asarray(case_metadata["field_strength"]),
            spacing=np.asarray(spacing, dtype=np.float32),
            split=np.asarray(split_name),
            source_shape=np.asarray(image_volume.shape, dtype=np.int16),
            target_components=np.asarray(b0, dtype=np.int16),
            target_holes=np.asarray(b1, dtype=np.int16),
            target_topology_ok=np.asarray(int(topology_ok), dtype=np.uint8),
        )
        split_files[split_name].append(output_name)
        record["saved"] = 1
        sample_records.append(record.copy())
        slice_records.append(record)
        summary["saved_slices"] += 1
        summary["saved_by_split"][split_name] += 1
        summary["saved_by_case"][case_id] += 1


def preprocess_mnms2(args):
    raw_root = Path(args.raw_root)
    raw_dataset_dir = raw_root / "dataset"
    metadata_path = raw_root / "dataset_information.csv"
    readme_path = raw_root / "readme.txt"
    output_dir = Path(args.output_dir)
    split_json = Path(args.split_json)

    if not raw_dataset_dir.exists():
        raise FileNotFoundError(f"MnM2 dataset directory not found: {raw_dataset_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"dataset_information.csv not found: {metadata_path}")
    if not readme_path.exists():
        raise FileNotFoundError(f"readme.txt not found: {readme_path}")

    metadata_rows = read_metadata(metadata_path)
    case_metadata_map = metadata_by_case(metadata_rows)

    split_metadata = None
    if args.case_split_json:
        case_splits = load_case_splits(args.case_split_json)
        split_source = f"case_split_json:{args.case_split_json}"
        split_metadata = {"source_path": str(args.case_split_json)}
    elif args.stratified_split:
        case_splits, split_metadata = make_stratified_case_splits(
            metadata_rows,
            ratios={
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": args.test_ratio,
            },
            seed=args.seed,
            attempts=args.stratified_attempts,
        )
        split_source = "patient_level_stratified_metadata"
    else:
        official_splits, official_split_field = official_case_splits(metadata_rows)
        if official_splits:
            case_splits = official_splits
            split_source = f"metadata:{official_split_field}"
        else:
            case_splits = sequential_case_splits(metadata_rows)
            split_source = "readme_sequential_160_train_40_val_160_test"

    assert_no_patient_leakage(case_splits)

    selected_case_splits = select_case_subset(
        case_splits,
        max_cases=args.max_cases,
        max_cases_per_split=args.max_cases_per_split,
    )
    assert_no_patient_leakage(selected_case_splits)

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for npz_path in output_dir.glob("*.npz"):
            npz_path.unlink()

    prior = create_prior((args.target_height, args.target_width), args.prior_radius, args.prior_thickness)
    split_files = {"train": [], "val": [], "test": []}
    sample_records = []
    slice_records = []
    summary = {
        "dataset": "mnms2",
        "preprocess_version": args.preprocess_version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw_root": str(raw_root),
        "raw_dataset_dir": str(raw_dataset_dir),
        "metadata_path": str(metadata_path),
        "readme_path": str(readme_path),
        "output_dir": str(output_dir),
        "split_json": str(split_json),
        "split_source": split_source,
        "split_metadata": split_metadata,
        "naming_rule": "{case_id}/{case_id}_SA_{ED|ES}.nii with labels {case_id}_SA_{ED|ES}_gt.nii",
        "target_shape": [args.target_height, args.target_width],
        "myo_label": args.myo_label,
        "phases": list(PHASES),
        "use_topology_filter": not args.no_topology_filter,
        "include_empty_slices": args.include_empty_slices,
        "image_normalization": args.normalization,
        "spacing_policy": "record_only_no_resampling",
        "metadata_subject_count": len(metadata_rows),
        "case_counts": {name: len(values) for name, values in case_splits.items()},
        "selected_case_counts": {name: len(values) for name, values in selected_case_splits.items()},
        "total_slices_seen": 0,
        "empty_myo_slices": 0,
        "topology_filtered_slices": 0,
        "topology_abnormal_slices": 0,
        "saved_slices": 0,
        "saved_by_split": defaultdict(int),
        "saved_by_case": defaultdict(int),
        "source_shapes": {},
        "source_spacings": {},
        "source_label_values": set(),
        "missing_pairs": [],
    }

    for split_name in ("train", "val", "test"):
        for case_id in tqdm(selected_case_splits[split_name], desc=f"{split_name} cases"):
            if case_id not in case_metadata_map:
                raise KeyError(f"Missing metadata for case {case_id}")
            for phase in PHASES:
                process_phase(
                    case_id=case_id,
                    split_name=split_name,
                    phase=phase,
                    raw_dataset_dir=raw_dataset_dir,
                    output_dir=output_dir,
                    prior=prior,
                    target_shape=(args.target_height, args.target_width),
                    myo_label=args.myo_label,
                    use_topology_filter=not args.no_topology_filter,
                    include_empty_slices=args.include_empty_slices,
                    normalization=args.normalization,
                    case_metadata=case_metadata_map[case_id],
                    split_files=split_files,
                    sample_records=sample_records,
                    slice_records=slice_records,
                    summary=summary,
                )

    for split_name in split_files:
        split_files[split_name] = sorted(split_files[split_name])

    total_seen = summary["total_slices_seen"]
    nonempty_seen = total_seen - summary["empty_myo_slices"]
    summary["myo_nonempty_ratio"] = float(nonempty_seen / total_seen) if total_seen else None
    summary["saved_by_split"] = dict(summary["saved_by_split"])
    summary["saved_by_case"] = dict(summary["saved_by_case"])
    summary["source_label_values"] = sorted(summary["source_label_values"])
    summary["sample_counts"] = {name: len(values) for name, values in split_files.items()}
    split_stats = build_split_stats(selected_case_splits, case_metadata_map, sample_records, slice_records)

    manifest = {
        "version": 2 if args.preprocess_version == "v2" else 1,
        "dataset": "mnms2",
        "data_dir": str(output_dir),
        "created_at": summary["created_at"],
        "split_source": split_source,
        "split_metadata": split_metadata,
        "case_splits": selected_case_splits,
        "case_counts": summary["selected_case_counts"],
        "sample_counts": summary["sample_counts"],
        "split_stats": split_stats,
        "preprocess": {
            "version": args.preprocess_version,
            "target_shape": [args.target_height, args.target_width],
            "myo_label": args.myo_label,
            "label_rule": "MYO is label 2; dataloader converts label == 2 to binary target.",
            "view": "SA",
            "phases": list(PHASES),
            "crop_or_pad": "center",
            "image_normalization": args.normalization,
            "spacing_policy": "record_only_no_resampling",
            "prior": {
                "shape": [args.target_height, args.target_width],
                "radius": args.prior_radius,
                "thickness": args.prior_thickness,
            },
            "topology_filter": not args.no_topology_filter,
            "include_empty_slices": args.include_empty_slices,
        },
        "sample_records": sample_records,
        "splits": split_files,
    }

    write_json(split_json, manifest)
    write_json(output_dir / "preprocess_summary.json", summary)
    write_split_artifacts(
        output_dir=output_dir,
        split_json=split_json,
        case_splits=selected_case_splits,
        case_metadata_map=case_metadata_map,
        sample_records=sample_records,
        slice_records=slice_records,
        split_stats=split_stats,
        case_split_csv=args.case_split_csv,
        slice_records_csv=args.slice_records_csv,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Convert M&Ms-2 short-axis ED/ES NIfTI files to TEDS-Net 2D npz.")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split-json", default=str(DEFAULT_SPLIT_JSON))
    parser.add_argument("--case-split-json", default=None)
    parser.add_argument("--case-split-csv", default=None)
    parser.add_argument("--slice-records-csv", default=None)
    parser.add_argument("--stratified-split", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratified-attempts", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["train"])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["val"])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT_RATIOS["test"])
    parser.add_argument("--preprocess-version", choices=["legacy", "v2"], default="legacy")
    parser.add_argument("--normalization", choices=["minmax", "robust"], default="minmax")
    parser.add_argument("--target-height", type=int, default=TARGET_SHAPE[0])
    parser.add_argument("--target-width", type=int, default=TARGET_SHAPE[1])
    parser.add_argument("--prior-radius", type=int, default=PRIOR_RADIUS)
    parser.add_argument("--prior-thickness", type=int, default=PRIOR_THICKNESS)
    parser.add_argument("--myo-label", type=int, default=MYO_LABEL)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-cases-per-split", type=int, default=None)
    parser.add_argument("--include-empty-slices", action="store_true")
    parser.add_argument("--no-topology-filter", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    preprocess_mnms2(parse_args())
