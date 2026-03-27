import random

import torch
from torch.utils.data import DataLoader


def _build_loader_kwargs(batch_size, worker_count, shuffle):
    worker_count = max(0, int(worker_count))
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": shuffle,
        "num_workers": worker_count,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if worker_count > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def _single_item_collate(batch):
    return batch[0]


def _split_acdc_patients(manifest, validation_ratio, seed):
    training_patients = sorted(
        {
            record["patient_id"]
            for record in manifest["records"]
            if record["source_subset"] == "training"
        }
    )
    testing_patients = sorted(
        {
            record["patient_id"]
            for record in manifest["records"]
            if record["source_subset"] == "testing"
        }
    )

    if len(training_patients) < 2:
        raise RuntimeError("ACDC training 病人数不足，无法继续划分训练集与验证集。")
    if not testing_patients:
        raise RuntimeError("ACDC testing 样本为空，无法继续评估。")

    shuffled_patients = list(training_patients)
    random.Random(seed).shuffle(shuffled_patients)
    validation_count = max(1, int(len(shuffled_patients) * validation_ratio))
    validation_count = min(validation_count, len(shuffled_patients) - 1)

    validation_patients = sorted(shuffled_patients[:validation_count])
    train_patients = sorted(shuffled_patients[validation_count:])
    return {
        "train": train_patients,
        "validation": validation_patients,
        "test": testing_patients,
    }


def _filter_records(records, patient_ids, source_subset):
    patient_lookup = set(patient_ids)
    return [
        record
        for record in records
        if record["source_subset"] == source_subset and record["patient_id"] in patient_lookup
    ]


def setup_acdc_dataloader(params, stages=None):
    from dataloaders.ACDC import ACDC_dataclass

    if stages is None:
        stages = ["train", "validation", "test"]

    manifest = ACDC_dataclass.get_manifest(params.dataset.processed_data_path)
    split_patients = _split_acdc_patients(
        manifest=manifest,
        validation_ratio=params.dataset.validation_ratio,
        seed=params.seed,
    )
    records = manifest["records"]

    train_records = _filter_records(records, split_patients["train"], source_subset="training")
    validation_records = _filter_records(records, split_patients["validation"], source_subset="training")
    test_records = _filter_records(records, split_patients["test"], source_subset="testing")

    params.acdc_manifest_summary = manifest["summary"]
    params.acdc_split_counts = {
        "train_patients": len(split_patients["train"]),
        "validation_patients": len(split_patients["validation"]),
        "test_patients": len(split_patients["test"]),
        "train_slices": len(train_records),
        "validation_slices": len(validation_records),
        "test_slices": len(test_records),
    }

    dataloader_dict = {}
    if "train" in stages:
        dataloader_dict["train"] = DataLoader(
            ACDC_dataclass(params, train_records, subset="Train", return_metadata=False),
            **_build_loader_kwargs(params.batch, params.num_workers, shuffle=True),
        )
    if "validation" in stages:
        dataloader_dict["validation"] = DataLoader(
            ACDC_dataclass(params, validation_records, subset="Validation", return_metadata=False),
            **_build_loader_kwargs(params.batch, params.num_workers, shuffle=False),
        )
    if "test" in stages:
        dataloader_dict["test"] = DataLoader(
            ACDC_dataclass(params, test_records, subset="Test", return_metadata=True),
            collate_fn=_single_item_collate,
            **_build_loader_kwargs(params.eval_batch, params.num_workers, shuffle=False),
        )
    return dataloader_dict
