import random

import torch
from torch.utils.data import DataLoader


def _build_loader_kwargs(batch_size, worker_count, shuffle):
    worker_count = max(0, worker_count)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": worker_count,
        "pin_memory": torch.cuda.is_available(),
    }
    if worker_count > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def _single_item_collate(batch):
    return batch[0]


def setup_mnist_dataloader(params, subset_list):
    from dataloaders.mnist import MNIST_dataclass as MyDataset

    dataloader_dict = {}
    if "train" in subset_list:
        training_set = MyDataset(params, subset="Train")
        dataloader_dict["train"] = DataLoader(
            training_set,
            **_build_loader_kwargs(params.batch, params.num_workers, shuffle=True),
        )
    if "validation" in subset_list:
        validation_set = MyDataset(params, subset="Validation")
        dataloader_dict["validation"] = DataLoader(
            validation_set,
            **_build_loader_kwargs(params.batch, params.num_workers, shuffle=False),
        )
    if "test" in subset_list:
        test_set = MyDataset(params, subset="Test")
        dataloader_dict["test"] = DataLoader(
            test_set,
            **_build_loader_kwargs(params.eval_batch, params.num_workers, shuffle=False),
        )
    return dataloader_dict


def _split_acdc_patients(manifest, validation_ratio, seed):
    training_patients = sorted(
        {
            record["patient_id"]
            for record in manifest["records"]
            if record["source_subset"] == "training"
        }
    )
    test_patients = sorted(
        {
            record["patient_id"]
            for record in manifest["records"]
            if record["source_subset"] == "testing"
        }
    )

    if len(training_patients) < 2:
        raise RuntimeError("ACDC training 病人数不足，无法继续划分 train / validation。")
    if not test_patients:
        raise RuntimeError("ACDC testing 缓存为空，无法构建测试集。")

    shuffled_patients = list(training_patients)
    random.Random(seed).shuffle(shuffled_patients)
    validation_count = max(1, int(len(shuffled_patients) * validation_ratio))
    validation_patients = sorted(shuffled_patients[:validation_count])
    train_patients = sorted(shuffled_patients[validation_count:])

    return {
        "train": train_patients,
        "validation": validation_patients,
        "test": test_patients,
    }


def _filter_records(records, patient_ids, source_subset):
    patient_lookup = set(patient_ids)
    return [
        record for record in records
        if record["source_subset"] == source_subset and record["patient_id"] in patient_lookup
    ]


def setup_acdc_dataloader(params, subset_list):
    from dataloaders.ACDC import ACDC_dataclass as MyDataset

    manifest = MyDataset.get_manifest(params.dataset.processed_data_path)
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
    if "train" in subset_list:
        training_set = MyDataset(params, train_records, subset="Train", return_metadata=False)
        dataloader_dict["train"] = DataLoader(
            training_set,
            **_build_loader_kwargs(params.batch, params.num_workers, shuffle=True),
        )

    if "validation" in subset_list:
        validation_set = MyDataset(params, validation_records, subset="Validation", return_metadata=False)
        dataloader_dict["validation"] = DataLoader(
            validation_set,
            **_build_loader_kwargs(params.batch, params.num_workers, shuffle=False),
        )

    if "test" in subset_list:
        test_set = MyDataset(params, test_records, subset="Test", return_metadata=True)
        dataloader_dict["test"] = DataLoader(
            test_set,
            collate_fn=_single_item_collate,
            **_build_loader_kwargs(1, params.num_workers, shuffle=False),
        )

    return dataloader_dict
