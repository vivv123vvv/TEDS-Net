from dataloaders.acdc_npz import ACDCNpzDataset


class ACDC_dataclass(ACDCNpzDataset):
    """保留原文件入口，底层实现复用 `.npz` 缓存读取器。"""

    def __init__(self, params, records, subset, return_metadata=False):
        self.params = params
        self.subset = subset
        super().__init__(
            processed_data_path=params.dataset.processed_data_path,
            records=records,
            return_metadata=return_metadata,
        )
