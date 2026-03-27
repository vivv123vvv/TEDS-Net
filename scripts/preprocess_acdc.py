import argparse
import json

from parameters.acdc_parameters import Parameters
from utils.acdc_preprocess import preprocess_acdc_dataset


def main():
    parser = argparse.ArgumentParser(description="预处理官方 ACDC 数据集并生成 2D 缓存。")
    parser.add_argument("--raw-data-path", required=True, help="官方 ACDC 根目录，或包含 database/ 的上级目录。")
    parser.add_argument("--processed-data-path", required=True, help="预处理缓存输出目录。")
    parser.add_argument("--force", action="store_true", help="即使 manifest 已存在也强制重建缓存。")
    parser.add_argument("--limit-patients", type=int, default=0, help="仅处理前 N 个病人，便于 smoke test。")
    args = parser.parse_args()

    params = Parameters()
    manifest = preprocess_acdc_dataset(
        raw_data_path=args.raw_data_path,
        processed_data_path=args.processed_data_path,
        target_size=params.dataset.inshape,
        prior_radius=params.dataset.ps_meas[0],
        prior_thickness=params.dataset.ps_meas[1],
        force=args.force,
        limit_patients=args.limit_patients,
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
