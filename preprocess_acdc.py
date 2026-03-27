import argparse
import json
import os

from parameters.acdc_parameters import Parameters
from utils.acdc_preprocess import preprocess_acdc_dataset


def _resolve_path(project_root, path_value):
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(project_root, path_value))


def build_argument_parser():
    defaults = Parameters()
    parser = argparse.ArgumentParser(description="预处理官方 ACDC 数据集并生成 manifest。")
    parser.add_argument("--raw-data-path", default=defaults.dataset.raw_data_path, help="官方 ACDC 原始数据目录，支持传入 `Resources/` 或 `database/` 根目录。")
    parser.add_argument("--processed-data-path", default=defaults.dataset.processed_data_path, help="预处理缓存输出目录。")
    parser.add_argument("--force", action="store_true", help="强制覆盖已有预处理缓存与 manifest。")
    parser.add_argument("--limit-patients", type=int, default=0, help="仅处理前 N 个病人，默认 0 表示处理全部病人。")
    return parser


def main(args=None):
    project_root = os.path.abspath(os.path.dirname(__file__))
    parser = build_argument_parser()
    parsed_args = parser.parse_args(args=args)

    params = Parameters()
    raw_data_path = _resolve_path(project_root, parsed_args.raw_data_path)
    processed_data_path = _resolve_path(project_root, parsed_args.processed_data_path)

    manifest = preprocess_acdc_dataset(
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path,
        target_size=params.dataset.inshape,
        prior_radius=params.dataset.ps_meas[0],
        prior_thickness=params.dataset.ps_meas[1],
        force=parsed_args.force,
        limit_patients=parsed_args.limit_patients,
    )

    print("ACDC 预处理完成。")
    print(f"manifest: {manifest['manifest_path']}")
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
