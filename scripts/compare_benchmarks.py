import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.acdc_benchmark import DEFAULT_REPORTS_DIR, discover_run_dirs, write_comparison_artifacts


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate benchmark summaries into comparison.csv and comparison.md.")
    parser.add_argument("--run-dirs", nargs="*", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_REPORTS_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    run_dirs = [Path(path) for path in args.run_dirs] if args.run_dirs else discover_run_dirs(output_dir)
    result = write_comparison_artifacts(run_dirs, output_dir)
    print(f"Wrote {len(result['rows'])} comparison row(s) to {result['csv_path']} and {result['md_path']}")


if __name__ == "__main__":
    main()
