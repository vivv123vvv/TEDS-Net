# PR Notes: ACDC benchmarking and local report automation

## Suggested title

Add reproducible ACDC benchmark reporting and formal baseline workflow

## Summary

This PR turns the ACDC training and evaluation flow into a reproducible benchmark pipeline. It fixes the data split, records per-epoch training throughput and memory, records per-sample and per-case evaluation metrics, and writes local benchmark reports under `reports/benchmarks/<run_name>/`. A comparison script is included so future experiments can be measured against the same baseline fields.

## What changed

- Added a fixed split manifest at `parameters/acdc_split.json`
- Extended `ACDCNpzDataset` to support fixed file lists and metadata (`sample_id`, `case_id`)
- Added shared benchmark helpers in `utils/acdc_benchmark.py`
- Updated `trainACDC.py` to:
  - accept benchmark-oriented CLI flags
  - write `train_epochs.csv` and `train_summary.json`
  - save run-specific checkpoints
  - auto-run evaluation after training
  - auto-refresh comparison outputs
- Updated `evaluate_results.py` to:
  - accept benchmark-oriented CLI flags
  - write `eval_per_sample.csv`, `eval_per_case.csv`, and `eval_summary.json`
  - support warmup batches and bounded evaluation
- Added `scripts/compare_benchmarks.py` for benchmark aggregation
- Documented the workflow and baseline results in `doc/acdc_benchmark_reports.md`
- Ignored local benchmark artifacts and local caches in `.gitignore`

## Validation

- Smoke training run with auto-evaluation completed successfully
- Standalone evaluation run completed successfully
- Comparison aggregation across multiple runs completed successfully
- Formal ACDC baseline run completed successfully on `2026-04-17`

## Formal baseline conclusion

Run name: `acdc-formal-20260417`

- Best validation Dice: `0.8819`
- Test Dice: `0.8649`
- Mean epoch time: `7.476 s`
- Mean forward time: `5.16 ms`
- P50 / P95 forward time: `4.68 ms / 9.02 ms`
- Peak GPU memory train / eval: `222.04 MB / 48.72 MB`
- Jacobian `< 0` ratio: `0.0`

## Notes

- Full benchmark artifacts were generated locally and are intentionally not committed.
- Local outputs for the formal run are stored in `reports/benchmarks/acdc-formal-20260417/`.
