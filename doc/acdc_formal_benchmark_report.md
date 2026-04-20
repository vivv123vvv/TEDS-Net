# ACDC Formal Benchmark Report

## Overview

This report summarizes the formal ACDC benchmark run for the reproducible benchmarking workflow added in this PR.

- Run name: `acdc-formal-20260417`
- Run date: `2026-04-17`
- Device: `NVIDIA GeForce RTX 5060`
- Dataset split manifest: `parameters/acdc_split.json`
- Training samples: `1247`
- Validation samples: `356`
- Test samples: `179`
- Epochs: `200`
- Batch size: `5`
- Learning rate: `0.0001`

The benchmark artifacts were generated locally and intentionally excluded from Git. The local output directory is:

```text
reports/benchmarks/acdc-formal-20260417/
```

The best checkpoint is stored locally at:

```text
checkpoints/acdc-formal-20260417/best_teds_net.pth
```

## Metrics Summary

| Metric | Value |
| --- | ---: |
| Best validation Dice | `0.8819` |
| Test Dice | `0.8649` |
| Jacobian `< 0` ratio | `0.0` |
| Mean epoch time | `7.476 s` |
| Mean forward time | `5.16 ms` |
| P50 forward time | `4.68 ms` |
| P95 forward time | `9.02 ms` |
| Peak GPU memory during training | `222.04 MB` |
| Peak GPU memory during evaluation | `48.72 MB` |

## Result Interpretation

The formal run establishes a reproducible baseline for future ACDC experiments. The best validation Dice reached `0.8819`, and the held-out test Dice reached `0.8649`, which is the primary segmentation-quality metric for follow-up comparisons.

The deformation regularity result is strong for this run: the measured Jacobian `< 0` ratio is `0.0` on the fixed test split. This means the benchmark did not detect folding on the evaluated displacement fields, so future replacement experiments should preserve this result or explicitly justify any tradeoff.

The speed and memory numbers are lightweight on the current GPU. Training averaged `7.476 s` per epoch with a peak allocation of `222.04 MB`, while evaluation averaged `5.16 ms` per forward pass with `48.72 MB` peak memory. These values are now suitable as the baseline row for later integrator or architecture comparisons.

## Generated Local Artifacts

The formal run generated the following local files:

- `reports/benchmarks/acdc-formal-20260417/train_epochs.csv`
- `reports/benchmarks/acdc-formal-20260417/train_summary.json`
- `reports/benchmarks/acdc-formal-20260417/eval_per_sample.csv`
- `reports/benchmarks/acdc-formal-20260417/eval_per_case.csv`
- `reports/benchmarks/acdc-formal-20260417/eval_summary.json`
- `reports/benchmarks/comparison.csv`
- `reports/benchmarks/comparison.md`

These files are not committed because they are local experiment outputs. The PR commits only the benchmark pipeline, fixed split manifest, and report documentation.

## Reproduction Commands

Full training and automatic evaluation:

```powershell
C:\ProgramData\Anaconda3\envs\TEDS-Net\python.exe trainACDC.py --run-name acdc-formal-20260417
```

Standalone evaluation against the best checkpoint:

```powershell
C:\ProgramData\Anaconda3\envs\TEDS-Net\python.exe evaluate_results.py ^
  --checkpoint checkpoints\acdc-formal-20260417\best_teds_net.pth ^
  --run-name acdc-formal-20260417
```

Refresh the comparison table:

```powershell
C:\ProgramData\Anaconda3\envs\TEDS-Net\python.exe scripts\compare_benchmarks.py
```

## PR Conclusion

This PR is ready to serve as the baseline benchmarking foundation. Future experiments can add new run directories under `reports/benchmarks/<run_name>/` locally, then compare them against `acdc-formal-20260417` using `scripts/compare_benchmarks.py`.
