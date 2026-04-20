# ACDC Benchmark Reports

## Default locations

- Split manifest: `parameters/acdc_split.json`
- Run reports: `reports/benchmarks/<run_name>/`
- Best checkpoint per run: `checkpoints/<run_name>/best_teds_net.pth`

## Train + auto-evaluate

Run from the repo root:

```powershell
C:\ProgramData\Anaconda3\envs\TEDS-Net\python.exe trainACDC.py --run-name acdc-baseline
```

Training writes:

- `train_epochs.csv`
- `train_summary.json`

After the best checkpoint is saved, the script automatically runs evaluation on the configured split and writes:

- `eval_per_sample.csv`
- `eval_per_case.csv`
- `eval_summary.json`

It also refreshes:

- `reports/benchmarks/comparison.csv`
- `reports/benchmarks/comparison.md`

## Smoke run

```powershell
C:\ProgramData\Anaconda3\envs\TEDS-Net\python.exe trainACDC.py ^
  --run-name smoke-acdc ^
  --epochs 2 ^
  --max-train-batches 2 ^
  --max-val-batches 2 ^
  --eval-max-samples 8
```

## Standalone evaluation

```powershell
C:\ProgramData\Anaconda3\envs\TEDS-Net\python.exe evaluate_results.py ^
  --checkpoint checkpoints\\acdc-baseline\\best_teds_net.pth ^
  --run-name acdc-baseline
```

If `--checkpoint` is omitted and `checkpoints\best_teds_net.pth` does not exist, the evaluator falls back to the most recently updated `checkpoints\<run_name>\best_teds_net.pth`.

## Manual comparison refresh

```powershell
C:\ProgramData\Anaconda3\envs\TEDS-Net\python.exe scripts\compare_benchmarks.py
```

## Formal baseline

Formal run name: `acdc-formal-20260417`

- Validation Dice: `0.8819`
- Mean epoch time: `7.476 s`
- Peak GPU memory during training: `222.04 MB`
- Test Dice: `0.8649`
- Mean forward time: `5.16 ms`
- P50 / P95 forward time: `4.68 ms / 9.02 ms`
- Jacobian `< 0` ratio: `0.0`
- Peak GPU memory during evaluation: `48.72 MB`

Local artifacts for this run live under:

- `reports/benchmarks/acdc-formal-20260417/`
- `checkpoints/acdc-formal-20260417/best_teds_net.pth`
