#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-tedsnet_py39_acdc}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DATA_DIR="${RAW_DATA_DIR:-${PROJECT_ROOT}/Resources}"
PROCESSED_DATA_DIR="${PROCESSED_DATA_DIR:-${PROJECT_ROOT}/results/preprocessed/acdc_ring_144x208}"
RUN_NAME="${RUN_NAME:-acdc_batch200}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
BATCH_SIZE="${BATCH_SIZE:-200}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-200}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VALIDATION_BATCHES="${MAX_VALIDATION_BATCHES:-0}"
MAX_TEST_BATCHES="${MAX_TEST_BATCHES:-0}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"

if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile ]; then
        source /etc/profile
    fi
fi

module load anaconda/3.10

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found after loading anaconda/3.10." >&2
    exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" python=3.9.0
fi

# shellcheck disable=SC1091
source activate "${ENV_NAME}"

python -m pip install --upgrade pip

if ! python -c "import torch, torchvision; assert torch.version.cuda is not None" >/dev/null 2>&1; then
    python -m pip install --index-url "${TORCH_INDEX_URL}" torch==2.4.1 torchvision==0.19.1
fi

python -m pip install -r "${PROJECT_ROOT}/requirements-server-py39.txt"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; cannot verify GPU availability." >&2
    exit 1
fi

nvidia-smi

python -c "import torch; assert torch.cuda.is_available(), 'GPU unavailable; stop as required.'; print('torch version:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu count:', torch.cuda.device_count()); print('current device:', torch.cuda.get_device_name(0))"

PREPROCESS_ARGS=()
if [ "${FORCE_PREPROCESS}" = "1" ]; then
    PREPROCESS_ARGS+=("--force")
fi

python "${PROJECT_ROOT}/scripts/preprocess_acdc.py" \
    --raw-data-path "${RAW_DATA_DIR}" \
    --processed-data-path "${PROCESSED_DATA_DIR}" \
    "${PREPROCESS_ARGS[@]}"

TRAIN_ARGS=(
    --dataset ACDC
    --raw-data-path "${RAW_DATA_DIR}"
    --processed-data-path "${PROCESSED_DATA_DIR}"
    --run-name "${RUN_NAME}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
)

if [ "${MAX_TRAIN_BATCHES}" != "0" ]; then
    TRAIN_ARGS+=(--max-train-batches "${MAX_TRAIN_BATCHES}")
fi
if [ "${MAX_VALIDATION_BATCHES}" != "0" ]; then
    TRAIN_ARGS+=(--max-validation-batches "${MAX_VALIDATION_BATCHES}")
fi
if [ "${MAX_TEST_BATCHES}" != "0" ]; then
    TRAIN_ARGS+=(--max-test-batches "${MAX_TEST_BATCHES}")
fi

python "${PROJECT_ROOT}/scripts/train_runner.py" "${TRAIN_ARGS[@]}"
