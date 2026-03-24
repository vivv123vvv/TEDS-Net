#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-tedsnet_py39}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/tmp}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile ]; then
        # 某些服务器需要先加载 profile 才能使用 module 命令。
        source /etc/profile
    fi
fi

module load anaconda/3.10

if ! command -v conda >/dev/null 2>&1; then
    echo "未找到 conda，请先确认 anaconda/3.10 模块已正确加载。" >&2
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
    echo "未找到 nvidia-smi，无法按规则确认 GPU 是否可用。" >&2
    exit 1
fi

nvidia-smi

python -c "import torch; assert torch.cuda.is_available(), 'GPU 不可用，按仓库规则停止运行。'; print('torch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('GPU 数量:', torch.cuda.device_count()); print('当前设备:', torch.cuda.get_device_name(0))"

mkdir -p "${DATA_DIR}"

python "${PROJECT_ROOT}/scripts/train_runner.py" \
    --dataset mnist \
    --epochs 1 \
    --batch-size 64 \
    --num-workers 0 \
    --data-path "${DATA_DIR}" \
    --max-train-batches 1 \
    --max-validation-batches 1 \
    --max-test-batches 1 \
    --skip-plot
