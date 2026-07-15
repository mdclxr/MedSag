#!/bin/bash
set -euo pipefail

export MEDSAM2_REPO="${MEDSAM2_REPO:-/home/mdc/MedSAM2}"
export MEDSAM2_CONFIG="${MEDSAM2_CONFIG:-$MEDSAM2_REPO/sam2/configs/sam2.1_hiera_t512.yaml}"
export MEDSAM2_DEVICE="${MEDSAM2_DEVICE:-cuda:0}"
export MEDSAM2_ENV_PREFIX="${MEDSAM2_ENV_PREFIX:-/home/mdc/.conda/envs/medsam2}"
export LD_LIBRARY_PATH="$MEDSAM2_ENV_PREFIX/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$MEDSAM2_ENV_PREFIX/lib/python3.12/site-packages/nvidia/cusparse/lib:$MEDSAM2_ENV_PREFIX/lib/python3.12/site-packages/nvidia/cublas/lib:$MEDSAM2_ENV_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$MEDSAM2_ENV_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$MEDSAM2_ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"

if [ -z "${MEDSAM2_CHECKPOINT:-}" ]; then
  if [ -f "$MEDSAM2_REPO/checkpoints/MedSAM2_latest.pt" ]; then
    export MEDSAM2_CHECKPOINT="$MEDSAM2_REPO/checkpoints/MedSAM2_latest.pt"
  elif [ -f "$MEDSAM2_REPO/checkpoints/MedSAM2_2411.pt" ]; then
    export MEDSAM2_CHECKPOINT="$MEDSAM2_REPO/checkpoints/MedSAM2_2411.pt"
  else
    echo "No MedSAM2 checkpoint found in $MEDSAM2_REPO/checkpoints"
    exit 1
  fi
fi

python -m uvicorn medsam2_service_app.app:app --host 127.0.0.1 --port 7001
