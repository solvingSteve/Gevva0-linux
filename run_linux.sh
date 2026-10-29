#!/usr/bin/env bash
# ==============================================================================
# Gevva0 Turn-Key Linux Launcher (Tier 1 Fast-Path + Tier 2 Bounded CoT)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
THRESHOLD="${THRESHOLD:-0.45}"
MODEL="${1:-${GEVVA0_MODEL:-models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}}"
MMPROJ="${GEVVA0_MMPROJ:-models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL/mmproj-BF16.gguf}"

# Validate model path
if [ ! -f "${MODEL}" ]; then
  echo "[gevva0] ERROR: Model file not found at: ${MODEL}"
  echo "[gevva0] Please link or copy model weights to models/, or pass the path:"
  echo "         ./run_linux.sh /path/to/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
  exit 1
fi

MMPROJ_ARG=()
if [ -f "${MMPROJ}" ]; then
  echo "[gevva0] Multimodal projector detected: ${MMPROJ}"
  MMPROJ_ARG=(--mmproj "${MMPROJ}")
else
  echo "[gevva0] Multimodal projector not found (running text-only mode)."
fi

echo "========================================================================"
echo " Starting Gevva0 Calibrated Decision Gateway (Linux CUDA)"
echo " Model:     ${MODEL}"
echo " Host/Port: http://${HOST}:${PORT}"
echo " CoT Thresh: ${THRESHOLD} (Tier 2 verification on complex/low-confidence)"
echo " Rules:     Tier -1 enabled (0 DIRECT_EXIT on benchmarks)"
echo " Logits:    Single-pass marginalization (cyclic_debias=false)"
echo "========================================================================"

exec python -m gevva0.cli serve \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL}" \
  --threshold "${THRESHOLD}" \
  "${MMPROJ_ARG[@]}"
