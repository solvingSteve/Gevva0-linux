#!/usr/bin/env bash
# ==============================================================================
# Gevva0 Turn-Key Linux Launcher
# Supports:
#   1. Minimum: Tier 1 + Tier 2 (LLM Only)
#      Usage: ./run_linux.sh
#   2. Full Cascade: Tier -1 + Tier 0 + Tier 1 + Tier 2 (LLM + Embedding)
#      Usage: ./run_linux.sh --full-cascade
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
THRESHOLD="${THRESHOLD:-0.45}"
MODEL="${GEVVA0_MODEL:-models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
MMPROJ="${GEVVA0_MMPROJ:-models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL/mmproj-BF16.gguf}"
TIER0_MODEL="${GEVVA0_TIER0_MODEL:-models/embeddings/embeddinggemma-2-onnx}"

FULL_CASCADE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full-cascade|--tier0)
      FULL_CASCADE=1
      shift
      ;;
    --tier1-tier2|--no-tier0)
      FULL_CASCADE=0
      shift
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    -*)
      echo "[gevva0] Unknown option: $1"
      exit 1
      ;;
    *)
      MODEL="$1"
      shift
      ;;
  esac
done

# Validate LLM model path
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

TIER0_ARG=()
if [ "${FULL_CASCADE}" -eq 1 ] || [ "${GEVVA0_FULL_CASCADE:-0}" -eq 1 ]; then
  echo "========================================================================"
  echo " MODE: FULL 4-TIER CASCADE (Tier -1 + Tier 0 + Tier 1 + Tier 2)"
  echo " Tier -1: Static Routing Rules (<50 µs)"
  echo " Tier 0:  Absorbing Markov Chain Early Exits (~16-25 ms)"
  echo " Tier 1:  Gemma 4 Fast-Path Logit Readout (37 ms)"
  echo " Tier 2:  Bounded Hot-KV Chain-of-Thought (<= 256 tokens)"
  echo "========================================================================"
  TIER0_ARG=(--tier0)
  if [ -d "${TIER0_MODEL}" ] || [ -f "${TIER0_MODEL}" ]; then
    TIER0_ARG+=(--tier0-model "${TIER0_MODEL}")
  fi
else
  echo "========================================================================"
  echo " MODE: TIER 1 + TIER 2 CASCADE (Minimum LLM Evaluation)"
  echo " Tier -1: Static Routing Rules (<50 µs)"
  echo " Tier 1:  Gemma 4 Fast-Path Logit Readout (37 ms)"
  echo " Tier 2:  Bounded Hot-KV Chain-of-Thought (<= 256 tokens)"
  echo "========================================================================"
fi

echo " Model:       ${MODEL}"
echo " Host/Port:   http://${HOST}:${PORT}"
echo " CoT Thresh:  ${THRESHOLD}"
echo " Debiasing:   Single-pass marginalization (cyclic_debias=false)"
echo "========================================================================"

exec python -m gevva0.cli serve \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL}" \
  --threshold "${THRESHOLD}" \
  "${MMPROJ_ARG[@]}" \
  "${TIER0_ARG[@]}"
