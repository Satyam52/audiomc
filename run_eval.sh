#!/bin/bash
# AudioMC text-output eval orchestration (conda: slm).
#
# Prerequisites:
#   sbatch deploy/qwen3_omni_serve.sh
#   sbatch deploy/judge_gemma4_serve.sh
#
# Usage:
#   ./run_eval.sh
#   ./run_eval.sh --limit 5
#
# Edit MODEL_TARGET / JUDGE below when the serve node IP changes.

set -eo pipefail

# ---- edit these when redeploying ----
MODEL_TARGET="Qwen/Qwen3-Omni-30B-A3B-Instruct+192.168.1.49"   # port defaults to 8091
JUDGE="google/gemma-4-26B-A4B-it+192.168.1.50:8000"
# -------------------------------------

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/eval${PYTHONPATH:+:$PYTHONPATH}"

# conda activate scripts reference unset backup vars; don't use `set -u` around them
__conda_setup="$('/home/speech-nlp-cse/24m0756/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate slm
set -u

LIMIT_ARGS=()
if [[ "${1:-}" == "--limit" ]]; then
  LIMIT_ARGS=(--limit "${2:?}")
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="outputs/${STAMP}"
mkdir -p data "$OUT_DIR" logs/eval

if [[ ! -f data/metadata.jsonl ]]; then
  echo "[1/3] Preparing AudioMC data..."
  python eval/prepare_data.py --out-dir data "${LIMIT_ARGS[@]}" \
    2>&1 | tee "logs/eval/prepare_data_${STAMP}.log"
else
  echo "[1/3] data/metadata.jsonl exists — skip download"
fi

PRED="${OUT_DIR}/predictions.jsonl"
JUDGED="${OUT_DIR}/judged.jsonl"
SUMMARY="${OUT_DIR}/all.json"
EVAL_LOG="logs/eval/run_${STAMP}.log"

echo "MODEL_TARGET=${MODEL_TARGET}"
echo "JUDGE=${JUDGE}"
echo "OUT_DIR=${OUT_DIR}"

{
  echo "[2/3] Inference (text output) -> $PRED"
  python eval/infer_text.py \
    --metadata data/metadata.jsonl \
    --model-target "$MODEL_TARGET" \
    --out "$PRED" \
    "${LIMIT_ARGS[@]}"

  echo "[3/3] Judge -> $JUDGED"
  python eval/judge.py \
    --metadata data/metadata.jsonl \
    --predictions "$PRED" \
    --judge "$JUDGE" \
    --out "$JUDGED" \
    --summary "$SUMMARY" \
    "${LIMIT_ARGS[@]}"

  echo "Done. Summary: $SUMMARY"
  cat "$SUMMARY"
} 2>&1 | tee "$EVAL_LOG"
