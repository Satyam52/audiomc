#!/bin/bash
#SBATCH --job-name=judge_gemma4
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --output=/home/speech-nlp-cse/24m0756/abhishek/audiomc/logs/judge/gemma4_%j.log

# AudioMC judge: google/gemma-4-26B-A4B-it on gemma sandbox
# (clone of gemma4_audio_sandbox under audiomc/).

AUDIOMC_DIR="/home/speech-nlp-cse/24m0756/abhishek/audiomc"
mkdir -p "${AUDIOMC_DIR}/logs/judge"
cd "$AUDIOMC_DIR"

__conda_setup="$('/home/speech-nlp-cse/24m0756/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate gemma
export PATH="/home/speech-nlp-cse/24m0756/anaconda3/envs/gemma/bin:$PATH"

IFILE="${IFILE:-${AUDIOMC_DIR}/gemma4_sandbox}"
if [ ! -e "$IFILE" ]; then
  IFILE="${AUDIOMC_DIR}/containers/gemma4_sandbox"
fi
if [ ! -e "$IFILE" ]; then
  echo "ERROR: sandbox not found: ${AUDIOMC_DIR}/gemma4_sandbox"
  exit 1
fi

MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
PORT="${PORT:-8000}"
TP="${TP:-2}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

NODE_IP=$(hostname -i | awk '{print $1}')
NODE_NAME=$(hostname -s)

echo "=============================================="
echo "  AudioMC Judge: ${MODEL}"
echo "=============================================="
echo "  Image: ${IFILE}"
echo "  TP:    ${TP}"
echo "  Job:   ${SLURM_JOB_ID:-local}"
echo "  Node:  ${NODE_NAME}"
echo "  IP:    ${NODE_IP}"
echo "  Port:  ${PORT}"
echo "=============================================="
echo ""
echo ">>> Access: http://${NODE_IP}:${PORT}/v1"
echo ">>> JUDGE=${MODEL}+${NODE_IP}:${PORT}"
echo ""

export USER=user
export LOGNAME=user

JOB_TMP="$HOME/.cache/container_tmp/${SLURM_JOB_ID:-$$}"
mkdir -p "$JOB_TMP" "$JOB_TMP/torchinductor_user"

echo "[$(date)] Starting judge server..."
apptainer exec --cleanenv --nv \
    --contain \
    --no-home \
    --home /tmp \
    --workdir /tmp \
    -B /dev/shm \
    -B "$HF_HOME:$HF_HOME" \
    -B "$HOME:$HOME" \
    -B "$JOB_TMP:/tmp" \
    --env HF_HOME="$HF_HOME" \
    --env HF_HUB_OFFLINE_MODE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env TMPDIR="/tmp" \
    --env VLLM_DO_NOT_TRACK=1 \
    --env USER=user \
    "$IFILE" \
    vllm serve "$MODEL" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --dtype bfloat16 \
        --trust-remote-code \
        --tensor-parallel-size "$TP" \
        --gpu-memory-utilization 0.90 \
        --max-model-len 8192

echo "Job finished at $(date)"
