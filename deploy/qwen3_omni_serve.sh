#!/bin/bash
#SBATCH --job-name=qwen3_omni
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --output=/home/speech-nlp-cse/24m0756/abhishek/audiomc/logs/serve/qwen3_omni_%j.log

# Serve Qwen3-Omni-30B-A3B-Instruct (thinker / text output) via gemma sandbox vLLM.
# Sandbox is a clone of af3_qwen_vllm_deployment/gemma4_audio_sandbox
# (same image used for Qwen2.5-Omni; includes av/soundfile + Qwen3 thinker).

AUDIOMC_DIR="/home/speech-nlp-cse/24m0756/abhishek/audiomc"
mkdir -p "${AUDIOMC_DIR}/logs/serve"
cd "$AUDIOMC_DIR"

__conda_setup="$('/home/speech-nlp-cse/24m0756/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate gemma
export PATH="/home/speech-nlp-cse/24m0756/anaconda3/envs/gemma/bin:$PATH"

# Clone of gemma4_audio_sandbox under audiomc/
IFILE="${IFILE:-${AUDIOMC_DIR}/gemma4_sandbox}"
if [ ! -e "$IFILE" ]; then
  IFILE="${AUDIOMC_DIR}/containers/gemma4_sandbox"
fi
if [ ! -e "$IFILE" ]; then
  echo "ERROR: sandbox not found: ${AUDIOMC_DIR}/gemma4_sandbox"
  echo "Clone with: rsync -a af3_qwen_vllm_deployment/gemma4_audio_sandbox/ audiomc/gemma4_sandbox/"
  exit 1
fi

MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
PORT="${PORT:-8091}"
TP="${TP:-2}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

NODE_IP=$(hostname -i | awk '{print $1}')
NODE_NAME=$(hostname -s)
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)

echo "=============================================="
echo "  Qwen3-Omni vLLM Server (text / thinker)"
echo "=============================================="
echo "  Model: ${MODEL}"
echo "  Image: ${IFILE}"
echo "  GPUs:  ${NGPU} (TP=${TP})"
echo "  Job:   ${SLURM_JOB_ID:-local}"
echo "  Node:  ${NODE_NAME}"
echo "  IP:    ${NODE_IP}"
echo "  Port:  ${PORT}"
echo "=============================================="
echo ""
echo ">>> Access: http://${NODE_IP}:${PORT}/v1"
echo ">>> MODEL_TARGET=${MODEL}+${NODE_IP}"
echo ""

export USER=user
export LOGNAME=user

JOB_TMP="$HOME/.cache/container_tmp/${SLURM_JOB_ID:-$$}"
mkdir -p "$JOB_TMP" "$JOB_TMP/torchinductor_user"

echo "[$(date)] Starting Qwen3-Omni server..."
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
        --gpu-memory-utilization 0.80 \
        --max-model-len 16384 \
        --limit-mm-per-prompt '{"audio": 8}'

echo "Job finished at $(date)"
