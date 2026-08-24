#!/usr/bin/env bash
#SBATCH -J gs-codec-decorrelation
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm/slurm-gs-codec-decorrelation-%j.out
#SBATCH -e logs/slurm/slurm-gs-codec-decorrelation-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-globalsplat}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
EXPERIMENT="${EXPERIMENT:-re10k_32k}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/globalsplat-re10k-32k.ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/codec_decorrelation}"
SCENES="${SCENES:-200}"
CONTEXT="${CONTEXT:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

[[ -f "${DATASET_ROOT}/test/index.json" ]] || {
    echo "ERROR: missing dataset index: ${DATASET_ROOT}/test/index.json"; exit 1;
}
[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing checkpoint: ${CHECKPOINT}"; exit 1; }

mkdir -p "${OUTPUT_ROOT}/ctx${CONTEXT}"
cd "${REPO_DIR}"

srun python scripts/analysis/analyze_codec_decorrelation.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset-root "${DATASET_ROOT}" \
    --experiment "${EXPERIMENT}" \
    --context "${CONTEXT}" \
    --scenes "${SCENES}" \
    --texture-bits 7 \
    --geometry-bits 9 \
    --scale-mode channel \
    --lags 1 2 4 8 16 32 64 128 256 \
    --num-workers "${NUM_WORKERS}" \
    --device cuda \
    --seed 0 \
    --output "${OUTPUT_ROOT}/ctx${CONTEXT}"

echo "Results: ${OUTPUT_ROOT}/ctx${CONTEXT}"
