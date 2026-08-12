#!/usr/bin/env bash
#SBATCH -J globalsplat-latent-corr
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm/slurm-globalsplat-latent-corr-%A.out
#SBATCH -e logs/slurm/slurm-globalsplat-latent-corr-%A.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/globalSplat" ]]; then
    REPO_DIR="${SUBMIT_DIR}/globalSplat"
else
    REPO_DIR="${SUBMIT_DIR}"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-globalsplat}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
EXPERIMENT="${EXPERIMENT:-re10k_32k}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/globalsplat-re10k-32k.ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/latent_correlation}"
NUM_BATCHES="${NUM_BATCHES:-200}"
NUM_WORKERS="${NUM_WORKERS:-4}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

[[ -d "${REPO_DIR}" ]] || { echo "ERROR: missing repository: ${REPO_DIR}"; exit 1; }
[[ -f "${DATASET_ROOT}/test/index.json" ]] || {
    echo "ERROR: missing dataset index: ${DATASET_ROOT}/test/index.json"; exit 1;
}
[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing checkpoint: ${CHECKPOINT}"; exit 1; }

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_DIR}"

echo "Repository: ${REPO_DIR}"
echo "Dataset:    ${DATASET_ROOT}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Experiment: ${EXPERIMENT}"
echo "Batches:    ${NUM_BATCHES}"
nvidia-smi

for CONTEXT in 12 24 36; do
    echo "===== latent correlation: context=${CONTEXT} ====="
    srun python scripts/analysis/analyze_scene_latent_correlation.py \
        --checkpoint "${CHECKPOINT}" \
        --dataset-root "${DATASET_ROOT}" \
        --experiment "${EXPERIMENT}" \
        --context "${CONTEXT}" \
        --batches "${NUM_BATCHES}" \
        --num-workers "${NUM_WORKERS}" \
        --device cuda \
        --output "${OUTPUT_ROOT}/re10k_ctx${CONTEXT}_n${NUM_BATCHES}.json" \
        --raw-output "${OUTPUT_ROOT}/re10k_ctx${CONTEXT}_n${NUM_BATCHES}_raw.pt"

    srun python scripts/analysis/plot_latent_correlation.py \
        --input "${OUTPUT_ROOT}/re10k_ctx${CONTEXT}_n${NUM_BATCHES}_raw.pt" \
        --output "${OUTPUT_ROOT}/re10k_ctx${CONTEXT}_n${NUM_BATCHES}_hist.png" \
        --title "GlobalSplat ${EXPERIMENT}: RE10K context=${CONTEXT}, n=${NUM_BATCHES}"
done

echo "Correlation reports written to ${OUTPUT_ROOT}"
find "${OUTPUT_ROOT}" -type f \
    \( -name '*.json' -o -name '*_raw.pt' -o -name '*_hist.png' -o -name '*_hist.pdf' \) -print
