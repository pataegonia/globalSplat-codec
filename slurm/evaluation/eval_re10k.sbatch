#!/usr/bin/env bash
#SBATCH -J globalsplat-re10k-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v9
#SBATCH -t 1-00:00:00
#SBATCH -o logs/slurm/slurm-globalsplat-re10k-eval-%A.out
#SBATCH -e logs/slurm/slurm-globalsplat-re10k-eval-%A.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/globalSplat" ]]; then
    REPO_DIR="${SUBMIT_DIR}/globalSplat"
else
    REPO_DIR="${SUBMIT_DIR}"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-globalsplat}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
MODEL_VARIANT="${MODEL_VARIANT:-re10k_32k}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/globalsplat-re10k-32k.ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/eval/re10k-32k}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_IMAGE="${SAVE_IMAGE:-false}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

[[ -d "${REPO_DIR}" ]] || { echo "ERROR: missing repository: ${REPO_DIR}"; exit 1; }
[[ -f "${DATASET_ROOT}/test/index.json" ]] || {
    echo "ERROR: missing dataset index: ${DATASET_ROOT}/test/index.json"
    exit 1
}
[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing checkpoint: ${CHECKPOINT}"; exit 1; }

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_DIR}"

echo "Repository: ${REPO_DIR}"
echo "Dataset:    ${DATASET_ROOT}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Model:      ${MODEL_VARIANT}"
echo "Output:     ${OUTPUT_ROOT}"
nvidia-smi

for CONTEXT_VIEWS in 12 24 36; do
    DATASET_CONFIG="re10k_eval_ctx${CONTEXT_VIEWS}"
    OUTPUT_DIR="${OUTPUT_ROOT}/ctx${CONTEXT_VIEWS}"

    echo "===== GlobalSplat RE10K: context=${CONTEXT_VIEWS}, targets=3 ====="
    srun python -m globalsplat.main \
        "+experiment=${MODEL_VARIANT}" \
        "dataset=${DATASET_CONFIG}" \
        mode=test \
        "dataset.dataset_roots=['${DATASET_ROOT}']" \
        "checkpointing.load=${CHECKPOINT}" \
        optimizer.batch_size=1 \
        "optimizer.num_workers=${NUM_WORKERS}" \
        trainer.devices=1 \
        test.compute_scores=true \
        "test.save_image=${SAVE_IMAGE}" \
        test.save_gt_image=false \
        test.save_input_images=false \
        test.save_video=false \
        "test.output_path=${OUTPUT_DIR}"

    echo "Completed context=${CONTEXT_VIEWS}: ${OUTPUT_DIR}"
done

echo "All RE10K evaluations completed: ${OUTPUT_ROOT}"
find "${OUTPUT_ROOT}" -type f \
    \( -name 'scores*.json' -o -name 'benchmark.json' -o -name 'peak_memory.json' \) \
    -print
