#!/usr/bin/env bash
#SBATCH -J globalsplat-codec-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 6-00:00:00
#SBATCH --array=0-2
#SBATCH -o logs/slurm/slurm-globalsplat-codec-eval-%A_%a.out
#SBATCH -e logs/slurm/slurm-globalsplat-codec-eval-%A_%a.err

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/latent_codec_eval}"
CONTEXT="${CONTEXT:-12}"
NUM_WORKERS="${NUM_WORKERS:-8}"

BITS_LIST=(8 6 4)
BITS="${BITS_LIST[${SLURM_ARRAY_TASK_ID}]}"
DATASET_CONFIG="re10k_eval_ctx${CONTEXT}"
OUTPUT_DIR="${OUTPUT_ROOT}/ctx${CONTEXT}_${BITS}bit"

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
python -c "from compressai.ans import RansEncoder, RansDecoder" || {
    echo "ERROR: CompressAI rANS backend is unavailable. Install it with: pip install compressai"
    exit 1
}

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

echo "Repository: ${REPO_DIR}"
echo "Dataset:    ${DATASET_ROOT}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Experiment: ${EXPERIMENT}"
echo "Context:    ${CONTEXT}"
echo "Codec bits: ${BITS}"
echo "Output:     ${OUTPUT_DIR}"
nvidia-smi

# The deterministic index contains all 5,601 RE10K test scenes. No
# limit_test_batches override is supplied, so the complete index is evaluated.
srun python -m globalsplat.main \
    "+experiment=${EXPERIMENT}" \
    "dataset=${DATASET_CONFIG}" \
    mode=test \
    "dataset.dataset_roots=['${DATASET_ROOT}']" \
    "checkpointing.load=${CHECKPOINT}" \
    "model.latent_codec_bits=${BITS}" \
    model.latent_codec_backend=compressai \
    optimizer.batch_size=1 \
    "optimizer.num_workers=${NUM_WORKERS}" \
    trainer.devices=1 \
    test.compute_scores=true \
    test.save_image=false \
    test.save_gt_image=false \
    test.save_input_images=false \
    test.save_video=false \
    "test.output_path=${OUTPUT_DIR}"

echo "Completed ${BITS}-bit full-test evaluation: ${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -type f \
    \( -name 'scores*.json' -o -name 'codec*.json' -o -name 'benchmark.json' \) -print
