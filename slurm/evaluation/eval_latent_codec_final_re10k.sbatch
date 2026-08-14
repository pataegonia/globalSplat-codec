#!/usr/bin/env bash
#SBATCH -J globalsplat-final-codec
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 3-00:00:00
#SBATCH -o logs/slurm/slurm-globalsplat-final-codec-%j.out
#SBATCH -e logs/slurm/slurm-globalsplat-final-codec-%j.err

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
GLOBALSPLAT_CDF_PROFILE="${GLOBALSPLAT_CDF_PROFILE:-${REPO_DIR}/outputs/grouped_cdf/ctx12_g9_t7/grouped_cdf_profiles.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/latent_codec_final/ctx12_g9_t7_g32_t32}"
CONTEXT="${CONTEXT:-12}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATASET_CONFIG="re10k_eval_ctx${CONTEXT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export GLOBALSPLAT_CDF_PROFILE

[[ -f "${DATASET_ROOT}/test/index.json" ]] || {
    echo "ERROR: missing test index: ${DATASET_ROOT}/test/index.json"; exit 1;
}
[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing checkpoint: ${CHECKPOINT}"; exit 1; }
[[ -f "${GLOBALSPLAT_CDF_PROFILE}" ]] || {
    echo "ERROR: missing fixed CDF profile: ${GLOBALSPLAT_CDF_PROFILE}"; exit 1;
}
python -c "from compressai.ans import RansEncoder, RansDecoder" || {
    echo "ERROR: CompressAI unavailable in ${CONDA_ENV_NAME}"; exit 1;
}

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"
echo "Final codec: G9/T7, channel FP32 scale, fixed G32/T32 CDF16, rANS"
echo "Shared profile: ${GLOBALSPLAT_CDF_PROFILE}"
echo "Node constraint: ariel-v10"
echo "Output: ${OUTPUT_DIR}"
nvidia-smi

srun python -m globalsplat.main \
    "+experiment=${EXPERIMENT}" \
    "+codec=final_g9_t7_g32_t32" \
    "dataset=${DATASET_CONFIG}" \
    mode=test \
    "dataset.dataset_roots=['${DATASET_ROOT}']" \
    "checkpointing.load=${CHECKPOINT}" \
    optimizer.batch_size=1 \
    "optimizer.num_workers=${NUM_WORKERS}" \
    trainer.devices=1 \
    test.compute_scores=true \
    test.save_image=false \
    test.save_gt_image=false \
    test.save_input_images=false \
    test.save_video=false \
    test.profile_modules=false \
    "test.output_path=${OUTPUT_DIR}"

echo "Completed: ${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -type f \
    \( -name 'scores_all_avg.json' -o -name 'codec_summary.json' \) -print
