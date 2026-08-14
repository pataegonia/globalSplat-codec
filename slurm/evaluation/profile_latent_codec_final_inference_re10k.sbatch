#!/usr/bin/env bash
#SBATCH -J globalsplat-codec-profile
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm/slurm-globalsplat-codec-profile-%j.out
#SBATCH -e logs/slurm/slurm-globalsplat-codec-profile-%j.err

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
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/latent_codec_final_profile/inference_only_ctx12_g9_t7_g32_t32}"
CONTEXT="${CONTEXT:-12}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PROFILE_SCENES="${PROFILE_SCENES:-200}"
PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-5}"
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
echo "Inference-only final codec profile (NO PSNR/SSIM/LPIPS modules)"
echo "Scenes=${PROFILE_SCENES}, warmup=${PROFILE_WARMUP_STEPS}, node=ariel-v10"
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
    test.compute_scores=false \
    test.save_image=false \
    test.save_gt_image=false \
    test.save_input_images=false \
    test.save_video=false \
    test.profile_modules=true \
    "test.profile_warmup_steps=${PROFILE_WARMUP_STEPS}" \
    "test.max_batches=${PROFILE_SCENES}" \
    "test.output_path=${OUTPUT_DIR}"

echo "Completed inference-only codec profile: ${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -type f \
    \( -name 'codec_summary.json' -o -name 'module_profile_summary.json' \
       -o -name 'module_profile_per_scene.json' -o -name 'peak_memory.json' \) -print
