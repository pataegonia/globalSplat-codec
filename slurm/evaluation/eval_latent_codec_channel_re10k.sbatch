#!/usr/bin/env bash
#SBATCH -J globalsplat-channel-codec
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 3-00:00:00
#SBATCH --array=0-4
#SBATCH -o logs/slurm/slurm-globalsplat-channel-codec-%A_%a.out
#SBATCH -e logs/slurm/slurm-globalsplat-channel-codec-%A_%a.err

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/latent_codec_channel_eval}"
CONTEXT="${CONTEXT:-12}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# A: 8/8 max, B: 10/8 max, C: 10/6 max, D: 10/8 p99.9, E: 12/8 max.
GEO_BITS=(8 10 10 10 12)
TEX_BITS=(8 8 6 8 8)
CLIP=(null null null 99.9 null)
LABEL=(g8_t8_max g10_t8_max g10_t6_max g10_t8_p99.9 g12_t8_max)

IDX="${SLURM_ARRAY_TASK_ID}"
GB="${GEO_BITS[$IDX]}"
TB="${TEX_BITS[$IDX]}"
CP="${CLIP[$IDX]}"
RUN_LABEL="${LABEL[$IDX]}"
DATASET_CONFIG="re10k_eval_ctx${CONTEXT}"
OUTPUT_DIR="${OUTPUT_ROOT}/ctx${CONTEXT}_${RUN_LABEL}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

[[ -f "${DATASET_ROOT}/test/index.json" ]] || {
    echo "ERROR: missing dataset index: ${DATASET_ROOT}/test/index.json"; exit 1;
}
[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing checkpoint: ${CHECKPOINT}"; exit 1; }
python -c "from compressai.ans import RansEncoder, RansDecoder" || {
    echo "ERROR: CompressAI unavailable in ${CONDA_ENV_NAME}"; exit 1;
}

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"
echo "Codec experiment=${RUN_LABEL}, geo=${GB}, tex=${TB}, clip=${CP}, context=${CONTEXT}"
nvidia-smi

srun python -m globalsplat.main \
    "+experiment=${EXPERIMENT}" \
    "dataset=${DATASET_CONFIG}" \
    mode=test \
    "dataset.dataset_roots=['${DATASET_ROOT}']" \
    "checkpointing.load=${CHECKPOINT}" \
    ++model.latent_codec_bits=0 \
    "++model.latent_codec_geometry_bits=${GB}" \
    "++model.latent_codec_texture_bits=${TB}" \
    ++model.latent_codec_backend=compressai \
    ++model.latent_codec_scale_mode=channel \
    "++model.latent_codec_clip_percentile=${CP}" \
    optimizer.batch_size=1 \
    "optimizer.num_workers=${NUM_WORKERS}" \
    trainer.devices=1 \
    test.compute_scores=true \
    test.save_image=false \
    test.save_gt_image=false \
    test.save_input_images=false \
    test.save_video=false \
    "test.output_path=${OUTPUT_DIR}"

echo "Completed: ${OUTPUT_DIR}"
