#!/usr/bin/env bash
#SBATCH -J globalsplat-codec-qat
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
# Lightning DDP expects one externally launched Slurm task per requested GPU.
# Each rank receives one GPU while Trainer(devices=4) describes the node total.
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v8
#SBATCH -t 3-00:00:00
#SBATCH -o logs/slurm/slurm-globalsplat-codec-qat-%j.out
#SBATCH -e logs/slurm/slurm-globalsplat-codec-qat-%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/globalSplat" ]]; then
    REPO_DIR="${SUBMIT_DIR}/globalSplat"
else
    REPO_DIR="${SUBMIT_DIR}"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-globalsplat}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/globalsplat-re10k-32k.ckpt}"
GLOBALSPLAT_CDF_PROFILE="${GLOBALSPLAT_CDF_PROFILE:-${REPO_DIR}/outputs/grouped_cdf/ctx12_g9_t7/grouped_cdf_profiles.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/checkpoints/codec_qat}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_STEPS="${MAX_STEPS:-10000}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export GLOBALSPLAT_CDF_PROFILE

[[ -f "${DATASET_ROOT}/train/index.json" ]] || {
    echo "ERROR: missing train index: ${DATASET_ROOT}/train/index.json"; exit 1;
}
[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing checkpoint: ${CHECKPOINT}"; exit 1; }
[[ -f "${GLOBALSPLAT_CDF_PROFILE}" ]] || {
    echo "ERROR: missing fixed CDF profile: ${GLOBALSPLAT_CDF_PROFILE}"; exit 1;
}
python -c "from compressai.ans import RansEncoder, RansDecoder" || {
    echo "ERROR: CompressAI unavailable in ${CONDA_ENV_NAME}"; exit 1;
}

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_DIR}"
echo "Decoder-only QAT: post-Mixer G9/T7, fixed G32/T32 profile"
echo "Frozen: tokenizer + scene tokens + dual-stream slot encoder"
echo "Trainable: Gaussian decoder only"
echo "Initial checkpoint: ${CHECKPOINT}"
echo "Shared profile: ${GLOBALSPLAT_CDF_PROFILE}"
echo "Steps=${MAX_STEPS}, LR=${LEARNING_RATE}, GPUs=4, node=ariel-v10"
nvidia-smi

srun python -m globalsplat.main \
    +experiment=re10k_32k_codec_qat_decoder \
    "dataset.dataset_roots=['${DATASET_ROOT}']" \
    "checkpointing.load=${CHECKPOINT}" \
    checkpointing.resume=false \
    checkpointing.auto_resume=false \
    "output_dir=${OUTPUT_ROOT}" \
    "optimizer.lr=${LEARNING_RATE}" \
    "optimizer.num_workers=${NUM_WORKERS}" \
    "trainer.max_steps=${MAX_STEPS}" \
    trainer.devices=4 \
    trainer.num_nodes=1 \
    trainer.strategy=ddp \
    trainer.precision=bf16-mixed

echo "Completed decoder-only codec QAT"
find "${OUTPUT_ROOT}/globalsplat_re10k_32k_codec_qat_decoder" \
    -type f \( -name '*.ckpt' -o -name 'last.ckpt' \) -print
