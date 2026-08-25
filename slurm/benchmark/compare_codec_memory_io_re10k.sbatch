#!/usr/bin/env bash
#SBATCH -J gs-codec-io-ab
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 1-00:00:00
#SBATCH -o logs/slurm/slurm-gs-codec-io-ab-%j.out
#SBATCH -e logs/slurm/slurm-gs-codec-io-ab-%j.err

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
PROFILE_BUNDLE="${PROFILE_BUNDLE:-${REPO_DIR}/outputs/grouped_cdf/ctx12_g9_t7/grouped_cdf_profiles.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/codec_memory_io_ab/ctx12_g9_t7_g32_t32}"
CONTEXT="${CONTEXT:-12}"
SCENE_LIMIT="${SCENE_LIMIT:-5601}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

[[ -f "${DATASET_ROOT}/test/index.json" ]] || { echo "ERROR: missing dataset"; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing checkpoint: ${CHECKPOINT}"; exit 1; }
[[ -f "${PROFILE_BUNDLE}" ]] || { echo "ERROR: missing profile: ${PROFILE_BUNDLE}"; exit 1; }
python -c "from compressai.ans import RansEncoder, RansDecoder" || exit 1

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"
nvidia-smi

srun python scripts/benchmark/compare_codec_memory_io.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset-root "${DATASET_ROOT}" \
    --profile-bundle "${PROFILE_BUNDLE}" \
    --output-dir "${OUTPUT_DIR}" \
    --experiment "${EXPERIMENT}" \
    --context "${CONTEXT}" \
    --scene-limit "${SCENE_LIMIT}" \
    --num-workers "${NUM_WORKERS}" \
    --progress-every "${PROGRESS_EVERY}"

echo "Result: ${OUTPUT_DIR}/summary.json"
