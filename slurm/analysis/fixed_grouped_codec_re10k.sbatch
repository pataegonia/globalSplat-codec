#!/usr/bin/env bash
#SBATCH -J globalsplat-fixed-cdf
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v8
#SBATCH -t 1-00:00:00
#SBATCH --array=0-3%4
#SBATCH -o logs/slurm/slurm-globalsplat-fixed-cdf-%A_%a.out
#SBATCH -e logs/slurm/slurm-globalsplat-fixed-cdf-%A_%a.err

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/fixed_grouped_codec/ctx12_g9_t7}"
CONTEXT="${CONTEXT:-12}"
SCENE_LIMIT="${SCENE_LIMIT:-5601}"
NUM_WORKERS="${NUM_WORKERS:-4}"

MODES=(adaptive_k1 fixed_k1 fixed_g32_t16 fixed_g32_t32)
MODE="${MODES[${SLURM_ARRAY_TASK_ID}]}"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODE}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

[[ -f "${DATASET_ROOT}/test/index.json" ]] || {
    echo "ERROR: missing test index: ${DATASET_ROOT}/test/index.json"; exit 1;
}
[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing checkpoint: ${CHECKPOINT}"; exit 1; }
[[ -f "${PROFILE_BUNDLE}" ]] || { echo "ERROR: missing profile: ${PROFILE_BUNDLE}"; exit 1; }
python -c "from compressai.ans import RansEncoder, RansDecoder" || {
    echo "ERROR: CompressAI unavailable in ${CONDA_ENV_NAME}"; exit 1;
}

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"
echo "Actual fixed-CDF codec: ${MODE}, G9/T7, ${SCENE_LIMIT} test scenes"
echo "Node constraint: ariel-v10"
echo "Output: ${OUTPUT_DIR}"
nvidia-smi

srun python scripts/analysis/eval_fixed_grouped_codec.py \
    --mode "${MODE}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset-root "${DATASET_ROOT}" \
    --profile-bundle "${PROFILE_BUNDLE}" \
    --output-dir "${OUTPUT_DIR}" \
    --experiment "${EXPERIMENT}" \
    --context "${CONTEXT}" \
    --scene-limit "${SCENE_LIMIT}" \
    --texture-bits 7 \
    --geometry-bits 9 \
    --alpha 0.5 \
    --cdf-precision 16 \
    --num-workers "${NUM_WORKERS}" \
    --device cuda

echo "Completed: ${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -maxdepth 1 -type f \
    \( -name '*.json' -o -name '*.jsonl' \) -print
