#!/usr/bin/env bash
#SBATCH -J globalsplat-cdf-phase1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -w ariel-v8
#SBATCH -p batch_ugrad
#SBATCH -t 1-00:00:00
#SBATCH -o logs/slurm/slurm-globalsplat-cdf-phase1-%j.out
#SBATCH -e logs/slurm/slurm-globalsplat-cdf-phase1-%j.err

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
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/cdf_phase1}"
CONTEXT="${CONTEXT:-12}"
GEOMETRY_BITS="${GEOMETRY_BITS:-9}"
TEXTURE_BITS="${TEXTURE_BITS:-7}"
CALIBRATION_SCENES="${CALIBRATION_SCENES:-1000}"
EVALUATION_SCENES="${EVALUATION_SCENES:-5601}"
HEADER_SCENES="${HEADER_SCENES:-100}"
EVALUATION_NUM_WORKERS="${EVALUATION_NUM_WORKERS:-4}"
SMOOTHING_ALPHA="${SMOOTHING_ALPHA:-0.5}"

RUN_LABEL="ctx${CONTEXT}_g${GEOMETRY_BITS}_t${TEXTURE_BITS}_cal${CALIBRATION_SCENES}_eval${EVALUATION_SCENES}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_LABEL}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

[[ -f "${DATASET_ROOT}/train/index.json" ]] || {
    echo "ERROR: missing train index: ${DATASET_ROOT}/train/index.json"; exit 1;
}
[[ -f "${DATASET_ROOT}/test/index.json" ]] || {
    echo "ERROR: missing test index: ${DATASET_ROOT}/test/index.json"; exit 1;
}
[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing checkpoint: ${CHECKPOINT}"; exit 1; }
python -c "from compressai.ans import RansEncoder, RansDecoder" || {
    echo "ERROR: CompressAI unavailable in ${CONDA_ENV_NAME}"; exit 1;
}

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"
echo "Phase-1 CDF analysis: ${RUN_LABEL}"
echo "Calibration=train (${CALIBRATION_SCENES}), evaluation=fixed test (${EVALUATION_SCENES})"
echo "Output: ${OUTPUT_DIR}"
nvidia-smi

srun python scripts/analysis/analyze_cdf_phase1.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --experiment "${EXPERIMENT}" \
    --context "${CONTEXT}" \
    --calibration-scenes "${CALIBRATION_SCENES}" \
    --evaluation-scenes "${EVALUATION_SCENES}" \
    --header-scenes "${HEADER_SCENES}" \
    --geometry-bits "${GEOMETRY_BITS}" \
    --texture-bits "${TEXTURE_BITS}" \
    --alpha "${SMOOTHING_ALPHA}" \
    --cdf-precision 16 \
    --calibration-num-workers 0 \
    --evaluation-num-workers "${EVALUATION_NUM_WORKERS}" \
    --device cuda

echo "Completed: ${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -maxdepth 1 -type f \
    \( -name '*.json' -o -name '*.npz' \) -print
