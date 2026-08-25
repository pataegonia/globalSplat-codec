#!/usr/bin/env bash
#SBATCH -J gs-lt-full
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 6-00:00:00
#SBATCH --array=0-1
#SBATCH -o logs/slurm/slurm-gs-lt-full-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-lt-full-%A_%a.err

set -euo pipefail
: "${FULL_MULTIPLIERS:?Set FULL_MULTIPLIERS to the two validation-selected values}"
read -r -a MULTIPLIERS <<< "${FULL_MULTIPLIERS}"
[[ "${#MULTIPLIERS[@]}" -eq 2 ]] || { echo "FULL_MULTIPLIERS must contain exactly 2 values"; exit 2; }

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"; ROOT_OUT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/learned_transform}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/globalsplat-re10k-32k.ckpt}"
CDF_PROFILE="${CDF_PROFILE:-${REPO_DIR}/outputs/grouped_cdf/ctx12_g9_t7/grouped_cdf_profiles.npz}"
LAMBDA_CONFIG="${LAMBDA_CONFIG:-${ROOT_OUT}/smoke_lambda_1/initial_scale.json}"
MULTIPLIER="${MULTIPLIERS[$SLURM_ARRAY_TASK_ID]}"; LABEL="${MULTIPLIER//./p}"
RESUME="${ROOT_OUT}/screen/lambda_${LABEL}/best.pt"; OUT="${ROOT_OUT}/full/lambda_${LABEL}"

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1
mkdir -p "${REPO_DIR}/logs/slurm" "${OUT}"; cd "${REPO_DIR}"
srun python scripts/learned_codec/train_transform.py \
  --checkpoint "${CHECKPOINT}" --dataset-root "${DATASET_ROOT}" --cdf-profile "${CDF_PROFILE}" \
  --output-dir "${OUT}" --lambda-config "${LAMBDA_CONFIG}" --lambda-multiplier "${MULTIPLIER}" \
  --resume "${RESUME}" --max-steps 40000 --schedule-steps 40000 --warmup-steps 1000 \
  --validate-every 2000 --save-every 2000 --early-stopping-steps 8000 \
  --accumulate-grad-batches 4
