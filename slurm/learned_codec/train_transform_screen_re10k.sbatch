#!/usr/bin/env bash
#SBATCH -J gs-lt-screen
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 4-00:00:00
#SBATCH --array=0-2
#SBATCH -o logs/slurm/slurm-gs-lt-screen-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-lt-screen-%A_%a.err

set -euo pipefail
: "${SCREEN_MULTIPLIERS:?Set SCREEN_MULTIPLIERS to the three validation-selected values, e.g. '0.5 1 2'}"
read -r -a MULTIPLIERS <<< "${SCREEN_MULTIPLIERS}"
[[ "${#MULTIPLIERS[@]}" -eq 3 ]] || { echo "SCREEN_MULTIPLIERS must contain exactly 3 values"; exit 2; }

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"; ROOT_OUT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/learned_transform}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/globalsplat-re10k-32k.ckpt}"
CDF_PROFILE="${CDF_PROFILE:-${REPO_DIR}/outputs/grouped_cdf/ctx12_g9_t7/grouped_cdf_profiles.npz}"
LAMBDA_CONFIG="${LAMBDA_CONFIG:-${ROOT_OUT}/smoke_lambda_1/initial_scale.json}"
MULTIPLIER="${MULTIPLIERS[$SLURM_ARRAY_TASK_ID]}"; LABEL="${MULTIPLIER//./p}"
RESUME="${ROOT_OUT}/pilot/lambda_${LABEL}/best.pt"; OUT="${ROOT_OUT}/screen/lambda_${LABEL}"

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1
mkdir -p "${REPO_DIR}/logs/slurm" "${OUT}"; cd "${REPO_DIR}"
srun python scripts/learned_codec/train_transform.py \
  --checkpoint "${CHECKPOINT}" --dataset-root "${DATASET_ROOT}" --cdf-profile "${CDF_PROFILE}" \
  --output-dir "${OUT}" --lambda-config "${LAMBDA_CONFIG}" --lambda-multiplier "${MULTIPLIER}" \
  --resume "${RESUME}" --max-steps 10000 --schedule-steps 40000 --warmup-steps 1000 \
  --validate-every 2000 --save-every 2000 --accumulate-grad-batches 4
