#!/usr/bin/env bash
#SBATCH -J gs-lt-cdf
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 3-00:00:00
#SBATCH --array=0-1
#SBATCH -o logs/slurm/slurm-gs-lt-cdf-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-lt-cdf-%A_%a.err

set -euo pipefail
: "${FINAL_MULTIPLIERS:?Set FINAL_MULTIPLIERS to the two final validation-selected values}"
read -r -a MULTIPLIERS <<< "${FINAL_MULTIPLIERS}"
[[ "${#MULTIPLIERS[@]}" -eq 2 ]] || { echo "FINAL_MULTIPLIERS must contain exactly 2 values"; exit 2; }

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"; ROOT_OUT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/learned_transform}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/globalsplat-re10k-32k.ckpt}"
MULTIPLIER="${MULTIPLIERS[$SLURM_ARRAY_TASK_ID]}"; LABEL="${MULTIPLIER//./p}"
TRANSFORM="${ROOT_OUT}/full/lambda_${LABEL}/best.pt"
PROFILE_DIR="${ROOT_OUT}/final_profiles/lambda_${LABEL}"

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1
mkdir -p "${REPO_DIR}/logs/slurm" "${PROFILE_DIR}"; cd "${REPO_DIR}"
srun python scripts/learned_codec/calibrate_transform_cdf.py \
  --checkpoint "${CHECKPOINT}" --transform-checkpoint "${TRANSFORM}" \
  --dataset-root "${DATASET_ROOT}" --calibration-scenes 1000 \
  --output-profile "${PROFILE_DIR}/grouped_cdf_profiles.npz" \
  --output-report "${PROFILE_DIR}/calibration_report.json"
