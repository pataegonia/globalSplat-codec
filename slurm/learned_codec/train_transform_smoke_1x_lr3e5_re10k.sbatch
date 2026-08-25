#!/usr/bin/env bash
#SBATCH -J gs-lt-1x-lr3e5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v11
#SBATCH -t 12:00:00
#SBATCH -o logs/slurm/slurm-gs-lt-1x-lr3e5-%j.out
#SBATCH -e logs/slurm/slurm-gs-lt-1x-lr3e5-%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/globalsplat-re10k-32k.ckpt}"
CDF_PROFILE="${CDF_PROFILE:-${REPO_DIR}/outputs/grouped_cdf/ctx12_g9_t7/grouped_cdf_profiles.npz}"
LAMBDA_CONFIG="${LAMBDA_CONFIG:-${REPO_DIR}/outputs/learned_transform/smoke_stage_a_v2_lambda_1/initial_scale.json}"
OUT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/learned_transform}/smoke_stage_a_v4_lambda_1_lr3e5"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1
mkdir -p "${REPO_DIR}/logs/slurm" "${OUT}"
cd "${REPO_DIR}"

python -c 'import json, math, sys; value = float(json.load(open(sys.argv[1], encoding="utf-8"))["lambda0"]); expected = float(sys.argv[2]); print(f"selected lambda: {value:.17g}"); assert math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15), (value, expected)' \
  "${LAMBDA_CONFIG}" 0.047844308833249735

srun python scripts/learned_codec/train_transform.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATASET_ROOT}" \
  --cdf-profile "${CDF_PROFILE}" \
  --output-dir "${OUT}" \
  --max-steps 300 \
  --schedule-steps 40000 \
  --warmup-steps 1000 \
  --validate-every 50 \
  --save-every 100 \
  --accumulate-grad-batches 4 \
  --learning-rate 3e-5 \
  --minimum-rate-correlation 0.2 \
  --maximum-malformed-fraction 0.05 \
  --maximum-smoke-psnr-drop 0.05 \
  --lambda-config "${LAMBDA_CONFIG}" \
  --lambda-multiplier 1.0
