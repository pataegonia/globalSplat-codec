#!/usr/bin/env bash
#SBATCH -J gs-lt-texture-only
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v11
#SBATCH -t 12:00:00
#SBATCH -o logs/slurm/slurm-gs-lt-texture-only-%j.out
#SBATCH -e logs/slurm/slurm-gs-lt-texture-only-%j.err

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/globalsplat-re10k-32k.ckpt}"
CDF_PROFILE="${CDF_PROFILE:-${REPO_DIR}/outputs/grouped_cdf/ctx12_g9_t7/grouped_cdf_profiles.npz}"
LAMBDA_CONFIG="${LAMBDA_CONFIG:-${REPO_DIR}/outputs/learned_transform/smoke_stage_a_v2_lambda_1/initial_scale.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/learned_transform}"
PHASE1_OUT="${OUTPUT_ROOT}/texture_only_stage_a_v1_lambda_1_lr3e5_step0_300"
PHASE2_OUT="${OUTPUT_ROOT}/texture_only_stage_a_v1_lambda_1_lr9e6_step300_900"
RESUME_CHECKPOINT="${PHASE1_OUT}/step_000300.pt"
REFERENCE_CHECKPOINT="${PHASE1_OUT}/step_000000.pt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1
mkdir -p "${REPO_DIR}/logs/slurm" "${PHASE1_OUT}" "${PHASE2_OUT}"
cd "${REPO_DIR}"

python -c 'import json, math, sys; value = float(json.load(open(sys.argv[1], encoding="utf-8"))["lambda0"]); expected = float(sys.argv[2]); assert math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15), (value, expected); print("texture-only lambda: {:.17g}".format(value))' \
  "${LAMBDA_CONFIG}" 0.047844308833249735

srun python scripts/learned_codec/train_transform.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATASET_ROOT}" \
  --cdf-profile "${CDF_PROFILE}" \
  --output-dir "${PHASE1_OUT}" \
  --texture-only \
  --max-steps 300 \
  --schedule-steps 40000 \
  --warmup-steps 1000 \
  --validate-every 100 \
  --save-every 100 \
  --accumulate-grad-batches 4 \
  --learning-rate 3e-5 \
  --minimum-rate-correlation 0.2 \
  --maximum-malformed-fraction 0.05 \
  --maximum-smoke-psnr-drop 0.05 \
  --lambda-config "${LAMBDA_CONFIG}" \
  --lambda-multiplier 1.0

python -c 'import math, sys, torch; resume = torch.load(sys.argv[1], map_location="cpu", weights_only=False); reference = torch.load(sys.argv[2], map_location="cpu", weights_only=False); expected = float(sys.argv[3]); assert int(resume["step"]) == 300, resume["step"]; assert int(reference["step"]) == 0, reference["step"]; assert bool(resume["args"]["texture_only"]); assert bool(reference["args"]["texture_only"]); assert math.isclose(float(resume["lambda"]), expected, rel_tol=1e-12, abs_tol=1e-15), resume["lambda"]; print("texture-only continuation verified at step {}".format(resume["step"]))' \
  "${RESUME_CHECKPOINT}" "${REFERENCE_CHECKPOINT}" 0.047844308833249735

srun python scripts/learned_codec/train_transform.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset-root "${DATASET_ROOT}" \
  --cdf-profile "${CDF_PROFILE}" \
  --output-dir "${PHASE2_OUT}" \
  --texture-only \
  --resume "${RESUME_CHECKPOINT}" \
  --selection-reference-checkpoint "${REFERENCE_CHECKPOINT}" \
  --max-steps 900 \
  --schedule-steps 40000 \
  --warmup-steps 1000 \
  --validate-every 100 \
  --save-every 100 \
  --accumulate-grad-batches 4 \
  --learning-rate 9e-6 \
  --fixed-resume-learning-rate 9e-6 \
  --minimum-rate-correlation 0.2 \
  --maximum-malformed-fraction 0.05 \
  --maximum-smoke-psnr-drop 0.05 \
  --lambda-config "${LAMBDA_CONFIG}" \
  --lambda-multiplier 1.0
