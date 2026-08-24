#!/usr/bin/env bash
#SBATCH -J gs-group-rd-cal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 3-00:00:00
#SBATCH -o logs/slurm/slurm-gs-group-rd-cal-%j.out
#SBATCH -e logs/slurm/slurm-gs-group-rd-cal-%j.err
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"; DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/globalsplat-re10k-32k.ckpt}"
CDF_PROFILE="${GLOBALSPLAT_CDF_PROFILE:-${REPO_DIR}/outputs/grouped_cdf/ctx12_g9_t7/grouped_cdf_profiles.npz}"
OUT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/optimized_group_quant}"; SCENES="${CALIBRATION_SCENES:-50}"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1; cd "${REPO_DIR}"; mkdir -p "${OUT}"
srun python scripts/analysis/measure_group_quant_operating_points.py --checkpoint "${CHECKPOINT}" --dataset-root "${DATASET_ROOT}" \
  --cdf-profile "${CDF_PROFILE}" --calibration-scenes "${SCENES}" --context 12 --num-workers 0 --device cuda \
  --multipliers 1 1.25 1.5 1.75 2 2.5 3 --output "${OUT}/calibration_operating_points.json"
python scripts/analysis/optimize_group_quant_allocation.py --operating-points "${OUT}/calibration_operating_points.json" \
  --targets 5 7.5 10 12.5 15 --output-profile "${OUT}/optimized_profile.json"
