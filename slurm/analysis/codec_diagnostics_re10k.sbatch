#!/usr/bin/env bash
#SBATCH -J gs-codec-diag
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 3-00:00:00
#SBATCH -o logs/slurm/slurm-gs-codec-diag-%j.out
#SBATCH -e logs/slurm/slurm-gs-codec-diag-%j.err
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"; DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"; CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/globalsplat-re10k-32k.ckpt}"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV_NAME:-globalsplat}"; cd "${REPO_DIR}"; export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
OUT="${OUTPUT_ROOT:-outputs/codec_diagnostics}"; mkdir -p "${OUT}"
COMMON=(--checkpoint "${CHECKPOINT}" --dataset-root "${DATASET_ROOT}" --context "${CONTEXT:-12}" --device cuda --num-workers "${NUM_WORKERS:-4}")
srun python scripts/analysis/analyze_channel_sensitivity.py "${COMMON[@]}" --output "${OUT}/channel_sensitivity.json"
srun python scripts/analysis/analyze_transform_coding.py "${COMMON[@]}" --output "${OUT}/transform_coding.json"
srun python scripts/analysis/analyze_geo_texture_prediction.py "${COMMON[@]}" --output "${OUT}/geo_texture_prediction.json"
