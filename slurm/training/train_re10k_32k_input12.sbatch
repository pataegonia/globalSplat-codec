#!/usr/bin/env bash
#SBATCH -J globalsplat-re10k-32k-in12
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v12
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-globalsplat-re10k-32k-in12-%A.out
#SBATCH -e logs/slurm/slurm-globalsplat-re10k-32k-in12-%A.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/globalSplat" ]]; then
    REPO_DIR="${SUBMIT_DIR}/globalSplat"
else
    REPO_DIR="${SUBMIT_DIR}"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-globalsplat}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-globalsplat_re10k_32k_input12}"
NUM_WORKERS="${NUM_WORKERS:-8}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

[[ -d "${REPO_DIR}" ]] || { echo "ERROR: missing repository: ${REPO_DIR}"; exit 1; }
[[ -f "${DATASET_ROOT}/train/index.json" ]] || {
    echo "ERROR: missing dataset index: ${DATASET_ROOT}/train/index.json"
    exit 1
}

cd "${REPO_DIR}"

echo "Repository: ${REPO_DIR}"
echo "Dataset:    ${DATASET_ROOT}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Recipe:     32K Gaussians, 12 input views per subset forward"
echo "Node:       ${SLURMD_NODENAME:-unknown}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES:-unknown}"
nvidia-smi

# 4096 latent tokens x 8 splats/token = 32,768 Gaussians.
# Subset consistency shares two anchors and splits the other context views:
# 2 + (22 - 2) / 2 = 12 input views per model forward.
# 8 GPUs x batch 2 = global batch 16.
srun python -m globalsplat.main \
    +experiment=re10k_32k \
    "dataset.dataset_roots=['${DATASET_ROOT}']" \
    dataset.num_context_views=22 \
    "experiment_name=${EXPERIMENT_NAME}" \
    trainer.devices=8 \
    trainer.num_nodes=1 \
    trainer.strategy=ddp \
    trainer.precision=bf16-mixed \
    optimizer.batch_size=2 \
    "optimizer.num_workers=${NUM_WORKERS}"
