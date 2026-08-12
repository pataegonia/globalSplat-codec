#!/usr/bin/env bash
#SBATCH -J globalsplat-re10k-16k
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm/slurm-globalsplat-re10k-16k-%A.out
#SBATCH -e logs/slurm/slurm-globalsplat-re10k-16k-%A.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/globalSplat" ]]; then
    REPO_DIR="${SUBMIT_DIR}/globalSplat"
else
    REPO_DIR="${SUBMIT_DIR}"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-globalsplat}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-globalsplat_re10k_16k}"
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
echo "Node:       ${SLURMD_NODENAME:-unknown}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES:-unknown}"
nvidia-smi

# 8 GPUs x batch 2 = global batch 16 (the released training recipe).
# auto_resume remains enabled, so resubmitting continues the latest last.ckpt.
srun python -m globalsplat.main \
    +experiment=re10k_16k \
    "dataset.dataset_roots=['${DATASET_ROOT}']" \
    "experiment_name=${EXPERIMENT_NAME}" \
    trainer.devices=8 \
    trainer.num_nodes=1 \
    trainer.strategy=ddp \
    trainer.precision=bf16-mixed \
    optimizer.batch_size=2 \
    "optimizer.num_workers=${NUM_WORKERS}"
