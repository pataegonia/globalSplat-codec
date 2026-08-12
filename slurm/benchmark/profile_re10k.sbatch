#!/usr/bin/env bash
#SBATCH -J globalsplat-profile
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v9
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm/slurm-globalsplat-profile-%A.out
#SBATCH -e logs/slurm/slurm-globalsplat-profile-%A.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-/ceph_data/clue9986/globalSplat}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-globalsplat}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/globalsplat-re10k-32k.ckpt}"
EXPERIMENT="${EXPERIMENT:-re10k_32k}"
NUM_SCENES="${NUM_SCENES:-200}"
WARMUP="${WARMUP:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing checkpoint: ${CHECKPOINT}"; exit 1; }
[[ -f "${DATASET_ROOT}/test/index.json" ]] || {
    echo "ERROR: missing dataset index: ${DATASET_ROOT}/test/index.json"; exit 1;
}

cd "${REPO_DIR}"
mkdir -p outputs/profile
nvidia-smi

for CONTEXT in 12 24 36; do
    echo "===== profiling context=${CONTEXT} ====="
    srun python scripts/benchmark/profile_re10k_dataset.py \
        --checkpoint "${CHECKPOINT}" \
        --dataset-root "${DATASET_ROOT}" \
        --experiment "${EXPERIMENT}" \
        --context "${CONTEXT}" \
        --num-scenes "${NUM_SCENES}" \
        --warmup "${WARMUP}" \
        --num-workers "${NUM_WORKERS}" \
        --output "outputs/profile/re10k_ctx${CONTEXT}_n${NUM_SCENES}.json"
done

echo "Profiles written to ${REPO_DIR}/outputs/profile"
