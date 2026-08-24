#!/usr/bin/env bash
# Final narrow decoder-aware group-quantization sweep.
#SBATCH -J gs-opt-rd-narrow
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 3-00:00:00
#SBATCH --array=0-3
#SBATCH -o logs/slurm/slurm-gs-opt-rd-narrow-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-opt-rd-narrow-%A_%a.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/globalsplat-re10k-32k.ckpt}"
CDF_PROFILE="${GLOBALSPLAT_CDF_PROFILE:-${REPO_DIR}/outputs/grouped_cdf/ctx12_g9_t7/grouped_cdf_profiles.npz}"
CALIBRATION="${CALIBRATION:-${REPO_DIR}/outputs/optimized_group_quant/calibration_operating_points.json}"
PROFILE="${PROFILE:-${REPO_DIR}/slurm/evaluation/optimized_group_quant_narrow_8_9p5.json}"
OUT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/optimized_group_quant_narrow_8_9p5/heldout}"

echo "0ebcb0fff236a66d1662a0c04665be41e74bc609ee034b1646c8be5377c61d5a  ${CALIBRATION}" | sha256sum --check --strict
echo "ba2508a6cb6b5a5a05346c16a78c42142410c9bcbd3a5c09d5d60dcc494ba3bc  ${PROFILE}" | sha256sum --check --strict

LABELS=(target_8pct target_8p5pct target_9pct target_9p5pct)
IDX="${SLURM_ARRAY_TASK_ID}"
LABEL="${LABELS[$IDX]}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export GLOBALSPLAT_CDF_PROFILE="${CDF_PROFILE}"
cd "${REPO_DIR}"

srun python -m globalsplat.main \
  +experiment=re10k_32k \
  +codec=final_g9_t7_g32_t32 \
  dataset=re10k_eval_ctx12 \
  mode=test \
  "dataset.dataset_roots=['${DATASET_ROOT}']" \
  "checkpointing.load=${CHECKPOINT}" \
  optimizer.batch_size=1 \
  "optimizer.num_workers=${NUM_WORKERS:-8}" \
  trainer.devices=1 \
  test.compute_scores=true \
  test.save_image=false \
  test.save_gt_image=false \
  test.save_input_images=false \
  test.save_video=false \
  "test.output_path=${OUT}/${LABEL}" \
  "model.latent_codec_group_quant_profile_path=${PROFILE}" \
  "model.latent_codec_group_quant_policy=${LABEL}" \
  model.latent_codec_group_quant_ablation=joint

RESULT="${OUT}/${LABEL}/globalsplat_re10k_32k"
test -f "${RESULT}/codec_summary.json"
test -f "${RESULT}/scores_all_avg.json"
