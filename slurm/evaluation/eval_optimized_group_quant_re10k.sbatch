#!/usr/bin/env bash
#SBATCH -J gs-opt-group-rd
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 3-00:00:00
#SBATCH --array=0-7
#SBATCH -o logs/slurm/slurm-gs-opt-group-rd-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-opt-group-rd-%A_%a.err
set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"; DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/globalsplat-re10k-32k.ckpt}"
CDF_PROFILE="${GLOBALSPLAT_CDF_PROFILE:-${REPO_DIR}/outputs/grouped_cdf/ctx12_g9_t7/grouped_cdf_profiles.npz}"
MANUAL_PROFILE="${MANUAL_PROFILE:-${REPO_DIR}/config/codec/group_variable_sensitivity_v1.json}"
OPTIMIZED_PROFILE="${OPTIMIZED_PROFILE:-${REPO_DIR}/outputs/optimized_group_quant/optimized_profile.json}"
OUT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/optimized_group_quant/heldout}"
LABELS=(baseline conservative_joint moderate_joint target_5pct target_7p5pct target_10pct target_12p5pct target_15pct)
POLICIES=(baseline conservative moderate target_5pct target_7p5pct target_10pct target_12p5pct target_15pct)
IDX="${SLURM_ARRAY_TASK_ID}"; LABEL="${LABELS[$IDX]}"; POLICY="${POLICIES[$IDX]}"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export GLOBALSPLAT_CDF_PROFILE="${CDF_PROFILE}"; cd "${REPO_DIR}"
PROFILE="${OPTIMIZED_PROFILE}"; [[ "${IDX}" -le 2 ]] && PROFILE="${MANUAL_PROFILE}"
ARGS=(+experiment=re10k_32k +codec=final_g9_t7_g32_t32 dataset=re10k_eval_ctx12 mode=test
 "dataset.dataset_roots=['${DATASET_ROOT}']" "checkpointing.load=${CHECKPOINT}" optimizer.batch_size=1
 "optimizer.num_workers=${NUM_WORKERS:-8}" trainer.devices=1 test.compute_scores=true test.save_image=false
 test.save_gt_image=false test.save_input_images=false test.save_video=false "test.output_path=${OUT}/${LABEL}")
if [[ "${POLICY}" != baseline ]]; then ARGS+=("model.latent_codec_group_quant_profile_path=${PROFILE}"
 "model.latent_codec_group_quant_policy=${POLICY}" model.latent_codec_group_quant_ablation=joint); fi
srun python -m globalsplat.main "${ARGS[@]}"
RESULT="${OUT}/${LABEL}/globalsplat_re10k_32k"; [[ -f "${RESULT}/codec_summary.json" ]] || { echo "missing codec summary"; exit 1; }
