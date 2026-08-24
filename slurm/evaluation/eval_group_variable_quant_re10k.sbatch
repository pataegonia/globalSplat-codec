#!/usr/bin/env bash
#SBATCH -J gs-group-quant-rd
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 3-00:00:00
#SBATCH --array=0-9
#SBATCH -o logs/slurm/slurm-gs-group-quant-rd-%A_%a.out
#SBATCH -e logs/slurm/slurm-gs-group-quant-rd-%A_%a.err
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATASET_ROOT="${DATASET_ROOT:-/data3/local_datasets/re10k}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/globalsplat-re10k-32k.ckpt}"
GLOBALSPLAT_CDF_PROFILE="${GLOBALSPLAT_CDF_PROFILE:-${REPO_DIR}/outputs/grouped_cdf/ctx12_g9_t7/grouped_cdf_profiles.npz}"
GLOBALSPLAT_GROUP_QUANT_PROFILE="${GLOBALSPLAT_GROUP_QUANT_PROFILE:-${REPO_DIR}/config/codec/group_variable_sensitivity_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/group_variable_quant_rd}"

LABELS=(baseline conservative_geo conservative_tex conservative_joint moderate_geo moderate_tex moderate_joint aggressive_geo aggressive_tex aggressive_joint)
POLICIES=(baseline conservative conservative conservative moderate moderate moderate aggressive aggressive aggressive)
ABLATIONS=(joint geometry_only texture_only joint geometry_only texture_only joint geometry_only texture_only joint)
IDX="${SLURM_ARRAY_TASK_ID}"; LABEL="${LABELS[$IDX]}"; POLICY="${POLICIES[$IDX]}"; ABLATION="${ABLATIONS[$IDX]}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-globalsplat}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" GLOBALSPLAT_CDF_PROFILE
[[ -f "${CHECKPOINT}" ]] || { echo "missing checkpoint: ${CHECKPOINT}"; exit 1; }
[[ -f "${DATASET_ROOT}/test/index.json" ]] || { echo "missing RE10K test index"; exit 1; }
[[ -f "${GLOBALSPLAT_CDF_PROFILE}" ]] || { echo "missing CDF profile"; exit 1; }
[[ -f "${GLOBALSPLAT_GROUP_QUANT_PROFILE}" ]] || { echo "missing group quantization profile"; exit 1; }
cd "${REPO_DIR}"

ARGS=(+experiment=re10k_32k +codec=final_g9_t7_g32_t32 dataset=re10k_eval_ctx12 mode=test
  "dataset.dataset_roots=['${DATASET_ROOT}']" "checkpointing.load=${CHECKPOINT}"
  model.latent_codec_bits=0 model.latent_codec_geometry_bits=9
  model.latent_codec_texture_bits=7 model.latent_codec_backend=compressai
  model.latent_codec_scale_mode=channel model.latent_codec_clip_percentile=null
  model.latent_codec_cdf_mode=fixed_grouped
  "model.latent_codec_profile_path=${GLOBALSPLAT_CDF_PROFILE}"
  model.latent_codec_geometry_groups=32 model.latent_codec_texture_groups=32
  model.latent_codec_cdf_precision=16 model.latent_codec_cdf_alpha=0.5
  optimizer.batch_size=1 "optimizer.num_workers=${NUM_WORKERS:-8}" trainer.devices=1
  test.compute_scores=true test.save_image=false test.save_gt_image=false
  test.save_input_images=false test.save_video=false "test.output_path=${OUTPUT_ROOT}/${LABEL}")
if [[ "${POLICY}" != baseline ]]; then
  ARGS+=("model.latent_codec_group_quant_profile_path=${GLOBALSPLAT_GROUP_QUANT_PROFILE}"
    "model.latent_codec_group_quant_policy=${POLICY}"
    "model.latent_codec_group_quant_ablation=${ABLATION}")
fi

# Refuse to spend a multi-day allocation if Hydra did not activate the codec.
CONFIG_DUMP="$(python -m globalsplat.main --cfg job "${ARGS[@]}")"
grep -q 'latent_codec_geometry_bits: 9' <<<"${CONFIG_DUMP}" || { echo "effective config lacks geometry G9"; exit 1; }
grep -q 'latent_codec_texture_bits: 7' <<<"${CONFIG_DUMP}" || { echo "effective config lacks texture T7"; exit 1; }
grep -q 'latent_codec_cdf_mode: fixed_grouped' <<<"${CONFIG_DUMP}" || { echo "effective config lacks fixed grouped CDF"; exit 1; }
echo "Effective codec config verified: G9/T7, fixed G32/T32, policy=${POLICY}, ablation=${ABLATION}"
srun python -m globalsplat.main "${ARGS[@]}"

RESULT_DIR="${OUTPUT_ROOT}/${LABEL}/globalsplat_re10k_32k"
[[ -f "${RESULT_DIR}/codec_summary.json" ]] || {
  echo "ERROR: codec_summary.json was not produced; codec path did not execute"; exit 1;
}
[[ -f "${RESULT_DIR}/scores_all_avg.json" ]] || {
  echo "ERROR: scores_all_avg.json was not produced"; exit 1;
}
