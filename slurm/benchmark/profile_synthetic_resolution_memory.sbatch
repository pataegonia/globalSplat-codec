#!/usr/bin/env bash
#SBATCH -J gs-synth-mem
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 1-00:00:00
#SBATCH -o logs/slurm/slurm-gs-synth-mem-%j.out
#SBATCH -e logs/slurm/slurm-gs-synth-mem-%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/globalSplat" ]]; then
    REPO_DIR="${SUBMIT_DIR}/globalSplat"
else
    REPO_DIR="${SUBMIT_DIR}"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-globalsplat}"
EXPERIMENT="${EXPERIMENT:-re10k_32k}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/checkpoints/globalsplat-re10k-32k.ckpt}"
PROFILE_BUNDLE="${PROFILE_BUNDLE:-${REPO_DIR}/outputs/grouped_cdf/ctx12_g9_t7/grouped_cdf_profiles.npz}"
CODEC_PRESET="${CODEC_PRESET:-final_g9_t7_g32_t32}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/synthetic_resolution_memory}"
SWEEP="${SWEEP:-combined}"
SQUARE_RESOLUTIONS="256 384 512 768 1024 1280 1536 1792 2048"
VIDEO_RESOLUTIONS="720x1280 1080x1920 1080x2048 1440x2560 2160x3840"
COMBINED_RESOLUTIONS="256 384 512 768 720x1280 1024 1280 1080x1920 1080x2048 1536 1792 1440x2560 2048 2160x3840"
if [[ -n "${RESOLUTIONS:-}" ]]; then
    SWEEP_TAG="custom"
else
    SWEEP_TAG="${SWEEP}"
    case "${SWEEP}" in
        square) RESOLUTIONS="${SQUARE_RESOLUTIONS}" ;;
        video) RESOLUTIONS="${VIDEO_RESOLUTIONS}" ;;
        combined) RESOLUTIONS="${COMBINED_RESOLUTIONS}" ;;
        *) echo "ERROR: SWEEP must be square, video, or combined"; exit 1 ;;
    esac
fi
CONTEXT_VIEWS="${CONTEXT_VIEWS:-12}"
TARGET_VIEWS="${TARGET_VIEWS:-3}"
PRECISION="${PRECISION:-fp32}"
WARMUP="${WARMUP:-1}"
RESOLUTION_TIMEOUT="${RESOLUTION_TIMEOUT:-3600}"
SEED="${SEED:-0}"
RUN_LABEL="${RUN_LABEL:-${SWEEP_TAG}_warmup${WARMUP}}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing checkpoint: ${CHECKPOINT}"; exit 1; }
python -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU is required'"
python -c "import gsplat" || { echo "ERROR: gsplat is not installed"; exit 1; }

CODEC_ARGS=()
RUN_TAG="no_codec"
if [[ "${CODEC_PRESET}" != "none" ]]; then
    [[ -f "${PROFILE_BUNDLE}" ]] || {
        echo "ERROR: missing grouped CDF profile: ${PROFILE_BUNDLE}"; exit 1;
    }
    export GLOBALSPLAT_CDF_PROFILE="${PROFILE_BUNDLE}"
    python -c "from compressai.ans import RansEncoder, RansDecoder" || {
        echo "ERROR: CompressAI rANS is unavailable"; exit 1;
    }
    CODEC_ARGS=(--codec "${CODEC_PRESET}")
    RUN_TAG="${CODEC_PRESET}"
fi

read -r -a RESOLUTION_ARGS <<< "${RESOLUTIONS}"
mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"
nvidia-smi

echo "experiment=${EXPERIMENT} codec=${CODEC_PRESET} precision=${PRECISION} warmup=${WARMUP}"
echo "sweep=${SWEEP_TAG} run_label=${RUN_LABEL}"
echo "context=${CONTEXT_VIEWS} target=${TARGET_VIEWS} resolutions=${RESOLUTIONS}"

srun python scripts/benchmark/profile_synthetic_resolution_memory.py \
    --resolutions "${RESOLUTION_ARGS[@]}" \
    --experiment "${EXPERIMENT}" \
    --checkpoint "${CHECKPOINT}" \
    "${CODEC_ARGS[@]}" \
    --context-views "${CONTEXT_VIEWS}" \
    --target-views "${TARGET_VIEWS}" \
    --precision "${PRECISION}" \
    --warmup "${WARMUP}" \
    --resolution-timeout "${RESOLUTION_TIMEOUT}" \
    --seed "${SEED}" \
    --output "${OUTPUT_DIR}/${RUN_TAG}_${PRECISION}_ctx${CONTEXT_VIEWS}_${RUN_LABEL}.json"

echo "Results: ${OUTPUT_DIR}/${RUN_TAG}_${PRECISION}_ctx${CONTEXT_VIEWS}_${RUN_LABEL}.{json,csv}"
