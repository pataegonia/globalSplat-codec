#!/usr/bin/env bash
#SBATCH -J globalsplat-grouped-cdf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p batch_ugrad
#SBATCH -w ariel-v10
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm/slurm-globalsplat-grouped-cdf-%j.out
#SBATCH -e logs/slurm/slurm-globalsplat-grouped-cdf-%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "${SUBMIT_DIR}/globalSplat" ]]; then
    REPO_DIR="${SUBMIT_DIR}/globalSplat"
else
    REPO_DIR="${SUBMIT_DIR}"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-globalsplat}"
PHASE1_DIR="${PHASE1_DIR:-${REPO_DIR}/outputs/cdf_phase1/ctx12_g9_t7_cal1000_eval5601}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/grouped_cdf/ctx12_g9_t7}"
BASELINE_TOTAL_BYTES="${BASELINE_TOTAL_BYTES:-3447609.138189609}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

[[ -f "${PHASE1_DIR}/histograms.npz" ]] || {
    echo "ERROR: missing ${PHASE1_DIR}/histograms.npz"; exit 1;
}
[[ -f "${PHASE1_DIR}/report.json" ]] || {
    echo "ERROR: missing ${PHASE1_DIR}/report.json"; exit 1;
}
python -c "from compressai.entropy_models.entropy_models import pmf_to_quantized_cdf" || {
    echo "ERROR: CompressAI unavailable in ${CONDA_ENV_NAME}"; exit 1;
}

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"
echo "Grouped fixed-CDF analysis from ${PHASE1_DIR}"
echo "Output: ${OUTPUT_DIR}"

srun python scripts/analysis/analyze_grouped_cdf.py \
    --histograms "${PHASE1_DIR}/histograms.npz" \
    --phase1-report "${PHASE1_DIR}/report.json" \
    --output-dir "${OUTPUT_DIR}" \
    --groups 1 4 8 16 32 512 \
    --alpha 0.5 \
    --cdf-precision 16 \
    --baseline-total-bytes "${BASELINE_TOTAL_BYTES}" \
    --binary-scale-bytes 4096

echo "Completed: ${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -maxdepth 1 -type f \
    \( -name '*.json' -o -name '*.npz' \) -print
