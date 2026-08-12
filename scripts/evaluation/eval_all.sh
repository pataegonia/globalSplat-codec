#!/usr/bin/env bash
# Run GlobalSplat's deterministic RE10K evaluation matrix.
#
# Example:
#   RE10K_ROOT=/data/re10k \
#     bash scripts/evaluation/eval_all.sh globalsplat-re10k-32k.ckpt re10k_32k
#
# Add WITH_ABLATION=1 to also run the four resolution controls from the paper.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 CHECKPOINT [EXPERIMENT]" >&2
  echo "  EXPERIMENT defaults to re10k_32k (also: re10k_16k, re10k_2k)" >&2
  exit 2
fi

CKPT="$(realpath "$1")"
EXPERIMENT="${2:-re10k_32k}"
RE10K_ROOT="${RE10K_ROOT:-../data/re10k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/eval_all}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_IMAGE="${SAVE_IMAGE:-false}"
WITH_ABLATION="${WITH_ABLATION:-0}"

[[ -f "$CKPT" ]] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }

check_dataset() {
  local name="$1" root="$2"
  [[ -f "$root/test/index.json" ]] || {
    echo "$name test index not found: $root/test/index.json" >&2
    exit 1
  }
}

check_dataset RE10K "$RE10K_ROOT"
mkdir -p "$OUTPUT_ROOT"

run_eval() {
  local family="$1" views="$2" root="$3"
  local dataset="${family}_eval_ctx${views}"
  local out="$OUTPUT_ROOT/$dataset"

  echo "===== $dataset ====="
  python -m globalsplat.main \
    "+experiment=$EXPERIMENT" \
    "dataset=$dataset" \
    mode=test \
    "checkpointing.load=$CKPT" \
    "dataset.dataset_roots=['$root']" \
    "test.output_path=$out" \
    "test.save_image=$SAVE_IMAGE" \
    optimizer.batch_size=1 \
    "optimizer.num_workers=$NUM_WORKERS" \
    trainer.devices=1
}

for views in 12 24 36; do
  run_eval re10k "$views" "$RE10K_ROOT"
done

if [[ "$WITH_ABLATION" == "1" ]]; then
  index="assets/eval_index/re10k/c3g_re10k_ctx_12v_trg_3v.json"
  run_ablation() {
    local mode="$1" resolution="$2"
    shift 2
    python scripts/evaluation/ablate_resolution.py \
      --ckpt "$CKPT" --experiment "$EXPERIMENT" --devices 1 \
      --mode "$mode" --resolution "$resolution" "$@" \
      --overrides "dataset.dataset_roots=['$RE10K_ROOT']" \
      "dataset.eval_index_path=$index" optimizer.batch_size=1 \
      "optimizer.num_workers=$NUM_WORKERS" \
      "test.output_path=$OUTPUT_ROOT/ablation/$mode"
  }

  run_ablation plain 512
  run_ablation render-only 512 --context-resolution 256
  run_ablation virtual 512 --phase 2
  run_ablation duplicate 256 --duplicate 4
fi

echo "All requested evaluations completed. Results: $OUTPUT_ROOT"
