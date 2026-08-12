#!/usr/bin/env bash
# Submit a repository SLURM job while ensuring its log directory exists.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
mkdir -p logs/slurm

if [[ $# -lt 1 ]]; then
    echo "usage: $0 SLURM_FILE [sbatch options...]" >&2
    exit 2
fi

JOB_FILE="$1"
shift
exec sbatch "$@" "${JOB_FILE}"
