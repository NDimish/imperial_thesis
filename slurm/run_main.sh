#!/usr/bin/env bash
# Runs main.py -- the checker-module side of the hospital clinical-
# documentation QA pipeline (AIChecker, AlignScoreChecker, SummaCChecker,
# FactKBChecker, KdbeChecker flagging omissions/hallucinations between each
# transcript and its SOAP note).
#
# Launch via SLURM:
#   sbatch slurm/sbatch_var.sh slurm/run_main.sh [limit]
# or run standalone for a local smoke test:
#   bash slurm/run_main.sh 5
# (RESULTS_DIR falls back to "Logs" when run outside sbatch_var.sh.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RESULTS_DIR="${RESULTS_DIR:-Logs}"
mkdir -p "$RESULTS_DIR"
export RESULTS_DIR

LIMIT="${1:-}"

echo "[$(date +%F' '%T)] Starting main.py -- results: $RESULTS_DIR"
python main.py $LIMIT 2>&1 | tee -a "$RESULTS_DIR/main_run.log"
echo "[$(date +%F' '%T)] main.py finished."
