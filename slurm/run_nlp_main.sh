#!/usr/bin/env bash
# Runs NLP_Main.py -- the condenser-module side of the hospital clinical-
# documentation QA pipeline (Medspacy/SciSpacy/Negspacy/QuickUMLS condensers,
# scored via KDE-based omission checking before/after condensing).
#
# Launch via SLURM:
#   sbatch slurm/sbatch_var.sh slurm/run_nlp_main.sh [limit]
# or run standalone for a local smoke test:
#   bash slurm/run_nlp_main.sh 5
# (RESULTS_DIR falls back to "Logs" when run outside sbatch_var.sh.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RESULTS_DIR="${RESULTS_DIR:-Logs}"
mkdir -p "$RESULTS_DIR"
export RESULTS_DIR

LIMIT="${1:-}"

echo "[$(date +%F' '%T)] Starting NLP_Main.py -- results: $RESULTS_DIR"
python NLP_Main.py $LIMIT 2>&1 | tee -a "$RESULTS_DIR/nlp_main_run.log"
echo "[$(date +%F' '%T)] NLP_Main.py finished."
