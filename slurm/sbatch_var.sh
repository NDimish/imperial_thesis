#!/usr/bin/env bash
#SBATCH --job-name=clinical_note_qa
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
# PLACEHOLDER -- set this to a real partition name on your cluster before
# submitting; "cpu" is a guess, not a verified value.
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
# CPU-only pipeline: all checker modules (AlignScoreChecker, SummaCChecker,
# FactKBChecker) hardcode device="cpu" today, and load_checker_modules() holds
# every model in memory at once (a handful of roberta-base-sized models) --
# 32G is a generous placeholder for that, not a measured requirement. No
# --gres=gpu is requested since nothing in this pipeline uses a GPU yet.
#SBATCH --mem=32G

# Generic SLURM launcher for the hospital clinical-documentation QA pipeline
# (SOAP note omission/hallucination checking + transcript condensing). Submits
# a target script (e.g. slurm/run_main.sh or slurm/run_nlp_main.sh) after
# activating this project's conda env, and exports a per-submission
# RESULTS_DIR so everything the run produces lands together under
# results/<timestamp>_<slurm-job-id>/ instead of scattering across the repo.
#
# Usage:
#   sbatch slurm/sbatch_var.sh slurm/run_main.sh [limit]
#   sbatch slurm/sbatch_var.sh slurm/run_nlp_main.sh [limit]
#
# Optional overrides at submission time:
#   sbatch --export=CONDA_ENV=other-env,VENV_PATH=/path/to/venv,SCRATCH_DIR=/data/you slurm/sbatch_var.sh <script> [args...]

set -euo pipefail

# --- scratch dir (quota escape hatch) ---------------------------------------
# Home directories on shared DoC-style clusters are quota-limited (often just
# ~12GB) regardless of how big the underlying filesystem's total capacity is
# -- this project's dependencies (torch, transformers, spacy, gensim's GloVe
# vectors) plus downloaded model checkpoints (AlignScore) can easily blow past
# that on their own. SCRATCH_DIR should point at a volume you've confirmed is
# actually writable for you and has real headroom -- check with e.g.:
#   touch /data/$USER/test && rm /data/$USER/test
# before trusting this default.
SCRATCH_DIR="${SCRATCH_DIR:-/data/$USER}"
mkdir -p "$SCRATCH_DIR/cache"
export HF_HOME="$SCRATCH_DIR/cache/huggingface"
export TORCH_HOME="$SCRATCH_DIR/cache/torch"
export GENSIM_DATA_DIR="$SCRATCH_DIR/cache/gensim-data"

# --- conda / venv setup -----------------------------------------------------
# CONDA_ENV: the conda env this project's checker/condenser modules are
# installed in (see requirements.txt at the repo root). Change this if you
# clone the env under a different name.
CONDA_ENV="${CONDA_ENV:-soap-checker}"
# VENV_PATH: fallback if conda itself isn't reachable on this node (e.g. no
# conda installed at all) -- point this at a plain venv's install root (the
# one containing bin/activate). Defaults under SCRATCH_DIR, not $HOME, so the
# venv itself doesn't refill your quota the way it did the first time round.
VENV_PATH="${VENV_PATH:-$SCRATCH_DIR/venv}"

activate_env() {
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV"
        return
    fi

    local conda_sh=""
    for candidate in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
        if [[ -f "$candidate" ]]; then
            conda_sh="$candidate"
            break
        fi
    done

    if [[ -n "$conda_sh" ]]; then
        # shellcheck disable=SC1090
        source "$conda_sh"
        conda activate "$CONDA_ENV"
        return
    fi

    local venv_activate="$VENV_PATH/bin/activate"
    if [[ -f "$venv_activate" ]]; then
        # shellcheck disable=SC1090
        source "$venv_activate"
        return
    fi

    echo "ERROR: conda not found, and no venv found at $venv_activate." >&2
    echo "Set CONDA_ENV to an env conda can see, or VENV_PATH to a venv root." >&2
    exit 1
}

echo "[$(date +%F' '%T)] Activating conda env: $CONDA_ENV"
activate_env
python --version

# --- results dir -------------------------------------------------------------
# Every submission gets its own folder: results/<timestamp>_<slurm-job-id>/
# The target scripts below (run_main.sh, run_nlp_main.sh) read $RESULTS_DIR,
# and the Python side (Modules/evaluate.py, Medical condensor/evaluate.py,
# main.py, NLP_Main.py) all fall back to "Logs" when RESULTS_DIR isn't set,
# so plain local runs (no sbatch) are unaffected.
RUN_TS="$(date +%Y%m%d_%H%M%S)"
JOB_CODE="${SLURM_JOB_ID:-local}"
export RESULTS_DIR="results/${RUN_TS}_${JOB_CODE}"
mkdir -p "$RESULTS_DIR"
echo "[$(date +%F' '%T)] Results dir: $RESULTS_DIR"

# --- dispatch ----------------------------------------------------------------
if [ "$#" -eq 0 ]; then
    echo "Error: No script provided."
    echo "Usage: sbatch $0 <path_to_script.sh> [args...]"
    exit 1
fi

TARGET_SCRIPT=$1
shift

if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "Error: File '$TARGET_SCRIPT' not found!"
    exit 1
fi

mkdir -p results/slurm

echo "[$(date +%F' '%T)] Launching $TARGET_SCRIPT $*"
bash "$TARGET_SCRIPT" "$@"
