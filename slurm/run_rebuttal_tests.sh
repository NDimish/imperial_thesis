#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

QAI_CONDA_ENV="${QAI_CONDA_ENV:-dev12}"
QAI_RESULTS_ROOT="${QAI_RESULTS_ROOT:-$REPO_ROOT/results/rebuttal_hpc}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/../venv}"

# Resume support: if a previous submission got killed by the SLURM time
# limit before finishing (e.g. E4's multi-hour real eval_cnot() path -- see
# comment below), resubmitting with the SAME OUT_DIR lets
# e4_rzz_entangler_demo.py's own per-trial checkpointing pick up where it
# left off instead of starting over from trial 1 (as long as REBUTTAL_E4_*
# below are unchanged from the run being resumed -- e4_rzz_entangler_demo.py
# refuses to resume from a checkpoint whose params don't match).
# Two ways to point at a previous run:
#   RESUME_RUN_ID=20260725_143055 sbatch ...   # resume that exact run id
#   RESUME=latest sbatch ...                    # resume the most recent rebuttal_tests_* dir
# Neither set -> a fresh RUN_ID/OUT_DIR is created, same as before.
RESUME_RUN_ID="${RESUME_RUN_ID:-}"
RESUME="${RESUME:-}"
if [[ -n "$RESUME_RUN_ID" ]]; then
    RUN_ID="$RESUME_RUN_ID"
    OUT_DIR="$QAI_RESULTS_ROOT/rebuttal_tests_$RUN_ID"
    if [[ ! -d "$OUT_DIR" ]]; then
        echo "ERROR: RESUME_RUN_ID=$RESUME_RUN_ID set but $OUT_DIR does not exist." >&2
        exit 1
    fi
    RESUMED=1
elif [[ "$RESUME" == "latest" ]]; then
    LATEST_DIR="$(ls -1dt "$QAI_RESULTS_ROOT"/rebuttal_tests_*/ 2>/dev/null | head -n1 || true)"
    if [[ -z "$LATEST_DIR" ]]; then
        echo "ERROR: RESUME=latest set but no previous rebuttal_tests_* dir found under $QAI_RESULTS_ROOT." >&2
        exit 1
    fi
    OUT_DIR="${LATEST_DIR%/}"
    RUN_ID="$(basename "$OUT_DIR" | sed 's/^rebuttal_tests_//')"
    RESUMED=1
else
    RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
    OUT_DIR="$QAI_RESULTS_ROOT/rebuttal_tests_$RUN_ID"
    RESUMED=0
fi

# Which experiments to run: "3", "4", or "3 4" (default: E3+E4).
# E4's real (non-mock) eval_cnot() path is confirmed slow-but-finite (not
# hung): a reps=1/shots=64/qotp_bits=4 smoke test on 2026-07-25 completed
# in ~3021s (~50min) for its single trial with a correct result (theta
# round-trip |err|=5.87e-02 rad, disentangle sanity ok=True), confirming the
# pipeline is correct end-to-end. Back to the full experiment defaults
# (reps=20, shots=4096, qotp_bits=12) below -- expect a multi-hour run (this
# is consistent with the 17h non-completion previously seen with these same
# settings; qotp_bits cost above 4 is untested and may scale worse than
# linearly). Override via REBUTTAL_E4_REPS / REBUTTAL_E4_SHOTS /
# REBUTTAL_E4_QOTP_BITS below to shrink back down for a smoke test.
REBUTTAL_START="${REBUTTAL_START:-3}"
REBUTTAL_END="${REBUTTAL_END:-4}"
REBUTTAL_E4_REPS="${REBUTTAL_E4_REPS:-20}"
REBUTTAL_E4_SHOTS="${REBUTTAL_E4_SHOTS:-4096}"
REBUTTAL_E4_QOTP_BITS="${REBUTTAL_E4_QOTP_BITS:-12}"
export REBUTTAL_E4_REPS REBUTTAL_E4_SHOTS REBUTTAL_E4_QOTP_BITS

mkdir -p "$OUT_DIR"

log() {
    echo "[$(date +%F' '%T)] $*"
}

activate_env() {
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "$QAI_CONDA_ENV"
        return
    fi

    local conda_sh=""
    for candidate in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
        if [[ -f "$candidate" ]]; then
            conda_sh="$candidate"
            break
        fi
    done

    if [[ -z "$conda_sh" ]]; then
        local venv_activate="$VENV_PATH/bin/activate"
        if [[ -f "$venv_activate" ]]; then
            # shellcheck disable=SC1090
            source "$venv_activate"
            return
        fi

        echo "ERROR: conda not found, and venv not found at $venv_activate." >&2
        exit 1
    fi

    # shellcheck disable=SC1090
    source "$conda_sh"
    conda activate "$QAI_CONDA_ENV"
}

SUBMIT_COUNT_FILE="$OUT_DIR/.submission_count"
SUBMIT_COUNT=$(( $(cat "$SUBMIT_COUNT_FILE" 2>/dev/null || echo 0) + 1 ))
echo "$SUBMIT_COUNT" > "$SUBMIT_COUNT_FILE"

cd "$REPO_ROOT"
RUN_START_TS=$(date +%s)
if (( RESUMED )); then
    log "===== SLURM submission #$SUBMIT_COUNT -- RESUMING run $RUN_ID ====="
else
    log "===== SLURM submission #$SUBMIT_COUNT -- NEW run $RUN_ID ====="
fi
log "Experiments requested this submission: E$REBUTTAL_START-E$REBUTTAL_END"
log "Results dir: $OUT_DIR"
if (( REBUTTAL_END >= 4 && REBUTTAL_START <= 4 )); then
    log "E4 params: reps=$REBUTTAL_E4_REPS shots=$REBUTTAL_E4_SHOTS qotp_bits=$REBUTTAL_E4_QOTP_BITS"
    log "E4 checkpoints per trial to $OUT_DIR/e4_rzz_entangler_demo.json -- if this submission also hits the wall-clock limit, resubmit with RESUME_RUN_ID=$RUN_ID (or RESUME=latest) and it will continue from the next un-run trial instead of restarting."
fi
log "Activating conda env: $QAI_CONDA_ENV"
activate_env
log "Env activated."

python --version | tee -a "$OUT_DIR/python_version.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | tee -a "$OUT_DIR/nvidia_smi.txt"
else
    echo "WARNING: nvidia-smi not found (E3/E4 are CPU-bound; GPU is not required)" | tee -a "$OUT_DIR/nvidia_smi.txt"
fi

log "Launching fed_learning/rebuttal_tests/run_rebuttal_tests.sh (E${REBUTTAL_START}-E${REBUTTAL_END})"
bash fed_learning/rebuttal_tests/run_rebuttal_tests.sh \
    "$REBUTTAL_START" "$REBUTTAL_END" "$OUT_DIR" \
    2>&1 | tee -a "$OUT_DIR/rebuttal_tests.log"
RUN_ELAPSED_S=$(( $(date +%s) - RUN_START_TS ))
log "run_rebuttal_tests.sh finished after ${RUN_ELAPSED_S}s (this submission)"

cat > "$OUT_DIR/summary.txt" <<EOF
phase: rebuttal_tests
run_id: $RUN_ID
status: success
conda_env: $QAI_CONDA_ENV
experiments: E${REBUTTAL_START}-E${REBUTTAL_END}
elapsed_s_this_submission: $RUN_ELAPSED_S
submission_count: $SUBMIT_COUNT
resumed: $(( RESUMED ))
outputs:
  - nvidia_smi.txt
  - rebuttal_tests.log (appended across submissions if resumed)
  - e3_aggregation_geometry.json (if E3 ran)
  - e4_rzz_entangler_demo.json / .png (if E4 ran; JSON is resumable/checkpointed per trial)
EOF

log "Rebuttal experiments completed successfully in ${RUN_ELAPSED_S}s (submission #$SUBMIT_COUNT). Summary: $OUT_DIR/summary.txt"
