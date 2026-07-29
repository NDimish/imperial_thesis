import argparse
import os
import sys

import matplotlib.pyplot as plt

from Modules.evaluate import Evaluate

sys.path.insert(0, "Medical condensor")
from medspacy_condenser import MedspacyCondenser
from negspacy_condenser import NegspacyCondenser
from quickumls_condenser import QuickUMLSCondenser
from scispacy_condenser import SciSpacyCondenser

INPUT_DIR = "prim57/cleaned transcripts"  # "Input"
OUTPUT_DIR = "prim57/bad notes lib"  # "Output"
LABELS_DIR = "prim57/bad notes labels lib"  # "Labels"
PLOT_PATH = "Logs/main_trends.png"

# Which NLP condenser module runs on each transcript before it reaches the checker
# modules below. Swap this to MedspacyCondenser / SciSpacyCondenser /
# NegspacyCondenser / QuickUMLSCondenser (needs QUICKUMLS_INSTALL_DIR configured).
CONDENSER = MedspacyCondenser

RUNS_PER_MODULE = 5

# Fixed categorical order (validated palette) -- one color per module, never cycled.
MODULE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def load_checker_modules():
    """Imports and instantiates each checker module.

    Skips (and reports) any that fail to load -- e.g. AlignScoreChecker until
    CKPT_PATH is configured with a downloaded AlignScore checkpoint.
    """
    modules = []

    try:
        from Modules.AI_checker import AIChecker
        modules.append(AIChecker())
    except Exception as e:
        print(f"Skipping AIChecker: {e}")

    try:
        from Modules.alignscore_checker import AlignScoreChecker
        modules.append(AlignScoreChecker())
    except Exception as e:
        print(f"Skipping AlignScoreChecker: {e}")

    try:
        from Modules.summac_checker import SummaCChecker
        modules.append(SummaCChecker())
    except Exception as e:
        print(f"Skipping SummaCChecker: {e}")

    try:
        from Modules.factkb_checker import FactKBChecker
        modules.append(FactKBChecker())
    except Exception as e:
        print(f"Skipping FactKBChecker: {e}")

    try:
        from Modules.kdbe_checker import KdbeChecker
        modules.append(KdbeChecker())
    except Exception as e:
        print(f"Skipping KdbeChecker: {e}")

    return modules


def read_input_file(filename, condenser):
    """Reads a single transcript file from the Input folder and runs it through
    the configured NLP condenser module (see CONDENSER at the top of this file)
    before returning it."""
    path = os.path.join(INPUT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        transcript = f.read()

    condensed, _ = condenser.condense(transcript)
    return condensed


def read_output_file(filename):
    """Reads a single SOAP record file from the Output folder."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def plot_trends(all_checkpoints):
    """Plots per-module batch-of-10 trends: precision, recall, f1, and elapsed time.
    One line per module, fixed categorical color order, four stacked single-axis
    panels (never dual-axis).
    """
    metrics = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("avg_elapsed", "Avg elapsed time (s)"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 11), sharex=True)
    fig.suptitle("Checker modules: per-module trends (averaged every 10 files)")

    for module_index, (module_name, checkpoints) in enumerate(all_checkpoints.items()):
        if not checkpoints:
            continue
        color = MODULE_COLORS[module_index % len(MODULE_COLORS)]
        x = [c["batch_index"] for c in checkpoints]

        for ax, (key, _) in zip(axes, metrics):
            y = [c.get(key) for c in checkpoints]
            ax.plot(x, y, marker="o", markersize=6, linewidth=2, color=color, label=module_name)

    for ax, (_, title) in zip(axes, metrics):
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, color="#e1e0d9", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Batch (every 10 files)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=1, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"\nSaved trend plot to {PLOT_PATH}")


def main(limit=None):
    input_files = sorted(os.listdir(INPUT_DIR))
    output_files = sorted(os.listdir(OUTPUT_DIR))

    file_count = len(input_files) if limit is None else min(limit, len(input_files))

    modules = load_checker_modules()
    print(f"Loaded {len(modules)} checker module(s).")

    condenser = CONDENSER()
    print(f"Using {condenser.__class__.__name__} to condense transcripts.")

    all_checkpoints = {}

    for module in modules:
        module_name = module.__class__.__name__
        print(f"\n=== Running {module_name} ({RUNS_PER_MODULE} runs x {file_count} files) ===")
        evaluator = Evaluate(LABELS_DIR)

        for run in range(1, RUNS_PER_MODULE + 1):
            for i in range(file_count):
                input_filename = input_files[i]
                output_filename = output_files[i]

                transcript = read_input_file(input_filename, condenser)
                soap_note = read_output_file(output_filename)

                errors, elapsed = module.check(transcript, soap_note)

                print(f"\n=== {module_name} on {input_filename} (run {run}/{RUNS_PER_MODULE}) ===")
                for error_type, error in errors:
                    print(f"{error_type}: {error}")
                print(f"Time to complete: {elapsed:.2f}s")

                evaluator.compare(errors, input_filename, elapsed, run=run)

        evaluator.results()
        all_checkpoints[module_name] = evaluator.checkpoints

    if any(all_checkpoints.values()):
        plot_trends(all_checkpoints)
    else:
        print("\nNo batches of 10+ records completed -- skipping plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOAP note checkers over transcript/output file pairs.")
    parser.add_argument(
        "limit",
        nargs="?",
        type=int,
        default=None,
        help="Number of file pairs to process per run, starting from the first (default: all files)",
    )
    args = parser.parse_args()

    print("started Main")

    main(args.limit)
