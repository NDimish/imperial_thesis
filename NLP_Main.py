import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, "Medical condensor")

from base import clean_transcript
from evaluate import NlpEvaluate

TRANSCRIPT_DIR = "prim57/cleaned transcripts"
SOAP_GROUND_DIR = "prim57/notes cleaned"
RESULTS_DIR = os.environ.get("RESULTS_DIR", "Logs")
PLOT_PATH = os.path.join(RESULTS_DIR, "nlp_main_trends.png")

RUNS_PER_MODULE = 1

# Fixed categorical order (validated palette) -- one color per module, never cycled.
MODULE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def load_condenser_modules():
    """Imports and instantiates each condenser module.

    Skips (and reports) any that fail to load -- e.g. QuickUMLSCondenser until
    QUICKUMLS_INSTALL_DIR is configured with a local UMLS install.
    """
    modules = []

    try:
        from medspacy_condenser import MedspacyCondenser
        modules.append(MedspacyCondenser())
    except Exception as e:
        print(f"Skipping MedspacyCondenser: {e}")

    try:
        from scispacy_condenser import SciSpacyCondenser
        modules.append(SciSpacyCondenser())
    except Exception as e:
        print(f"Skipping SciSpacyCondenser: {e}")

    try:
        from negspacy_condenser import NegspacyCondenser
        modules.append(NegspacyCondenser())
    except Exception as e:
        print(f"Skipping NegspacyCondenser: {e}")

    try:
        from quickumls_condenser import QuickUMLSCondenser
        modules.append(QuickUMLSCondenser())
    except Exception as e:
        print(f"Skipping QuickUMLSCondenser: {e}")

    return modules


def read_file(directory, filename):
    with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
        return f.read()


def plot_trends(all_checkpoints):
    """Plots per-module per-run trends: elapsed time, word reduction, and the
    condensed-vs-original diff in both directions. One line per module, fixed
    categorical color order, four stacked single-axis panels (never dual-axis).
    One point per run (e.g. 5 runs = 5 points).

    The two KDE diff panels are NOT the same kind of signal: transcript->soap is
    the genuine omission direction, groundedness soap->transcript is
    hallucination-adjacent (is SOAP content grounded in the transcript). Three
    more panels track cosine/ROUGE-based alternatives, added after the KDE-based
    groundedness score was found to have a severe length bias -- it penalizes
    condensing roughly in proportion to how much text is removed, almost
    regardless of whether what's removed is real content or pure filler:
      - diff_cosine_coverage: nearest-neighbor cosine similarity, each SOAP
        word's best match anywhere in the transcript (recall-like).
      - diff_cosine_f1: adds the mirror precision direction (is what survived
        condensing actually SOAP-relevant, not filler) and combines both into
        an F1-style harmonic mean.
      - diff_rouge1_f1: the same recall/precision/F1 idea via exact word
        overlap, no embeddings at all -- independent triangulation against the
        cosine-based metrics.
    None of these fit a density estimate, so none are subject to the KDE
    metrics' small-sample bias. For every diff panel, positive = condensing
    improved that direction, negative = it got worse -- but for the
    cosine/ROUGE panels specifically, 0 is the realistic ceiling (removing
    words can only hold a score steady or reduce it, never improve it).
    """
    metrics = [
        ("avg_elapsed", "Avg elapsed time (s)"),
        ("avg_percent_reduced", "Avg words reduced (%)"),
        ("avg_diff_transcript_to_soap", "Avg diff, omission transcript->soap (condensed - original; +improved)"),
        ("avg_diff_groundedness_soap_in_transcript", "Avg diff, groundedness soap->transcript (KDE, condensed - original; +improved)"),
        ("avg_diff_cosine_coverage", "Avg diff, cosine coverage/recall (condensed - original; 0=ceiling)"),
        ("avg_diff_cosine_f1", "Avg diff, cosine precision+recall F1 (condensed - original; 0=ceiling)"),
        ("avg_diff_rouge1_f1", "Avg diff, ROUGE-1 F1, no embeddings (condensed - original; 0=ceiling)"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 2.1 * len(metrics)), sharex=True)
    fig.suptitle("Medical condensor: per-module trends (one point per run)", y=0.995)

    for module_index, (module_name, checkpoints) in enumerate(all_checkpoints.items()):
        if not checkpoints:
            continue
        color = MODULE_COLORS[module_index % len(MODULE_COLORS)]
        x = [c["run"] for c in checkpoints]

        for ax, (key, title) in zip(axes, metrics):
            y = [c.get(key) for c in checkpoints]
            ax.plot(x, y, marker="o", markersize=6, linewidth=2, color=color, label=module_name)

    for ax, (key, title) in zip(axes, metrics):
        ax.set_title(title, fontsize=10, loc="left")
        ax.axhline(0, color="#c3c2b7", linewidth=1, linestyle="--") if "diff" in key else None
        ax.grid(True, color="#e1e0d9", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Run")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
        ncol=min(len(labels), 4), frameon=False, fontsize=8,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"\nSaved trend plot to {PLOT_PATH}")


def main(limit=None):
    transcript_files = sorted(os.listdir(TRANSCRIPT_DIR))
    file_count = len(transcript_files) if limit is None else min(limit, len(transcript_files))

    modules = load_condenser_modules()
    print(f"Loaded {len(modules)} condenser module(s).")

    all_checkpoints = {}

    for module in modules:
        module_name = module.__class__.__name__
        print(f"\n=== Running {module_name} ({RUNS_PER_MODULE} runs x {file_count} files) ===")
        evaluator = NlpEvaluate(module_name)

        for run in range(1, RUNS_PER_MODULE + 1):
            for i in range(file_count):
                filename = transcript_files[i]
                transcript = clean_transcript(read_file(TRANSCRIPT_DIR, filename))
                soap_ground = read_file(SOAP_GROUND_DIR, filename)

                condensed, elapsed = module.condense(transcript)
                print(f"[run {run}/{RUNS_PER_MODULE}] [{i + 1}/{file_count}] {filename}: condensed in {elapsed:.2f}s")

                evaluator.record(filename, transcript, condensed, soap_ground, elapsed, run=run)

            evaluator.checkpoint_run(run)

        evaluator.results()
        all_checkpoints[module_name] = evaluator.checkpoints

    if any(all_checkpoints.values()):
        plot_trends(all_checkpoints)
    else:
        print("\nNo completed runs -- skipping plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run each Medical condensor NLP module over transcript/ground-SOAP pairs."
    )
    parser.add_argument(
        "limit", nargs="?", type=int, default=None, help="Number of file pairs to process per run (default: all)"
    )
    args = parser.parse_args()

    print("started NLP Main")
    main(args.limit)
