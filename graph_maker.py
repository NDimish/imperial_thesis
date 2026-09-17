"""graph_maker.py -- rebuilds the thesis's data-driven figures from the
JSONs saved in data_jsons/.

Covers every numbered Figure in the paper that is an actual chart (bar/line)
backed by this project's own data:
    Figure 4  (left + right) -- corruption type / gold label distributions
    Figure 8                 -- AlignScore hallucination P/R/F1 per edit (Edit 2 slot replaced by d/p pair chunking)
    Figure 9                 -- AlignScore threshold re-sweep (d/p pair chunking)
    Figure 28                -- AlignScore threshold sweep from 0.20: Edit 1 vs d/p pair chunking
    Figure 11                -- NoteExtractor TP/FP/FN by extraction type
    Figure 12                -- NoteExtractor precision/recall/F1 by type
    Figure 13                -- NoteExtractor false-positive root causes
    Figure 14                -- NoteExtractor false-negative root causes
    Figure 21                -- EmbedKDECheck TP/FP/FN by direction
    Figure 22                -- AlignScore TP/FP/FN by direction
    Figure 23                -- EmbedKDECheck TP/FN by corruption type (Figure 4's types)
    Figure 24                -- AlignScore TP/FN by corruption type (Figure 4's types)
    Figure 25                -- medspaCy/UMLS/ConText checker TP/FN by corruption type (Figure 4's types)
    Figure 26                -- medspaCy/UMLS/ConText + number-context checker TP/FN by corruption type

Figures left out on purpose: 3a/3b (external report, not this project's
data), 5/6/10/15 (architecture/workflow diagrams, not data charts), and
7/16/17/18/19/20 (product screenshots).

Output is vector (SVG + PDF) by default, so every figure stays crisp at any
zoom level -- no raster blur. Pass --png to also/instead write a PNG.

Usage:
    python graph_maker.py                # writes every figure as SVG + PDF
    python graph_maker.py fig09          # writes only that one figure
    python graph_maker.py --formats svg  # writes only SVG
    python graph_maker.py --png          # also writes a high-dpi PNG
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DATA_DIR = "data_jsons"
OUT_DIR = "graph_maker_output"

# Vector formats by default -- infinitely zoomable, no raster blur.
DEFAULT_FORMATS = ["svg", "pdf"]

COLORS = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "grey": "#7f7f7f",
}


def _load(filename):
    with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
        return json.load(f)


def _save(fig, name):
    """Save `fig` under OUT_DIR/name (name's extension is ignored -- the
    figure is written once per format in _FORMATS, e.g. name.svg + name.pdf)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(name)[0]
    for fmt in _FORMATS:
        path = os.path.join(OUT_DIR, f"{stem}.{fmt}")
        # dpi only matters for the raster (png) case; vector formats ignore it.
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"saved {path}")
    plt.close(fig)


# Populated by main() from --formats/--png; module-level default covers
# direct calls like `python -c "import graph_maker; graph_maker.fig04()"`.
_FORMATS = list(DEFAULT_FORMATS)


def fig04():
    left = _load("fig04_left_corruption_type_distribution.json")
    right = _load("fig04_right_note_extractor_gold_label_distribution.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- left: library-based corruption types ---
    items = sorted(left["counts"].items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    total = left["total_issues"]
    bar_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["green"]]
    ax1.barh(labels, values, color=bar_colors[: len(labels)])
    for i, v in enumerate(values):
        ax1.text(v + total * 0.01, i, f"{v} ({v / total * 100:.0f}%)", va="center", fontsize=9)
    ax1.invert_yaxis()
    ax1.set_title("Library-based corruption types", fontsize=11)
    ax1.set_xlabel(f"prim_lib_injection.py -- {total} issues across {left['n_files']} notes")
    ax1.set_xlim(0, max(values) * 1.35)

    # --- right: note-extractor gold label types ---
    items2 = sorted(right["counts"].items(), key=lambda kv: kv[1], reverse=True)
    labels2 = [k for k, _ in items2]
    values2 = [v for _, v in items2]
    total2 = right["total_labels"]
    ax2.barh(labels2, values2, color=[COLORS["blue"], COLORS["orange"], COLORS["green"]][: len(labels2)])
    for i, v in enumerate(values2):
        ax2.text(v + total2 * 0.01, i, f"{v} ({v / total2 * 100:.0f}%)", va="center", fontsize=9)
    ax2.invert_yaxis()
    ax2.set_title("Note-extractor gold label types", fontsize=11)
    ax2.set_xlabel(f"labels_extraction ground truth -- {total2} labels across {right['n_files']} notes")
    ax2.set_xlim(0, max(values2) * 1.35)

    fig.suptitle("Figure: injected corruption distributions & ground-truth type labels", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "fig04_corruption_and_label_distributions.png")


def fig08():
    d = _load("fig08_alignscore_edit_stage_results.json")
    stages = d["stages"]
    labels = [s["stage"] for s in stages]
    precision = [s["precision"] for s in stages]
    recall = [s["recall"] for s in stages]
    f1 = [s["f1"] for s in stages]

    x = range(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar([i - width for i in x], precision, width, label="Precision", color=COLORS["blue"])
    ax.bar([i for i in x], recall, width, label="Recall", color=COLORS["orange"])
    ax.bar([i + width for i in x], f1, width, label="F1", color=COLORS["green"])

    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - width, p + 0.01, f"{p:.2f}", ha="center", fontsize=8)
        ax.text(i, r + 0.01, f"{r:.2f}", ha="center", fontsize=8)
        ax.text(i + width, f + 0.01, f"{f:.2f}", ha="center", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(precision + recall + f1) * 1.25)
    ax.set_title("Figure: AlignScoreChecker hallucination-direction P/R/F1 across each edit\n(same 10-file sample: prim1, prim10-prim18)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig08_alignscore_edit_stages.png")


def fig09():
    d = _load("fig09_alignscore_threshold_sweep.json")
    sweep = d["sweep"]
    thresholds = [p["threshold"] for p in sweep]
    precision = [p["precision"] for p in sweep]
    recall = [p["recall"] for p in sweep]
    f1 = [p["f1"] for p in sweep]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(thresholds, precision, marker="o", color=COLORS["blue"], label="Precision")
    ax.plot(thresholds, recall, marker="o", color=COLORS["orange"], label="Recall")
    ax.plot(thresholds, f1, marker="o", color=COLORS["green"], label="F1")

    old_label = d.get("old_threshold_label", f"old: {d['old_threshold']}")
    new_label = d.get("new_threshold_label", f"new: {d['new_threshold']} (best F1)")

    ax.axvline(d["old_threshold"], color=COLORS["grey"], linestyle="--", linewidth=1)
    ax.text(d["old_threshold"], 0.02, old_label, rotation=90, va="bottom", fontsize=8, color=COLORS["grey"])
    ax.axvline(d["new_threshold"], color="black", linestyle="--", linewidth=1)
    ax.text(d["new_threshold"], 0.02, new_label, rotation=90, va="bottom", fontsize=8)

    ax.set_xlabel("AlignScore THRESHOLD")
    ax.set_ylabel("Score")
    ax.set_title("Figure: threshold re-sweep after the splitter fix (Edit 1 + Edit 3)\nhallucination labels only, 10-file sample, scores cached from one model pass")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig09_alignscore_threshold_sweep.png")


def fig28():
    d = _load("fig28_edit1_vs_dppair_threshold_sweep.json")
    dp = [p for p in d["series"]["dp_pair_chunking"] if p["threshold"] >= 0.2]
    dp_x = [p["threshold"] for p in dp]

    fig, ax = plt.subplots(figsize=(9, 5.0))

    # d/p pair chunking only, hallucination-direction only -- same scope as
    # Figure 8, so the 0.30 point here matches Figure 8's Edit 3 bar exactly.
    ax.plot(dp_x, [p["precision"] for p in dp], marker="s", linestyle="-",
             color=COLORS["blue"], label="Precision")
    ax.plot(dp_x, [p["recall"] for p in dp], marker="s", linestyle="-",
             color=COLORS["orange"], label="Recall")
    ax.plot(dp_x, [p["f1"] for p in dp], marker="s", linestyle="-",
             color=COLORS["green"], label="F1")

    ax.axvline(0.30, color="black", linestyle=":", linewidth=1)
    ax.text(0.312, 0.58, "0.30 (Edit 3)", rotation=90, va="top", fontsize=8)

    ax.set_xlabel("AlignScore THRESHOLD")
    ax.set_ylabel("Score")
    ax.set_xlim(0.18, 0.52)
    ax.set_ylim(0.20, 0.60)
    ax.set_title(
        "Figure: d/p pair chunking Precision/Recall/F1 vs threshold from 0.20,\n"
        "hallucination labels only, 10-file sample",
        fontsize=12,
    )
    ax.legend(fontsize=9, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.22), frameon=True)

    fig.subplots_adjust(bottom=0.24)
    _save(fig, "fig28_edit1_vs_dppair_threshold_sweep.png")


def fig11():
    d = _load("fig11_fig12_note_extractor_by_type.json")
    by_type = d["by_type"]
    types = list(by_type.keys())
    tp = [by_type[t]["tp"] for t in types]
    fp = [by_type[t]["fp"] for t in types]
    fn = [by_type[t]["fn"] for t in types]

    fig, ax = plt.subplots(figsize=(9, 4))
    y = range(len(types))
    left = [0] * len(types)
    for vals, color, name in [(tp, COLORS["green"], "True positive"), (fp, COLORS["orange"], "False positive"), (fn, COLORS["red"], "False negative")]:
        ax.barh(list(y), vals, left=left, color=color, label=name)
        for i, v in enumerate(vals):
            if v:
                ax.text(left[i] + v / 2, i, str(v), ha="center", va="center", fontsize=8, color="white")
        left = [l + v for l, v in zip(left, vals)]

    totals = [t + f + n for t, f, n in zip(tp, fp, fn)]
    for i, tot in enumerate(totals):
        ax.text(tot + max(totals) * 0.01, i, f"{tot} total", va="center", fontsize=8)

    ax.set_yticks(list(y))
    ax.set_yticklabels([t.capitalize() for t in types])
    ax.invert_yaxis()
    ax.set_xlabel("Flags")
    ax.set_title(f"Figure: TP / FP / FN by extraction type ({d['file_count']} files)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    fig.tight_layout()
    _save(fig, "fig11_tp_fp_fn_by_type.png")


def fig12():
    d = _load("fig11_fig12_note_extractor_by_type.json")
    by_type = d["by_type"]
    types = list(by_type.keys())
    labels = [t.capitalize() for t in types] + ["Overall"]
    precision = [by_type[t]["precision"] for t in types] + [d["overall"]["precision"]]
    recall = [by_type[t]["recall"] for t in types] + [d["overall"]["recall"]]
    f1 = [by_type[t]["f1"] for t in types] + [d["overall"]["f1"]]

    x = range(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width for i in x], precision, width, label="Precision", color=COLORS["blue"])
    ax.bar([i for i in x], recall, width, label="Recall", color=COLORS["green"])
    ax.bar([i + width for i in x], f1, width, label="F1", color=COLORS["grey"])

    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - width, p + 0.01, f"{p:.2f}", ha="center", fontsize=8)
        ax.text(i, r + 0.01, f"{r:.2f}", ha="center", fontsize=8)
        ax.text(i + width, f + 0.01, f"{f:.2f}", ha="center", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Figure: precision, recall, and F1 by extraction type")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig12_precision_recall_f1_by_type.png")


def _cause_bar(counts, display_labels, total, title, xlabel, out_name):
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [display_labels.get(k, k) for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, values, color=COLORS["orange"] if "positive" in title.lower() else COLORS["red"])
    for i, v in enumerate(values):
        ax.text(v + total * 0.01, i, f"{v} ({v / total * 100:.0f}%)", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.35)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, out_name)


def fig13():
    d = _load("fig13_fig14_note_extractor_error_causes.json")
    _cause_bar(
        d["fp_cause_counts"],
        d["fp_cause_display_labels"],
        d["fp"],
        f"Figure: root cause of each of the {d['fp']} false positives",
        f"False positives (of {d['fp']} total)",
        "fig13_false_positive_causes.png",
    )


def fig14():
    d = _load("fig13_fig14_note_extractor_error_causes.json")
    _cause_bar(
        d["fn_cause_counts"],
        d["fn_cause_display_labels"],
        d["fn"],
        f"Figure: root cause of each of the {d['fn']} false negatives",
        f"False negatives (of {d['fn']} total)",
        "fig14_false_negative_causes.png",
    )


def _tp_fp_fn_by_type_bar(data_file, title, out_name):
    """TP/FP/FN stacked horizontal bar per label type -- same layout as fig11
    (NoteExtractor), reused here for checkers whose ground truth splits into
    hallucination/omission directions instead of extraction types."""
    d = _load(data_file)
    by_type = d["by_type"]
    types = list(by_type.keys())
    tp = [by_type[t]["tp"] for t in types]
    fp = [by_type[t]["fp"] for t in types]
    fn = [by_type[t]["fn"] for t in types]

    fig, ax = plt.subplots(figsize=(9, 3))
    y = range(len(types))
    left = [0] * len(types)
    for vals, color, name in [(tp, COLORS["green"], "True positive"), (fp, COLORS["orange"], "False positive"), (fn, COLORS["red"], "False negative")]:
        ax.barh(list(y), vals, left=left, color=color, label=name)
        for i, v in enumerate(vals):
            if v:
                ax.text(left[i] + v / 2, i, str(v), ha="center", va="center", fontsize=8, color="white")
        left = [l + v for l, v in zip(left, vals)]

    totals = [t + f + n for t, f, n in zip(tp, fp, fn)]
    for i, tot in enumerate(totals):
        ax.text(tot + max(totals) * 0.01, i, f"{tot} total", va="center", fontsize=8)

    ax.set_yticks(list(y))
    ax.set_yticklabels([t.capitalize() for t in types])
    ax.invert_yaxis()
    ax.set_xlabel("Flags")
    ax.set_title(f"{title} ({d['file_count']} files)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    _save(fig, out_name)


def fig21():
    _tp_fp_fn_by_type_bar(
        "fig21_kde_tp_fp_fn_by_type.json",
        "Figure: EmbedKDECheck TP / FP / FN by direction",
        "fig21_kde_tp_fp_fn_by_type.png",
    )


def fig22():
    _tp_fp_fn_by_type_bar(
        "fig22_alignscore_tp_fp_fn_by_type.json",
        "Figure: AlignScore TP / FP / FN by direction",
        "fig22_alignscore_tp_fp_fn_by_type.png",
    )


def _tp_fn_by_corruption_type_bar(data_file, title, out_name):
    """TP/FN stacked horizontal bar per injected corruption type (the same
    detail_type categories/order as Figure 4 left), plus a second panel for
    total false positives on its OWN axis. detail_type is metadata on a
    matched ground-truth label (Modules/evaluate.py), so an FP -- a flag
    matching no label -- has no type to plot against; total_fp is 1-2 orders
    of magnitude bigger than any typed count, so it gets its own scale
    rather than crushing the typed bars flat on a shared one."""
    d = _load(data_file)
    labels_map = d["display_labels"]
    by_detail = d["by_detail_type"]
    # Figure 4's own order: highest injected count first.
    keys = sorted(by_detail.keys(), key=lambda k: by_detail[k]["total"], reverse=True)
    tp = [by_detail[k]["tp"] for k in keys]
    fn = [by_detail[k]["fn"] for k in keys]
    row_labels = [labels_map[k] for k in keys]
    totals = [t + f for t, f in zip(tp, fn)]

    fig, (ax, ax_fp) = plt.subplots(
        1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3.2, 1], "wspace": 0.55}
    )
    y = list(range(len(keys)))

    left = [0] * len(keys)
    for vals, color, name in [(tp, COLORS["green"], "True positive"), (fn, COLORS["red"], "False negative")]:
        ax.barh(y, vals, left=left, color=color, label=name)
        for i, v in enumerate(vals):
            if v:
                ax.text(left[i] + v / 2, y[i], str(v), ha="center", va="center", fontsize=9, color="white")
        left = [l + v for l, v in zip(left, vals)]

    for i, tot in enumerate(totals):
        ax.text(tot + max(totals) * 0.015, y[i], f"{tot} labelled", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(row_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(totals) * 1.3)
    ax.set_xlabel("Flags (of ground-truth labels\nin that category)")
    ax.set_title("By corruption type (typed)", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)

    # FP has no detail_type to split by, so it's one bar on its own axis,
    # not a row squeezed onto the typed chart's much smaller scale.
    ax_fp.bar([0], [d["total_fp"]], width=0.6, color=COLORS["orange"], label="False positive")
    ax_fp.text(0, d["total_fp"] * 1.02, str(d["total_fp"]), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_fp.set_xticks([0])
    ax_fp.set_xticklabels(["FP\n(untyped)"])
    ax_fp.set_ylim(0, d["total_fp"] * 1.2)
    ax_fp.set_ylabel("Flags")
    ax_fp.set_title("False positives", fontsize=10)

    fig.suptitle(f"{title} ({d['file_count']} files)", y=1.04)
    fig.tight_layout()
    _save(fig, out_name)


def fig21():
    _tp_fp_fn_by_type_bar(
        "fig21_kde_tp_fp_fn_by_type.json",
        "Figure: EmbedKDECheck TP / FP / FN by direction",
        "fig21_kde_tp_fp_fn_by_type.png",
    )


def fig22():
    _tp_fp_fn_by_type_bar(
        "fig22_alignscore_tp_fp_fn_by_type.json",
        "Figure: AlignScore TP / FP / FN by direction",
        "fig22_alignscore_tp_fp_fn_by_type.png",
    )


def fig23():
    """TP/FN by corruption type, raw transcript vs. medspaCy-condensed
    transcript, side by side per type -- tests the Sec 9.3.6 condenser
    pre-filter fix against the published (raw) result. Separate from
    _tp_fn_by_corruption_type_bar since fig24/25 only have one condition."""
    d = _load("fig23_kde_tp_fn_by_corruption_type.json")
    labels_map = d["display_labels"]
    raw = d["by_detail_type"]
    cond = d["by_detail_type_condensed"]
    keys = sorted(raw.keys(), key=lambda k: raw[k]["total"], reverse=True)
    row_labels = [labels_map[k] for k in keys]
    n = len(keys)
    totals = [raw[k]["total"] for k in keys]  # same in both conditions

    fig, (ax, ax_fp) = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [3.2, 1], "wspace": 0.55}
    )

    bar_h = 0.34
    y_idx = list(range(n))
    y_raw = [i - bar_h / 2 - 0.02 for i in y_idx]
    y_cond = [i + bar_h / 2 + 0.02 for i in y_idx]

    def draw(y_positions, data, hatch):
        tp = [data[k]["tp"] for k in keys]
        fn = [data[k]["fn"] for k in keys]
        left = [0] * n
        for vals, color, name in [(tp, COLORS["green"], "True positive"), (fn, COLORS["red"], "False negative")]:
            ax.barh(y_positions, vals, height=bar_h, left=left, color=color, hatch=hatch, edgecolor="white", linewidth=0.7)
            for i, v in enumerate(vals):
                if v >= 3:
                    ax.text(left[i] + v / 2, y_positions[i], str(v), ha="center", va="center", fontsize=7.5, color="white")
            left = [l + v for l, v in zip(left, vals)]

    draw(y_raw, raw, hatch=None)
    draw(y_cond, cond, hatch="///")

    for i, tot in enumerate(totals):
        ax.text(tot + max(totals) * 0.015, i, f"{tot} labelled", va="center", fontsize=8)

    ax.set_yticks(y_idx)
    ax.set_yticklabels(row_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(totals) * 1.3)
    ax.set_xlabel("Flags (of ground-truth labels\nin that category)")
    ax.set_title("By corruption type (typed)", fontsize=10)
    legend_elements = [
        Patch(facecolor=COLORS["green"], label="True positive"),
        Patch(facecolor=COLORS["red"], label="False negative"),
        Patch(facecolor="white", edgecolor="black", label="Raw transcript (published)"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="medspaCy-condensed"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)

    # FP has no detail_type (see helper docstring elsewhere) -- two bars, one
    # per condition, on their own axis since both dwarf the typed counts.
    fp_vals = [d["total_fp"], d["total_fp_condensed"]]
    ax_fp.bar([0, 1], fp_vals, width=0.6, color=[COLORS["orange"], COLORS["orange"]], hatch=[None, "///"], edgecolor="white", linewidth=0.7)
    for i, v in enumerate(fp_vals):
        ax_fp.text(i, v * 1.02, str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_fp.set_xticks([0, 1])
    ax_fp.set_xticklabels(["Raw", "Condensed"])
    ax_fp.set_ylim(0, max(fp_vals) * 1.2)
    ax_fp.set_ylabel("Flags")
    ax_fp.set_title("False positives", fontsize=10)

    fig.suptitle(f"Figure: EmbedKDECheck TP / FN by corruption type -- raw vs. medspaCy-condensed ({d['file_count']} files)", y=1.04, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig23_kde_tp_fn_by_corruption_type.png")


def fig24():
    _tp_fn_by_corruption_type_bar(
        "fig24_alignscore_tp_fn_by_corruption_type.json",
        "Figure: AlignScore TP / FN by corruption type",
        "fig24_alignscore_tp_fn_by_corruption_type.png",
    )


def fig25():
    _tp_fn_by_corruption_type_bar(
        "fig25_medspacy_tp_fn_by_corruption_type.json",
        "Figure: medspaCy/UMLS/ConText checker TP / FN by corruption type",
        "fig25_medspacy_tp_fn_by_corruption_type.png",
    )


def fig26():
    _tp_fn_by_corruption_type_bar(
        "fig26_medspacy_number_tp_fn_by_corruption_type.json",
        "Figure: medspaCy/UMLS/ConText + number-context checker TP / FN by corruption type",
        "fig26_medspacy_number_tp_fn_by_corruption_type.png",
    )


FIGURES = {
    "fig04": fig04,
    "fig08": fig08,
    "fig09": fig09,
    "fig28": fig28,
    "fig11": fig11,
    "fig12": fig12,
    "fig13": fig13,
    "fig14": fig14,
    "fig21": fig21,
    "fig22": fig22,
    "fig23": fig23,
    "fig24": fig24,
    "fig25": fig25,
    "fig26": fig26,
}


def main():
    global _FORMATS

    parser = argparse.ArgumentParser(description="Rebuild the thesis's data-driven figures as vector graphics.")
    parser.add_argument("figures", nargs="*", default=[],
                         help=f"which figures to render, e.g. fig09 (default: all -- choices: {', '.join(FIGURES)})")
    parser.add_argument("--formats", nargs="+", choices=["svg", "pdf", "png"], default=DEFAULT_FORMATS,
                         help="output format(s) (default: svg pdf -- both vector, no blur on zoom)")
    parser.add_argument("--png", action="store_true", help="shorthand for --formats png (in addition to any --formats given)")
    args = parser.parse_args()

    _FORMATS = list(dict.fromkeys(args.formats + (["png"] if args.png else [])))

    requested = args.figures or list(FIGURES.keys())
    for name in requested:
        if name not in FIGURES:
            print(f"unknown figure '{name}', choices: {list(FIGURES.keys())}")
            continue
        FIGURES[name]()


if __name__ == "__main__":
    main()
