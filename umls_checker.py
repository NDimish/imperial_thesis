"""Gold-standard entity-linking evaluation for the shared QuickUMLS matcher
(Medical condensor/umls_matching.py) -- the one component every condenser
and checker in this project depends on for "is this a real clinical
concept", but which had never been measured directly on its own. Every
other metric built this session (cosine coverage, ROUGE, checker
precision/recall) evaluates a downstream task; this evaluates the matching
layer itself, the same way an NER/entity-linking system is normally scored:
precision and recall against a human-annotated gold standard.

Two phases, run from the same script:

  extract -- scans FILE_LIMIT real transcript/SOAP files, runs the matcher
    exactly as currently configured (get_matcher(), i.e. threshold=0.75,
    ACCEPTED_SEMTYPES-restricted -- the real instance every condenser and
    checker uses), and aggregates every match by UNIQUE (term, semtypes)
    pair rather than by individual occurrence. Annotating every one of the
    thousands of raw occurrences across 20 files isn't tractable by hand;
    annotating the much smaller set of unique terms is, and each
    occurrence inherits its term's label when scoring. Saved to
    "umls span docs/" as one JSON file per source file, each entry with a
    "label" field left null for a human annotator to fill in with true
    (genuinely a clinical concept in this dataset) or false (a coincidental
    UMLS collision -- the same kind of judgment this session's denylists
    were built from, just applied completely and systematically here
    instead of only to whatever happened to surface as an obvious problem).

  score -- reads the annotated files back, computes precision (of
    everything the matcher flagged, how much is real) split out by (a) the
    matcher's raw output and (b) after the project's current denylist
    filtering (umls_matching.is_common_word), so the denylist's actual
    marginal contribution is visible as a real number instead of an
    assumption. Also reports separately by source register (transcript vs
    SOAP note), since this project's whole domain-mismatch thread predicts
    these may differ. Plots the results to Logs/umls_matching_precision.png.

Recall (of everything a human would call a real concept, how much did the
matcher actually find) is NOT measured here -- that requires reading raw
text for concepts that were never matched at all, a fundamentally different
and much slower annotation task than judging matches that already exist.
Documented as a known scope limit, not silently skipped.
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, "Medical condensor")
from base import clean_transcript
from umls_matching import ACCEPTED_SEMTYPES, GENERIC_WORD_DENYLIST, MIN_MATCH_LENGTH, THRESHOLD, is_common_word

from quickumls import QuickUMLS

TRANSCRIPT_DIR = "prim57/cleaned transcripts"
SOAP_GROUND_DIR = "prim57/notes cleaned"
SPAN_DOCS_DIR = "umls span docs"
RESULTS_DIR = os.environ.get("RESULTS_DIR", "Logs")
PLOT_PATH = os.path.join(RESULTS_DIR, "umls_matching_precision.png")

FILE_LIMIT = 20
CONTEXT_CHARS = 40  # characters of surrounding text saved per example, for the annotator to judge in context


def _get_raw_matcher(threshold=None):
    """A matcher tuned exactly like the project's shared get_matcher() --
    same threshold, same accepted semantic types -- but constructed
    directly here (not via umls_matching.get_matcher()'s cached singleton)
    so this script can extract matches without needing the project's
    QUICKUMLS_INSTALL_DIR wiring to already be configured for a different
    module's import side effects.

    threshold overrides the project's real THRESHOLD (0.75) for one-off
    experiments -- e.g. threshold=1.0 disables fuzzy matching entirely
    (QuickUMLS has no separate "exact match" flag; a similarity threshold of
    1.0 is what that means in its API -- only spans whose character-ngram
    profile is identical to a UMLS term survive). This does NOT change the
    live matcher every condenser/checker uses (umls_matching.THRESHOLD is
    untouched) -- that value was already tuned down from 0.85 in a real
    full-dataset test after raising it cost real matches like
    "smoke"->"smoker" (0.75) and "peanuts"->"peanut" (0.80). This override
    exists only to measure what an exact-match-only matcher's precision
    would look like, as a separate question from what the condensers
    should actually run with.
    """
    from umls_matching import QUICKUMLS_INSTALL_DIR
    return QuickUMLS(
        QUICKUMLS_INSTALL_DIR,
        threshold=THRESHOLD if threshold is None else threshold,
        min_match_length=MIN_MATCH_LENGTH,
        accepted_semtypes=ACCEPTED_SEMTYPES,
    )


def extract(threshold=None, span_dir=None):
    span_dir = span_dir or SPAN_DOCS_DIR
    os.makedirs(span_dir, exist_ok=True)
    matcher = _get_raw_matcher(threshold=threshold)

    files = sorted(os.listdir(TRANSCRIPT_DIR))[:FILE_LIMIT]
    print(f"Extracting UMLS matches from {len(files)} files (threshold={THRESHOLD if threshold is None else threshold}): {files}")

    for filename in files:
        with open(os.path.join(TRANSCRIPT_DIR, filename), encoding="utf-8") as f:
            transcript = clean_transcript(f.read())
        with open(os.path.join(SOAP_GROUND_DIR, filename), encoding="utf-8") as f:
            soap_note = f.read()

        terms = {}  # (term, frozenset(semtypes)) -> entry dict
        for source_name, text in (("transcript", transcript), ("soap_note", soap_note)):
            matches = matcher.match(text, best_match=True, ignore_syntax=False)
            for group in matches:
                for m in group:
                    key = (m["term"], frozenset(m["semtypes"]))
                    entry = terms.setdefault(key, {
                        "term": m["term"],
                        "semtypes": sorted(m["semtypes"]),
                        "count_transcript": 0,
                        "count_soap_note": 0,
                        "currently_denylisted": is_common_word(m["term"]),
                        "example_contexts": [],
                        "label": None,
                    })
                    entry[f"count_{source_name}"] += 1
                    if len(entry["example_contexts"]) < 3:
                        start, end = m["start"], m["end"]
                        ctx_start = max(0, start - CONTEXT_CHARS)
                        ctx_end = min(len(text), end + CONTEXT_CHARS)
                        entry["example_contexts"].append(
                            f"[{source_name}] ...{text[ctx_start:start]}[[{text[start:end]}]]{text[end:ctx_end]}..."
                        )

        out_path = os.path.join(span_dir, filename.replace(".txt", ".json"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sorted(terms.values(), key=lambda e: -(e["count_transcript"] + e["count_soap_note"])), f, indent=2)
        print(f"  {filename}: {len(terms)} unique (term, semtype) matches -> {out_path}")

    total_unlabeled = sum(
        1
        for filename in os.listdir(span_dir)
        for entry in json.load(open(os.path.join(span_dir, filename), encoding="utf-8"))
        if entry["label"] is None
    )
    print(f"\n{total_unlabeled} unique (term, semtype) entries across all files need a 'label': true/false.")
    print(f"Annotate them directly in the JSON files under {span_dir}/, then run: python umls_checker.py score")


def score(span_dir=None, plot_path=None, label=""):
    span_dir = span_dir or SPAN_DOCS_DIR
    plot_path = plot_path or PLOT_PATH
    files = sorted(f for f in os.listdir(span_dir) if f.endswith(".json"))
    if not files:
        print(f"No annotated files found in {span_dir}/ -- run 'python umls_checker.py extract' first.")
        return

    per_file = []
    unlabeled_count = 0
    raw_true = raw_total = 0
    filtered_true = filtered_total = 0
    by_source = {
        "transcript": {"true": 0, "total": 0},
        "soap_note": {"true": 0, "total": 0},
    }

    for filename in files:
        with open(os.path.join(span_dir, filename), encoding="utf-8") as f:
            entries = json.load(f)

        file_raw_true = file_raw_total = 0
        file_filtered_true = file_filtered_total = 0

        for entry in entries:
            if entry["label"] is None:
                unlabeled_count += 1
                continue
            label = bool(entry["label"])
            count = entry["count_transcript"] + entry["count_soap_note"]

            raw_total += count
            file_raw_total += count
            if label:
                raw_true += count
                file_raw_true += count

            # "filtered" = what the project's real pipeline actually keeps,
            # i.e. anything the current denylist would already exclude is
            # removed from consideration entirely (matching has_real_concept's
            # real behavior), not counted as either a hit or a miss.
            if not entry["currently_denylisted"]:
                filtered_total += count
                file_filtered_total += count
                if label:
                    filtered_true += count
                    file_filtered_true += count

            for source in ("transcript", "soap_note"):
                source_count = entry[f"count_{source}"]
                by_source[source]["total"] += source_count
                if label:
                    by_source[source]["true"] += source_count

        per_file.append({
            "filename": filename.replace(".json", ".txt"),
            "raw_precision": (file_raw_true / file_raw_total) if file_raw_total else None,
            "filtered_precision": (file_filtered_true / file_filtered_total) if file_filtered_total else None,
        })

    if unlabeled_count:
        print(f"WARNING: {unlabeled_count} entries are still unlabeled (label: null) -- excluded from scoring.")

    raw_precision = raw_true / raw_total if raw_total else 0.0
    filtered_precision = filtered_true / filtered_total if filtered_total else 0.0

    print(f"\n=== UMLS matching precision{label}, {len(files)} files, {raw_total} labeled occurrences ===")
    print(f"Raw matcher precision (before denylist): {raw_precision:.3f} ({raw_true}/{raw_total})")
    print(f"After current denylist filtering ({len(GENERIC_WORD_DENYLIST)} terms): {filtered_precision:.3f} ({filtered_true}/{filtered_total})")
    print(f"Denylist's actual marginal contribution: {filtered_precision - raw_precision:+.3f}")
    for source, stats in by_source.items():
        p = stats["true"] / stats["total"] if stats["total"] else 0.0
        print(f"  {source}: precision={p:.3f} ({stats['true']}/{stats['total']})")

    # "Accuracy" here is numerically identical to raw precision, not a
    # separate number -- explicitly, not silently. This evaluation only ever
    # sampled and labeled things the matcher DID match (true positives +
    # false positives); it never sampled spans the matcher SKIPPED, so there
    # are no labeled true negatives or false negatives to fold into a real
    # accuracy = (TP+TN)/(TP+TN+FP+FN). Printed anyway because it was asked
    # for, with this caveat attached rather than a number that looks like a
    # different, independent metric when it isn't one.
    print(
        f"\n'Accuracy' over the labeled sample (TP/(TP+FP) -- see note): {raw_precision:.3f} "
        f"-- IDENTICAL to raw precision above, not an independent number: this evaluation never "
        f"labeled the spans the matcher failed to match, so there are no measured TN/FN to fold in."
    )
    print("Recall was not measured (see module docstring) -- this is a precision-only evaluation.")

    _plot(per_file, raw_precision, filtered_precision, by_source, plot_path)
    return raw_precision, filtered_precision


def _plot(per_file, raw_precision, filtered_precision, by_source, plot_path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 12))
    fig.suptitle("UMLS matcher precision -- gold-standard entity-linking evaluation", y=0.99)

    # Panel 1: per-file precision, raw vs filtered
    filenames = [pf["filename"] for pf in per_file]
    raw_vals = [pf["raw_precision"] for pf in per_file]
    filtered_vals = [pf["filtered_precision"] for pf in per_file]
    x = range(len(filenames))
    axes[0].bar([i - 0.2 for i in x], raw_vals, width=0.4, label="Raw matcher", color="#B5502A")
    axes[0].bar([i + 0.2 for i in x], filtered_vals, width=0.4, label="After denylist filtering", color="#1F6F78")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(filenames, rotation=60, ha="right", fontsize=7)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Precision per file", fontsize=10, loc="left")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(True, axis="y", color="#e1e0d9", linewidth=0.8)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Panel 2: overall raw vs filtered
    axes[1].bar(["Raw matcher", "After denylist"], [raw_precision, filtered_precision], color=["#B5502A", "#1F6F78"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Overall precision, all files combined", fontsize=10, loc="left")
    for i, v in enumerate([raw_precision, filtered_precision]):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].grid(True, axis="y", color="#e1e0d9", linewidth=0.8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    # Panel 3: by source register
    sources = list(by_source.keys())
    precisions = [by_source[s]["true"] / by_source[s]["total"] if by_source[s]["total"] else 0.0 for s in sources]
    axes[2].bar(sources, precisions, color=["#eda100", "#008300"])
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Precision by source register (transcript vs SOAP note)", fontsize=10, loc="left")
    for i, v in enumerate(precisions):
        axes[2].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    axes[2].grid(True, axis="y", color="#e1e0d9", linewidth=0.8)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold-standard entity-linking evaluation for the shared UMLS matcher.")
    parser.add_argument("phase", choices=["extract", "score"], help="extract candidate matches, or score already-annotated files")
    parser.add_argument("--threshold", type=float, default=None, help="override QuickUMLS similarity threshold for extract (e.g. 1.0 = exact matches only, no fuzzy matching)")
    parser.add_argument("--span-dir", default=None, help="override the span-docs directory (keeps this run separate from the main gold-standard set)")
    parser.add_argument("--plot-path", default=None, help="override the output plot path")
    parser.add_argument("--label", default="", help="short text appended to the score() summary header, e.g. ' (exact match only)'")
    args = parser.parse_args()

    if args.phase == "extract":
        extract(threshold=args.threshold, span_dir=args.span_dir)
    else:
        score(span_dir=args.span_dir, plot_path=args.plot_path, label=args.label)
