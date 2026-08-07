"""Runs the checker pipeline recommended in the MetaMapLite/QuickUMLS & CQL
write-up: one condenser, then a set of complementary checkers fanned out
over the same transcript/SOAP-note pair, then an agreement-weighted merge
into a single typed error report -- instead of just concatenating every
checker's raw output.

Every stage below is a plain True/False toggle (PIPELINE_TOGGLES section) so
any condenser or checker can be switched off without touching the rest of
the file -- e.g. to isolate one checker's behavior, or to skip a checker
whose external dependency (a UMLS install, a Gemini API key) isn't
configured on this machine.

Usage:
    python pipeline.py            # run every enabled stage over all files
    python pipeline.py 10         # same, but only the first 10 file pairs

Each enabled checker still gets its own Modules/evaluate.py-style precision/
recall log+json in Logs/ (same convention as main.py), plus one more for the
merged/aggregated output when AGGREGATE_BY_CONFIDENCE is True, so the
ensemble's real value (or lack of it) is measurable the same way every other
checker in this project already is.
"""
import argparse
import importlib
import os
import re
import sys
import time

sys.path.insert(0, "Medical condensor")
from base import clean_transcript  # noqa: E402 -- path insert must run first

from Modules.evaluate import Evaluate  # noqa: E402

# ======================================================================
# PIPELINE TOGGLES -- flip any of these to switch that stage on/off.
# ======================================================================

# Condenser: strips turns with no real asserted UMLS concept from the
# transcript before any checker sees it. Set False to run checkers on the
# raw, uncondensed transcript instead.
USE_CONDENSER = True

# Checkers -- see the accompanying HTML write-up for what each one catches
# and why it's (or isn't) included by default.
RUN_MEDSPACY_UMLS_CHECKER = True    # backbone: omission / hallucination / status_flip, negation-aware
RUN_METAMAP_CUI_CHECKER = True      # fast independent CUI cross-check (MetaMapLite, falls back to QuickUMLS)
RUN_DETERMINISTIC_CHECKER = True    # dosage / frequency / numeric mismatch
RUN_CQL_CHECKER = True              # structural rule gates (synthetic FHIR resource consistency)
RUN_AI_CHECKER = False              # Gemini LLM free-form checker -- non-deterministic, API cost/outage risk, off by default

# Aggregation: if True, cluster overlapping findings across checkers (same
# error type, substring-overlapping text) and score them by how many
# independent checkers agree. If False, every checker's raw output is just
# scored on its own -- no merged report is produced.
AGGREGATE_BY_CONFIDENCE = True

# If True (and AGGREGATE_BY_CONFIDENCE is True), the merged report keeps
# ONLY clusters 2+ distinct checkers agreed on ("high" confidence) and drops
# single-checker ("medium") clusters entirely -- an actual noise filter, not
# just a label. If False, medium-confidence clusters are kept in the merged
# report too (still labeled with their confidence).
MERGE_KEEP_ONLY_HIGH_CONFIDENCE = True

# If True, every false positive is additionally classified as "text present
# elsewhere" (a significant word from the flagged text appears somewhere in
# the OTHER document too -- the checker found something real that just
# isn't the one specific error this file's label happens to mark, e.g. a
# CUI-matching/paraphrase gap) vs. "text absent everywhere" (more likely a
# genuine noise/matching artifact). This is a cheap heuristic, not ground
# truth -- see FP_SAMPLE_SIZE below for how many examples of each get
# printed for a manual spot-check.
RUN_FP_DIAGNOSTIC = True
FP_SAMPLE_SIZE = 3

# ======================================================================

INPUT_DIR = "prim57/cleaned transcripts"
OUTPUT_DIR = "prim57/bad notes lib"
LABELS_DIR = "prim57/bad notes labels lib"
RESULTS_DIR = os.environ.get("RESULTS_DIR", "Logs")


def load_condenser():
    """Returns a MedspacyCondenser instance, or None if USE_CONDENSER is False."""
    if not USE_CONDENSER:
        return None
    from medspacy_condenser import MedspacyCondenser
    print("Condenser: MedspacyCondenser (enabled)")
    return MedspacyCondenser()


def load_checkers():
    """Instantiates every checker whose toggle is True. Skips (and reports)
    any that fail to load -- e.g. a missing QuickUMLS install or Gemini API
    key -- the same graceful-degrade pattern main.py uses, so one missing
    local dependency doesn't take down the rest of the run."""
    checkers = []

    if RUN_MEDSPACY_UMLS_CHECKER:
        try:
            from Modules.medspacy_umls_checker import MedspacyUmlsChecker
            checkers.append(("MedspacyUmlsChecker", MedspacyUmlsChecker()))
        except Exception as e:
            print(f"Skipping MedspacyUmlsChecker: {e}")

    if RUN_METAMAP_CUI_CHECKER:
        try:
            from Modules.metamap_cui_checker import MetaMapCuiChecker
            checkers.append(("MetaMapCuiChecker", MetaMapCuiChecker()))
        except Exception as e:
            print(f"Skipping MetaMapCuiChecker: {e}")

    if RUN_DETERMINISTIC_CHECKER:
        try:
            checkers.append(("DeterministicChecker", _load_deterministic_checker()))
        except Exception as e:
            print(f"Skipping DeterministicChecker: {e}")

    if RUN_CQL_CHECKER:
        try:
            from Modules.cql_checker import CqlChecker
            checkers.append(("CqlChecker", CqlChecker()))
        except Exception as e:
            print(f"Skipping CqlChecker: {e}")

    if RUN_AI_CHECKER:
        try:
            from Modules.AI_checker import AIChecker
            checkers.append(("AIChecker", AIChecker()))
        except Exception as e:
            print(f"Skipping AIChecker: {e}")

    return checkers


def _load_deterministic_checker():
    """Modules/old/deterministic_checker.py imports its parent class as
    `from Modules.concept_checker import ConceptChecker, _is_junk_concept`
    -- stale relative to its actual current location, Modules/old/, from
    before it was moved into old/. Registering the real module
    (Modules.old.concept_checker) under that stale name in sys.modules
    first lets deterministic_checker.py's own import line resolve without
    editing either file."""
    concept_checker_mod = importlib.import_module("Modules.old.concept_checker")
    sys.modules.setdefault("Modules.concept_checker", concept_checker_mod)
    from Modules.old.deterministic_checker import DeterministicChecker
    return DeterministicChecker()


# ------------------------------------------------------------------
# Agreement-weighted merge
# ------------------------------------------------------------------

def merge_findings(per_checker_errors):
    """Clusters (checker_name, error_type, error_text) findings across every
    checker that ran on one file: two findings join the same cluster if they
    share an error type and their cleaned text substring-overlaps (the same
    "clean punctuation/case, then containment-match" rule
    Modules/evaluate.py's Evaluate.compare() already uses to match a
    prediction to a ground-truth label -- reused here so "two checkers agree"
    means the same thing as "a checker matched a label").

    Returns a list of (error_type, representative_text, confidence,
    [checker_names]) tuples. confidence is "high" when 2+ distinct checkers
    landed in the same cluster, "medium" otherwise (exactly one checker).
    """
    clusters = []  # [{"type", "clean_texts": [...], "checkers": set(), "raw": [(checker, text), ...]}]

    for checker_name, errors in per_checker_errors:
        for error_type, error_text in errors:
            p_type = str(error_type).strip().lower()
            cleaned = Evaluate._clean_text(error_text)
            if not cleaned:
                continue

            match = next(
                (
                    c for c in clusters
                    if c["type"] == p_type
                    and any(cleaned in t or t in cleaned for t in c["clean_texts"])
                ),
                None,
            )
            if match is None:
                match = {"type": p_type, "clean_texts": [], "checkers": set(), "raw": []}
                clusters.append(match)

            match["clean_texts"].append(cleaned)
            match["checkers"].add(checker_name)
            match["raw"].append((checker_name, error_text))

    merged = []
    for cluster in clusters:
        confidence = "high" if len(cluster["checkers"]) >= 2 else "medium"
        representative_text = cluster["raw"][0][1]
        merged.append((cluster["type"], representative_text, confidence, sorted(cluster["checkers"])))
    return merged


# ------------------------------------------------------------------
# False-positive diagnostic
# ------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "is", "are",
    "was", "were", "be", "been", "for", "with", "this", "that", "it", "its",
    "i", "you", "he", "she", "we", "they", "not", "no", "do", "does", "did",
    "have", "has", "had", "um", "uh", "yeah", "okay", "ok", "so", "just",
    "like", "affirmed", "negated", "uncertain", "family", "history",
}
_WORD_RE = re.compile(r"[a-zA-Z']+")


def _significant_words(text):
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) >= 4 and w not in _STOPWORDS}


def _has_textual_overlap(error_text, other_doc_text):
    """True if a significant word (>=4 chars, not a stopword) from
    error_text appears verbatim in other_doc_text, or shares a >=5-char
    stem with a word there -- the same stem fallback
    MedspacyUmlsChecker._cui_present already uses for inflected-form CUI
    mismatches (vomit/vomiting), applied here at the raw-text level instead
    of the CUI level so it also catches cases where the concept matcher
    missed the word entirely, not just resolved it to a different CUI."""
    words = _significant_words(error_text)
    if not words:
        return False
    other_lower = other_doc_text.lower()
    other_stems = {w[:5] for w in _significant_words(other_doc_text)}
    return any(w in other_lower or w[:5] in other_stems for w in words)


def _classify_predictions(errors, labels):
    """Mirrors Evaluate.compare()'s own matching algorithm (same
    clean-text + substring-containment rule) to recover, for each
    prediction, whether it was a true or false positive -- Evaluate.compare()
    itself only returns aggregate counts, not a per-prediction verdict, and
    editing it to also return one felt riskier than re-deriving the same
    (already-tested) logic here."""
    matched = [False] * len(labels)
    classified = []
    for error_type, error_text in errors:
        p_type = str(error_type).strip().lower()
        p_clean = Evaluate._clean_text(error_text)
        is_tp = False
        for i, label in enumerate(labels):
            if matched[i] or p_type != label["type"]:
                continue
            if p_clean and (p_clean in label["detail"] or label["detail"] in p_clean):
                matched[i] = True
                is_tp = True
                break
        classified.append((error_type, error_text, is_tp))
    return classified


def _diagnose_checker_output(name, errors, labels, transcript, soap_note, acc):
    """Classifies every prediction from one checker on one file as TP, or a
    false positive bucketed by _has_textual_overlap, and accumulates counts
    + a few example texts per bucket into acc[name]."""
    bucket = acc.setdefault(name, {
        "tp": 0, "fp_overlap": 0, "fp_no_overlap": 0,
        "samples_overlap": [], "samples_no_overlap": [],
    })
    for error_type, error_text, is_tp in _classify_predictions(errors, labels):
        if is_tp:
            bucket["tp"] += 1
            continue
        other_doc = soap_note if str(error_type).strip().lower() == "omission" else transcript
        if _has_textual_overlap(error_text, other_doc):
            bucket["fp_overlap"] += 1
            samples = bucket["samples_overlap"]
        else:
            bucket["fp_no_overlap"] += 1
            samples = bucket["samples_no_overlap"]
        if len(samples) < FP_SAMPLE_SIZE:
            samples.append((error_type, error_text))


def _print_fp_diagnostics(acc):
    print("\n=== False-positive diagnostic (heuristic, not ground truth) ===")
    print("'text present elsewhere' = a significant word from the flagged text also appears in the")
    print("other document -- the checker likely found something real, just not this file's one labeled")
    print("error. 'text absent everywhere' = found in neither document's raw text at all -- more likely")
    print("a genuine matching artifact (denylist gap, junk concept, duplicate-turn noise).\n")
    for name, bucket in acc.items():
        total_fp = bucket["fp_overlap"] + bucket["fp_no_overlap"]
        pct_overlap = (bucket["fp_overlap"] / total_fp * 100) if total_fp else 0.0
        print(
            f"{name}: TP={bucket['tp']} FP={total_fp} "
            f"-> {bucket['fp_overlap']} ({pct_overlap:.0f}%) text present elsewhere, "
            f"{bucket['fp_no_overlap']} ({100 - pct_overlap:.0f}%) text absent everywhere"
        )
        if bucket["samples_overlap"]:
            print("  sample 'text present elsewhere':")
            for etype, text in bucket["samples_overlap"]:
                print(f"    [{etype}] {text[:110]}")
        if bucket["samples_no_overlap"]:
            print("  sample 'text absent everywhere':")
            for etype, text in bucket["samples_no_overlap"]:
                print(f"    [{etype}] {text[:110]}")


# ------------------------------------------------------------------
# File I/O (same layout as main.py)
# ------------------------------------------------------------------

def read_input_file(filename, condenser):
    path = os.path.join(INPUT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        transcript = clean_transcript(f.read())
    if condenser is not None:
        transcript, _ = condenser.condense(transcript)
    return transcript


def read_output_file(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

def main(limit=None):
    input_files = sorted(os.listdir(INPUT_DIR))
    output_files = sorted(os.listdir(OUTPUT_DIR))
    file_count = len(input_files) if limit is None else min(limit, len(input_files))

    condenser = load_condenser()
    checkers = load_checkers()
    print(f"Loaded {len(checkers)} checker module(s): {[name for name, _ in checkers]}")
    print(f"Aggregation: {'agreement-weighted merge' if AGGREGATE_BY_CONFIDENCE else 'disabled (raw output only)'}")

    evaluators = {name: Evaluate(LABELS_DIR, name) for name, _ in checkers}
    merged_name = "Pipeline_Merged_HighConfidence" if MERGE_KEEP_ONLY_HIGH_CONFIDENCE else "Pipeline_Merged"
    merged_evaluator = Evaluate(LABELS_DIR, merged_name) if AGGREGATE_BY_CONFIDENCE else None
    fp_diagnostics = {} if RUN_FP_DIAGNOSTIC else None

    for i in range(file_count):
        input_filename = input_files[i]
        output_filename = output_files[i]
        print(f"\n=== [{i + 1}/{file_count}] {input_filename} ===")

        transcript = read_input_file(input_filename, condenser)
        soap_note = read_output_file(output_filename)

        per_checker_errors = []
        for name, checker in checkers:
            try:
                errors, elapsed = checker.check(transcript, soap_note)
            except Exception as e:
                print(f"  {name}: FAILED -- {type(e).__name__}: {e}")
                continue

            print(f"  {name}: {len(errors)} finding(s) in {elapsed:.2f}s")
            evaluators[name].compare(errors, input_filename, elapsed)
            per_checker_errors.append((name, errors))

            if RUN_FP_DIAGNOSTIC:
                labels = evaluators[name].clean_labels(input_filename)
                _diagnose_checker_output(name, errors, labels, transcript, soap_note, fp_diagnostics)

        if AGGREGATE_BY_CONFIDENCE:
            merged = merge_findings(per_checker_errors)
            high = sum(1 for _, _, conf, _ in merged if conf == "high")
            dropped = 0
            if MERGE_KEEP_ONLY_HIGH_CONFIDENCE:
                dropped = len(merged) - high
                merged = [m for m in merged if m[2] == "high"]
            print(f"  Merged: {len(merged)} finding(s) ({high} high-confidence"
                  + (f", {dropped} medium-confidence dropped)" if MERGE_KEEP_ONLY_HIGH_CONFIDENCE else ")"))
            for error_type, text, confidence, sources in merged:
                print(f"    [{confidence}] {error_type}: {text[:100]} (from: {', '.join(sources)})")
            merged_pairs = [(error_type, text) for error_type, text, _confidence, _sources in merged]
            merged_evaluator.compare(merged_pairs, input_filename)

    print("\n=== Per-checker results (overall + by error type) ===")
    for name in evaluators:
        _print_scores(name, evaluators[name].results())

    if merged_evaluator is not None:
        print(f"\n=== Merged pipeline results ({merged_name}) ===")
        _print_scores(merged_name, merged_evaluator.results())

    if fp_diagnostics is not None:
        _print_fp_diagnostics(fp_diagnostics)


def _print_scores(name, scores):
    """Prints overall precision/recall/F1 plus a separate line per error
    type (omission, hallucination, status_flip, ...) -- scores() already
    computes this breakdown (Evaluate.type_totals), it just isn't printed
    to console by default since Evaluate only logs to file."""
    o = scores["overall"]
    print(f"{name}: overall precision={o['precision']:.3f} recall={o['recall']:.3f} f1={o['f1']:.3f}")
    by_type = scores["by_type"]
    if not by_type:
        print("  (no typed findings)")
        return
    for error_type in sorted(by_type):
        t = by_type[error_type]
        print(
            f"  {error_type:<14} precision={t['precision']:.3f} recall={t['recall']:.3f} "
            f"f1={t['f1']:.3f}  (tp={t['tp']} fp={t['fp']} fn={t['fn']})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the recommended checker pipeline over transcript/SOAP-note pairs.")
    parser.add_argument(
        "limit",
        nargs="?",
        type=int,
        default=None,
        help="Number of file pairs to process, starting from the first (default: all files)",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    main(args.limit)
    print(f"\nTotal pipeline time: {time.perf_counter() - start:.1f}s")
