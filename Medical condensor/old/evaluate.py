import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kdbe_check import (
    check_omissions,
    check_omissions_bidirectional_cosine,
    check_omissions_cosine,
    check_omissions_rouge1,
)


class NlpEvaluate:
    """Tracks condenser-module runs: scores both the original (un-condensed) transcript
    and the condensed transcript against the ground-truth SOAP note via KDE-based
    scoring, and logs the result plus the difference between them.

    Unlike Modules/evaluate.py, there's no sentence-level ground truth for "was
    fluff correctly removed" -- this tracks the raw KDE scores and elapsed time per
    file instead of precision/recall.

    Two directions are tracked, and they are NOT the same kind of signal:
      - transcript_to_soap: the genuine omission direction (does the SOAP note
        cover the transcript's content).
      - groundedness_soap_in_transcript: a hallucination-adjacent signal (is the
        SOAP note's content grounded in the transcript) -- named for what it
        actually measures rather than reusing "omission" for both directions.

    diff_transcript_to_soap and diff_groundedness_soap_in_transcript are
    (original - condensed), not (condensed - original) -- deliberately, so the
    sign reads the intuitive way despite the underlying raw scores being
    LOWER-is-stronger-signal (see kdbe_check.py): condensing typically makes a
    text's own words match the other side's distribution worse, which drives
    the raw score DOWN, so (original - condensed) comes out POSITIVE. A
    positive diff here means condensing made that direction's omission/
    hallucination-adjacent signal WORSE (more likely), negative means it got
    better -- i.e. positive always correlates with "more risk," matching how
    a reader would read the number without needing to know the raw-score
    convention underneath it.

    A third, independent metric is also tracked: cosine_coverage (see
    kdbe_check.check_omissions_cosine), added after check_omissions was found
    to have a severe length bias -- a genuinely excellent, clinically-complete
    manual condensation of real transcripts scored FAR worse than several
    automated condensers here, purely because check_omissions refits a KDE
    from scratch on however many words survive condensing, and density
    estimates over small point clouds are intrinsically lower regardless of
    content quality. cosine_coverage sidesteps that (no density estimate at
    all -- just each SOAP word's best cosine match anywhere in the
    transcript), and was validated across 10 real files to track content
    quality instead of raw word count: a careful condensation barely lost
    coverage, random word deletion at the same retention lost far more.
    Unlike the KDE metrics, cosine_coverage's raw score is HIGHER-is-better
    (it's a genuine coverage score, not an inverted omission signal), so
    diff_cosine_coverage stays (condensed - original): positive means
    condensing improved that direction, negative means it got worse -- the
    OPPOSITE subtraction order from diff_transcript_to_soap/
    diff_groundedness_soap_in_transcript above, because each is already
    ordered so that positive consistently reads as "condensing made this
    worse" relative to that metric's own better/worse direction. Both metrics
    are kept side by side rather than one replacing the other, since they
    were validated on
    different things (KDE: original repo's calibration; cosine: this
    project's own 10-file manual-condensation test) and each remains useful
    to compare against the other's blind spots.

    Two more metrics extend the cosine idea to close specific gaps in it:
      - cosine_precision (see check_omissions_bidirectional_cosine): the
        mirror direction of cosine_coverage/cosine_recall. Coverage alone
        can't tell a condenser that keeps every SOAP-relevant word AND a
        pile of unrelated filler apart from one that keeps only the
        relevant words -- both would score identically on coverage.
        Precision checks whether the transcript's own surviving words are
        actually SOAP-relevant. cosine_f1 is their harmonic mean.
      - rouge1_recall/precision/f1 (see check_omissions_rouge1): the same
        recall/precision/F1 idea, but via exact word-overlap counts with no
        embeddings at all. This shares no machinery with the cosine or KDE
        metrics, so agreement between it and them is real independent
        triangulation rather than two views of the same computation.
    All five of these (cosine_recall/precision/f1, rouge1_recall/precision/f1)
    are higher-is-better with 0 as the realistic ceiling for a
    condensed-vs-original diff -- removing words can only hold a score
    steady or reduce it, never manufacture new overlap/similarity.

    Every individual record is kept (and persisted to JSON) regardless of run.
    A checkpoint -- one averaged snapshot -- is taken once per run via
    checkpoint_run(run), giving one point per run (e.g. 5 runs = 5 points)
    instead of an arbitrary batch size.
    """

    LOG_DIR = os.environ.get("RESULTS_DIR", "Logs")

    def __init__(self, module_name):
        self._module_name = module_name
        self._records = []
        self._checkpoints = []
        os.makedirs(self.LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = f"nlp_logging_{module_name}_{timestamp}.log"
        self._json_file = f"nlp_records_{module_name}_{timestamp}.json"

    @property
    def checkpoints(self):
        return self._checkpoints

    def record(self, filename, transcript, condensed_transcript, soap_ground, elapsed, run=None):
        """Scores both transcript and condensed_transcript against soap_ground, logs
        both plus the difference per direction, and stores it.

        diff_transcript_to_soap and diff_groundedness_soap_in_transcript are
        (original - condensed): positive means condensing made that KDE
        direction's risk signal WORSE (higher chance of omission/ungrounded
        content), negative means better. The other diff_* fields (cosine_*,
        rouge1_*) stay (condensed - original): positive means improved, 0 is
        the realistic ceiling. See the class docstring for why the two
        families use opposite subtraction order to reach the same "positive
        = worse for that metric" reading.

        If either side has too few recognized words to score (see
        kdbe_check.check_omissions), that direction's scores/diff are recorded as
        None rather than silently dropped -- results() reports how often this
        happens per field so a condenser that over-filters doesn't lose data
        unnoticed.
        """
        original_scores = check_omissions(transcript, soap_ground)
        condensed_scores = check_omissions(condensed_transcript, soap_ground)
        original_cosine = check_omissions_cosine(transcript, soap_ground)
        condensed_cosine = check_omissions_cosine(condensed_transcript, soap_ground)
        original_bidir = check_omissions_bidirectional_cosine(transcript, soap_ground)
        condensed_bidir = check_omissions_bidirectional_cosine(condensed_transcript, soap_ground)
        original_rouge = check_omissions_rouge1(transcript, soap_ground)
        condensed_rouge = check_omissions_rouge1(condensed_transcript, soap_ground)

        # original - condensed (not condensed - original like the diffs below):
        # positive = condensing raised the omission/ungrounded-content risk.
        diff_transcript_to_soap = self._diff(
            original_scores["omission_transcript_to_soap"], condensed_scores["omission_transcript_to_soap"]
        )
        diff_groundedness_soap_in_transcript = self._diff(
            original_scores["groundedness_soap_in_transcript"], condensed_scores["groundedness_soap_in_transcript"]
        )
        diff_cosine_coverage = self._diff(
            condensed_cosine["cosine_coverage"], original_cosine["cosine_coverage"]
        )
        diff_cosine_precision = self._diff(
            condensed_bidir["cosine_precision"], original_bidir["cosine_precision"]
        )
        diff_cosine_f1 = self._diff(condensed_bidir["cosine_f1"], original_bidir["cosine_f1"])
        diff_rouge1_recall = self._diff(condensed_rouge["rouge1_recall"], original_rouge["rouge1_recall"])
        diff_rouge1_precision = self._diff(condensed_rouge["rouge1_precision"], original_rouge["rouge1_precision"])
        diff_rouge1_f1 = self._diff(condensed_rouge["rouge1_f1"], original_rouge["rouge1_f1"])

        original_word_count = len(transcript.split())
        condensed_word_count = len(condensed_transcript.split())
        words_reduced = original_word_count - condensed_word_count
        percent_reduced = (words_reduced / original_word_count * 100) if original_word_count else 0.0

        entry = {
            "module": self._module_name,
            "run": run,
            "filename": filename,
            "elapsed": elapsed,
            "original_transcript_to_soap": original_scores["omission_transcript_to_soap"],
            "original_groundedness_soap_in_transcript": original_scores["groundedness_soap_in_transcript"],
            "condensed_transcript_to_soap": condensed_scores["omission_transcript_to_soap"],
            "condensed_groundedness_soap_in_transcript": condensed_scores["groundedness_soap_in_transcript"],
            "diff_transcript_to_soap": diff_transcript_to_soap,
            "diff_groundedness_soap_in_transcript": diff_groundedness_soap_in_transcript,
            "original_cosine_coverage": original_cosine["cosine_coverage"],
            "condensed_cosine_coverage": condensed_cosine["cosine_coverage"],
            "diff_cosine_coverage": diff_cosine_coverage,
            "original_cosine_precision": original_bidir["cosine_precision"],
            "condensed_cosine_precision": condensed_bidir["cosine_precision"],
            "diff_cosine_precision": diff_cosine_precision,
            "original_cosine_f1": original_bidir["cosine_f1"],
            "condensed_cosine_f1": condensed_bidir["cosine_f1"],
            "diff_cosine_f1": diff_cosine_f1,
            "original_rouge1_recall": original_rouge["rouge1_recall"],
            "condensed_rouge1_recall": condensed_rouge["rouge1_recall"],
            "diff_rouge1_recall": diff_rouge1_recall,
            "original_rouge1_precision": original_rouge["rouge1_precision"],
            "condensed_rouge1_precision": condensed_rouge["rouge1_precision"],
            "diff_rouge1_precision": diff_rouge1_precision,
            "original_rouge1_f1": original_rouge["rouge1_f1"],
            "condensed_rouge1_f1": condensed_rouge["rouge1_f1"],
            "diff_rouge1_f1": diff_rouge1_f1,
            "original_word_count": original_word_count,
            "condensed_word_count": condensed_word_count,
            "words_reduced": words_reduced,
            "percent_reduced": percent_reduced,
        }
        self._records.append(entry)
        self.write_json()

        self._log_(
            f"{filename}: run={run} elapsed={elapsed:.2f}s\n"
            f"  original  -- omission(transcript->soap)={self._fmt(original_scores['omission_transcript_to_soap'])} "
            f"groundedness(soap->transcript)={self._fmt(original_scores['groundedness_soap_in_transcript'])}\n"
            f"  condensed -- omission(transcript->soap)={self._fmt(condensed_scores['omission_transcript_to_soap'])} "
            f"groundedness(soap->transcript)={self._fmt(condensed_scores['groundedness_soap_in_transcript'])}\n"
            f"  diff (original - condensed, positive = condensing made it WORSE) -- "
            f"omission(transcript->soap)={self._fmt(diff_transcript_to_soap)} "
            f"groundedness(soap->transcript)={self._fmt(diff_groundedness_soap_in_transcript)}\n"
            f"  diff (condensed - original, positive = condensing IMPROVED it, 0=ceiling) -- "
            f"cosine_coverage={self._fmt(diff_cosine_coverage)} "
            f"cosine_precision={self._fmt(diff_cosine_precision)} "
            f"cosine_f1={self._fmt(diff_cosine_f1)} "
            f"rouge1_f1={self._fmt(diff_rouge1_f1)}\n"
            f"  words: {original_word_count} -> {condensed_word_count} "
            f"(reduced by {words_reduced}, {percent_reduced:.1f}%)"
        )

        return entry

    def checkpoint_run(self, run):
        """Averages every record from this run and stores/logs the snapshot.

        Call once after all files for a given run have been record()'d -- gives
        one checkpoint per run (e.g. 5 runs = 5 points), instead of an arbitrary
        batch size.
        """
        batch = [r for r in self._records if r.get("run") == run]
        if not batch:
            return None

        checkpoint = {
            "module": self._module_name,
            "run": run,
            "records_in_run": len(batch),
        }
        for key in (
            "elapsed",
            "original_transcript_to_soap",
            "original_groundedness_soap_in_transcript",
            "condensed_transcript_to_soap",
            "condensed_groundedness_soap_in_transcript",
            "diff_transcript_to_soap",
            "diff_groundedness_soap_in_transcript",
            "original_cosine_coverage",
            "condensed_cosine_coverage",
            "diff_cosine_coverage",
            "original_cosine_precision",
            "condensed_cosine_precision",
            "diff_cosine_precision",
            "original_cosine_f1",
            "condensed_cosine_f1",
            "diff_cosine_f1",
            "original_rouge1_recall",
            "condensed_rouge1_recall",
            "diff_rouge1_recall",
            "original_rouge1_precision",
            "condensed_rouge1_precision",
            "diff_rouge1_precision",
            "original_rouge1_f1",
            "condensed_rouge1_f1",
            "diff_rouge1_f1",
            "words_reduced",
            "percent_reduced",
        ):
            values = [r[key] for r in batch if r.get(key) is not None]
            checkpoint[f"avg_{key}"] = (sum(values) / len(values)) if values else None

        self._checkpoints.append(checkpoint)
        self.write_json()

        self._log_(
            f"[run {run}] avg_elapsed={self._fmt(checkpoint['avg_elapsed'])}s "
            f"avg_diff_transcript_to_soap(+worse)={self._fmt(checkpoint['avg_diff_transcript_to_soap'])} "
            f"avg_diff_groundedness_soap_in_transcript(+worse)={self._fmt(checkpoint['avg_diff_groundedness_soap_in_transcript'])} "
            f"avg_diff_cosine_coverage(+better)={self._fmt(checkpoint['avg_diff_cosine_coverage'])} "
            f"avg_diff_cosine_f1(+better)={self._fmt(checkpoint['avg_diff_cosine_f1'])} "
            f"avg_diff_rouge1_f1(+better)={self._fmt(checkpoint['avg_diff_rouge1_f1'])} "
            f"avg_percent_reduced={self._fmt(checkpoint['avg_percent_reduced'])}"
        )
        return checkpoint

    def write_json(self):
        """Persists all records and run checkpoints collected so far to disk."""
        path = os.path.join(self.LOG_DIR, self._json_file)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"module": self._module_name, "records": self._records, "checkpoints": self._checkpoints}, f, indent=2)

    def results(self):
        """Prints/logs a summary: average elapsed time and average scores for the
        original transcript, the condensed transcript, and the difference between them.

        Any field with one or more None values (too few recognized words to score --
        see check_omissions) gets its drop count reported alongside its average, so
        an over-aggressive condenser's data loss doesn't pass unnoticed.
        """
        if not self._records:
            self._log_("No results to show.")
            return

        self._log_(f"\n=== {self._module_name} Results ===")

        elapsed_values = [r["elapsed"] for r in self._records if r.get("elapsed") is not None]
        avg_elapsed = (sum(elapsed_values) / len(elapsed_values)) if elapsed_values else None
        self._log_(f"Files processed: {len(self._records)}")
        self._log_(f"Average elapsed time: {self._fmt(avg_elapsed)}s")

        self._log_(
            "Note: diff sign convention is NOT the same across all diff_* fields. "
            "diff_transcript_to_soap and diff_groundedness_soap_in_transcript are "
            "(original - condensed): POSITIVE means condensing made that KDE "
            "direction's risk WORSE (higher chance of omission/ungrounded "
            "content), negative means it got better. Every cosine_*/rouge1_* "
            "diff stays (condensed - original): POSITIVE means condensing "
            "IMPROVED it, negative means worse, and 0 is the realistic ceiling "
            "(removing words can only hold a score steady or reduce it, never "
            "manufacture new coverage). Check which family a field belongs to "
            "before reading its sign -- they point opposite ways on purpose, "
            "each chosen to match that metric's own raw-score direction."
        )

        self._log_average("Original omission (transcript->soap)", "original_transcript_to_soap")
        self._log_average("Original groundedness (soap->transcript)", "original_groundedness_soap_in_transcript")
        self._log_average("Condensed omission (transcript->soap)", "condensed_transcript_to_soap")
        self._log_average("Condensed groundedness (soap->transcript)", "condensed_groundedness_soap_in_transcript")
        self._log_average("Diff omission (transcript->soap)", "diff_transcript_to_soap")
        self._log_average("Diff groundedness (soap->transcript)", "diff_groundedness_soap_in_transcript")

        self._log_average("Original cosine coverage/recall", "original_cosine_coverage")
        self._log_average("Condensed cosine coverage/recall", "condensed_cosine_coverage")
        self._log_average("Diff cosine coverage/recall", "diff_cosine_coverage")

        self._log_average("Original cosine precision", "original_cosine_precision")
        self._log_average("Condensed cosine precision", "condensed_cosine_precision")
        self._log_average("Diff cosine precision", "diff_cosine_precision")

        self._log_average("Original cosine F1", "original_cosine_f1")
        self._log_average("Condensed cosine F1", "condensed_cosine_f1")
        self._log_average("Diff cosine F1", "diff_cosine_f1")

        self._log_average("Original ROUGE-1 recall", "original_rouge1_recall")
        self._log_average("Condensed ROUGE-1 recall", "condensed_rouge1_recall")
        self._log_average("Diff ROUGE-1 recall", "diff_rouge1_recall")

        self._log_average("Original ROUGE-1 precision", "original_rouge1_precision")
        self._log_average("Condensed ROUGE-1 precision", "condensed_rouge1_precision")
        self._log_average("Diff ROUGE-1 precision", "diff_rouge1_precision")

        self._log_average("Original ROUGE-1 F1", "original_rouge1_f1")
        self._log_average("Condensed ROUGE-1 F1", "condensed_rouge1_f1")
        self._log_average("Diff ROUGE-1 F1", "diff_rouge1_f1")

        self._log_average("Original word count", "original_word_count")
        self._log_average("Condensed word count", "condensed_word_count")
        self._log_average("Words reduced", "words_reduced")
        self._log_average("Percent reduced", "percent_reduced")

        self.write_json()

    def _log_average(self, label, key):
        values = [r[key] for r in self._records if r.get(key) is not None]
        dropped = len(self._records) - len(values)
        drop_note = f" ({dropped} record(s) dropped -- too few recognized words)" if dropped else ""
        if values:
            self._log_(f"Average {label}: {sum(values) / len(values):.2f}{drop_note}")
        else:
            self._log_(f"Average {label}: N/A{drop_note}")

    @staticmethod
    def _diff(minuend, subtrahend):
        """Generic a - b. Callers choose the argument order per metric so the
        result always reads as "positive = condensing made this worse" --
        see record()'s comment above each call site for which order that is
        for that particular metric."""
        if minuend is None or subtrahend is None:
            return None
        return minuend - subtrahend

    @staticmethod
    def _fmt(value):
        return f"{value:.2f}" if value is not None else "N/A"

    def _log_(self, info):
        """Prints and logs info."""
        print(info)
        path = os.path.join(self.LOG_DIR, self._log_file)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{info}\n")
