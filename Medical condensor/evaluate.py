import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kdbe_check import check_omissions


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

    For both diff_* fields (condensed - original): LOWER raw scores are the
    stronger signal (see kdbe_check.py), so a POSITIVE diff means condensing
    IMPROVED that direction (reduced the signal), and a NEGATIVE diff means it
    got WORSE -- the opposite of the usual "a bigger number is worse" reading.

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
        both plus the difference (condensed - original) per direction, and stores it.

        If either side has too few recognized words to score (see
        kdbe_check.check_omissions), that direction's scores/diff are recorded as
        None rather than silently dropped -- results() reports how often this
        happens per field so a condenser that over-filters doesn't lose data
        unnoticed.
        """
        original_scores = check_omissions(transcript, soap_ground)
        condensed_scores = check_omissions(condensed_transcript, soap_ground)

        diff_transcript_to_soap = self._diff(
            condensed_scores["omission_transcript_to_soap"], original_scores["omission_transcript_to_soap"]
        )
        diff_groundedness_soap_in_transcript = self._diff(
            condensed_scores["groundedness_soap_in_transcript"], original_scores["groundedness_soap_in_transcript"]
        )

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
            f"  diff (condensed - original, positive = improved) -- "
            f"omission(transcript->soap)={self._fmt(diff_transcript_to_soap)} "
            f"groundedness(soap->transcript)={self._fmt(diff_groundedness_soap_in_transcript)}\n"
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
            "words_reduced",
            "percent_reduced",
        ):
            values = [r[key] for r in batch if r.get(key) is not None]
            checkpoint[f"avg_{key}"] = (sum(values) / len(values)) if values else None

        self._checkpoints.append(checkpoint)
        self.write_json()

        self._log_(
            f"[run {run}] avg_elapsed={self._fmt(checkpoint['avg_elapsed'])}s "
            f"avg_diff_transcript_to_soap={self._fmt(checkpoint['avg_diff_transcript_to_soap'])} "
            f"avg_diff_groundedness_soap_in_transcript={self._fmt(checkpoint['avg_diff_groundedness_soap_in_transcript'])} "
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
            "Note: diff_* is (condensed - original); positive means condensing "
            "IMPROVED that direction (lower raw scores are the stronger signal), "
            "negative means it got worse."
        )

        self._log_average("Original omission (transcript->soap)", "original_transcript_to_soap")
        self._log_average("Original groundedness (soap->transcript)", "original_groundedness_soap_in_transcript")
        self._log_average("Condensed omission (transcript->soap)", "condensed_transcript_to_soap")
        self._log_average("Condensed groundedness (soap->transcript)", "condensed_groundedness_soap_in_transcript")
        self._log_average("Diff omission (transcript->soap)", "diff_transcript_to_soap")
        self._log_average("Diff groundedness (soap->transcript)", "diff_groundedness_soap_in_transcript")

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
    def _diff(condensed_value, original_value):
        if condensed_value is None or original_value is None:
            return None
        return condensed_value - original_value

    @staticmethod
    def _fmt(value):
        return f"{value:.2f}" if value is not None else "N/A"

    def _log_(self, info):
        """Prints and logs info."""
        print(info)
        path = os.path.join(self.LOG_DIR, self._log_file)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{info}\n")
