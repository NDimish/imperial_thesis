import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kdbe_check import check_omissions

BATCH_SIZE = 10


class NlpEvaluate:
    """Tracks condenser-module runs: scores both the original (un-condensed) transcript
    and the condensed transcript against the ground-truth SOAP note via KDE-based
    omission checking, and logs the result plus the difference between them.

    Unlike Modules/evaluate.py, there's no sentence-level ground truth for "was
    fluff correctly removed" -- this tracks the raw KDE omission scores and elapsed
    time per file instead of precision/recall.

    Records are grouped into batches of BATCH_SIZE; a batch average is snapshotted
    every time a batch completes, giving a trend over the course of the run rather
    than just one final average.
    """

    LOG_DIR = "Logs"

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
        """
        original_scores = check_omissions(transcript, soap_ground)
        condensed_scores = check_omissions(condensed_transcript, soap_ground)

        diff_transcript_to_soap = self._diff(
            condensed_scores["omission_transcript_to_soap"], original_scores["omission_transcript_to_soap"]
        )
        diff_soap_to_transcript = self._diff(
            condensed_scores["omission_soap_to_transcript"], original_scores["omission_soap_to_transcript"]
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
            "original_soap_to_transcript": original_scores["omission_soap_to_transcript"],
            "condensed_transcript_to_soap": condensed_scores["omission_transcript_to_soap"],
            "condensed_soap_to_transcript": condensed_scores["omission_soap_to_transcript"],
            "diff_transcript_to_soap": diff_transcript_to_soap,
            "diff_soap_to_transcript": diff_soap_to_transcript,
            "original_word_count": original_word_count,
            "condensed_word_count": condensed_word_count,
            "words_reduced": words_reduced,
            "percent_reduced": percent_reduced,
        }
        self._records.append(entry)
        self.write_json()

        self._log_(
            f"{filename}: run={run} elapsed={elapsed:.2f}s\n"
            f"  original transcript -> soap: transcript->soap={self._fmt(original_scores['omission_transcript_to_soap'])} "
            f"soap->transcript={self._fmt(original_scores['omission_soap_to_transcript'])}\n"
            f"  condensed transcript -> soap: transcript->soap={self._fmt(condensed_scores['omission_transcript_to_soap'])} "
            f"soap->transcript={self._fmt(condensed_scores['omission_soap_to_transcript'])}\n"
            f"  diff (condensed - original): transcript->soap={self._fmt(diff_transcript_to_soap)} "
            f"soap->transcript={self._fmt(diff_soap_to_transcript)}\n"
            f"  words: {original_word_count} -> {condensed_word_count} "
            f"(reduced by {words_reduced}, {percent_reduced:.1f}%)"
        )

        if len(self._records) % BATCH_SIZE == 0:
            self._checkpoint()

        return entry

    def _checkpoint(self):
        """Averages the most recent BATCH_SIZE records and stores/logs the snapshot."""
        batch = self._records[-BATCH_SIZE:]
        batch_index = len(self._records) // BATCH_SIZE

        checkpoint = {
            "module": self._module_name,
            "batch_index": batch_index,
            "records_in_batch": len(batch),
        }
        for key in (
            "elapsed",
            "original_transcript_to_soap",
            "original_soap_to_transcript",
            "condensed_transcript_to_soap",
            "condensed_soap_to_transcript",
            "diff_transcript_to_soap",
            "diff_soap_to_transcript",
            "words_reduced",
            "percent_reduced",
        ):
            values = [r[key] for r in batch if r.get(key) is not None]
            checkpoint[f"avg_{key}"] = (sum(values) / len(values)) if values else None

        self._checkpoints.append(checkpoint)
        self.write_json()

        self._log_(
            f"[batch {batch_index}] avg_elapsed={self._fmt(checkpoint['avg_elapsed'])}s "
            f"avg_diff_transcript_to_soap={self._fmt(checkpoint['avg_diff_transcript_to_soap'])} "
            f"avg_diff_soap_to_transcript={self._fmt(checkpoint['avg_diff_soap_to_transcript'])} "
            f"avg_percent_reduced={self._fmt(checkpoint['avg_percent_reduced'])}"
        )

    def write_json(self):
        """Persists all records and batch checkpoints collected so far to disk."""
        path = os.path.join(self.LOG_DIR, self._json_file)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"module": self._module_name, "records": self._records, "checkpoints": self._checkpoints}, f, indent=2)

    def results(self):
        """Prints/logs a summary: average elapsed time and average scores for the
        original transcript, the condensed transcript, and the difference between them."""
        if not self._records:
            self._log_("No results to show.")
            return

        self._log_(f"\n=== {self._module_name} Results ===")

        avg_elapsed = sum(r["elapsed"] for r in self._records) / len(self._records)
        self._log_(f"Files processed: {len(self._records)}")
        self._log_(f"Average elapsed time: {avg_elapsed:.2f}s")

        self._log_average("Original omission (transcript->soap)", "original_transcript_to_soap")
        self._log_average("Original omission (soap->transcript)", "original_soap_to_transcript")
        self._log_average("Condensed omission (transcript->soap)", "condensed_transcript_to_soap")
        self._log_average("Condensed omission (soap->transcript)", "condensed_soap_to_transcript")
        self._log_average("Difference (transcript->soap)", "diff_transcript_to_soap")
        self._log_average("Difference (soap->transcript)", "diff_soap_to_transcript")

        self._log_average("Original word count", "original_word_count")
        self._log_average("Condensed word count", "condensed_word_count")
        self._log_average("Words reduced", "words_reduced")
        self._log_average("Percent reduced", "percent_reduced")

        self.write_json()

    def _log_average(self, label, key):
        values = [r[key] for r in self._records if r.get(key) is not None]
        if values:
            self._log_(f"Average {label}: {sum(values) / len(values):.2f}")

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
