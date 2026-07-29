import json
import os
from datetime import datetime

BATCH_SIZE = 10


class Evaluate:
    """Compares predicted errors against ground-truth Labels and tracks precision/recall/F1/accuracy.

    Records are grouped into batches of BATCH_SIZE; a batch average is snapshotted
    every time a batch completes, giving a trend over the course of the run rather
    than just one final average.
    """

    LABELS_DIR = "Labels"
    LOG_DIR = "Logs"

    def __init__(self, labels_dir=None):
        self._records = []
        self._checkpoints = []
        os.makedirs(self.LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = f"evaluate_{timestamp}.log"
        self._json_file = f"evaluate_records_{timestamp}.json"
        if labels_dir is not None:
            self.LABELS_DIR = labels_dir

    @property
    def checkpoints(self):
        return self._checkpoints

    def compare(self, errors, filename, elapsed=None, run=None):
        """Compares predicted (type, error) pairs against the ground-truth labels for filename.

        errors: iterable of (type, error) pairs, as returned by a CheckerModule.check().
        filename: name of the file in Labels/ holding the ground-truth JSON errors.
        elapsed: seconds the module took to produce errors, if available.
        run: which repeat of the run this came from, if tracking multiple runs.

        A prediction counts as a true positive if its type matches a not-yet-matched
        ground-truth error of the same type and the error text overlaps (case-insensitive).
        Stores the per-file result and returns it.
        """
        true_errors = self._load_labels(filename)

        predicted = [(str(t).strip().lower(), str(e).strip().lower()) for t, e in errors]
        actual = [
            (str(item.get("type", "")).strip().lower(), str(item.get("error", "")).strip().lower())
            for item in true_errors
        ]

        matched_actual = [False] * len(actual)
        tp = 0
        fp = 0
        extra = []

        for p_type, p_error in predicted:
            match_found = False
            for i, (a_type, a_error) in enumerate(actual):
                if matched_actual[i]:
                    continue
                if p_type == a_type and (p_error == a_error or p_error in a_error or a_error in p_error):
                    matched_actual[i] = True
                    match_found = True
                    break
            if match_found:
                tp += 1
            else:
                fp += 1
                extra.append({"type": p_type, "error": p_error})

        missed = [true_errors[i] for i, matched in enumerate(matched_actual) if not matched]
        fn = len(missed)

        record = {
            "filename": filename,
            "run": run,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "elapsed": elapsed,
            **self._scores(tp, fp, fn),
        }
        self._records.append(record)
        self._log_missed(filename, missed, extra)
        self.write_json()

        if len(self._records) % BATCH_SIZE == 0:
            self._checkpoint()

        return record

    def _checkpoint(self):
        """Averages the most recent BATCH_SIZE records and stores/logs the snapshot."""
        batch = self._records[-BATCH_SIZE:]
        batch_index = len(self._records) // BATCH_SIZE

        batch_tp = sum(r["tp"] for r in batch)
        batch_fp = sum(r["fp"] for r in batch)
        batch_fn = sum(r["fn"] for r in batch)
        scores = self._scores(batch_tp, batch_fp, batch_fn)

        elapsed_values = [r["elapsed"] for r in batch if r.get("elapsed") is not None]

        checkpoint = {
            "batch_index": batch_index,
            "records_in_batch": len(batch),
            "avg_elapsed": (sum(elapsed_values) / len(elapsed_values)) if elapsed_values else None,
            **scores,
        }
        self._checkpoints.append(checkpoint)
        self.write_json()

        self._log_(
            f"[batch {batch_index}] precision={scores['precision']:.2f} recall={scores['recall']:.2f} "
            f"f1={scores['f1']:.2f} avg_elapsed={checkpoint['avg_elapsed']}"
        )

    def write_json(self):
        """Persists all records and batch checkpoints collected so far to disk."""
        path = os.path.join(self.LOG_DIR, self._json_file)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"records": self._records, "checkpoints": self._checkpoints}, f, indent=2)

    def results(self):
        """Prints per-file and overall precision/recall/F1/accuracy."""
        if not self._records:
            self._log_("No results to show.")
            return

        self._log_("\n=== Evaluation Results ===")
        for r in self._records:
            self._log_(
                f"{r['filename']}: precision={r['precision']:.2f} recall={r['recall']:.2f} "
                f"f1={r['f1']:.2f} accuracy={r['accuracy']:.2f} (TP={r['tp']} FP={r['fp']} FN={r['fn']})"
            )

        total_tp = sum(r["tp"] for r in self._records)
        total_fp = sum(r["fp"] for r in self._records)
        total_fn = sum(r["fn"] for r in self._records)
        overall = self._scores(total_tp, total_fp, total_fn)

        self._log_(
            f"\nOverall: precision={overall['precision']:.2f} recall={overall['recall']:.2f} "
            f"f1={overall['f1']:.2f} accuracy={overall['accuracy']:.2f} "
            f"(TP={total_tp} FP={total_fp} FN={total_fn})"
        )

        elapsed_values = [r["elapsed"] for r in self._records if r.get("elapsed") is not None]
        if elapsed_values:
            avg_elapsed = sum(elapsed_values) / len(elapsed_values)
            self._log_(f"Average elapsed time: {avg_elapsed:.2f}s")

        self.write_json()

    def _load_labels(self, filename):
        """Reads the ground-truth error list for filename from Labels/."""
        path = os.path.join(self.LABELS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _log_missed(self, filename, missed, extra):
        """Appends the filename, missed (FN) ground-truth errors, and added (FP) predicted
        errors that weren't in the labels, to the log file."""
        path = os.path.join(self.LOG_DIR, self._log_file)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{filename}\n")
            if missed:
                for item in missed:
                    f.write(f"  missed - {item.get('type', 'Unknown')}: {item.get('error', '')}\n")
            else:
                f.write("  missed - none\n")
            if extra:
                for item in extra:
                    f.write(f"  added - {item.get('type', 'Unknown')}: {item.get('error', '')}\n")
            else:
                f.write("  added - none\n")

    def _log_(self, info):
        """print and logs info"""
        print(info)
        path = os.path.join(self.LOG_DIR, self._log_file)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n {info}\n")

    @staticmethod
    def _scores(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0

        return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}
