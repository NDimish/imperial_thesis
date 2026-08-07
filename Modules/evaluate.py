import json
import os
import re
from datetime import datetime

from Modules.risk_taxonomy import SEVERITY_RANK

# Hallucination labels come from datamakerfiles/prim_lib_injection.py (and its
# datamakerfiles/prim_lib_injection_extra.py variant) in one of four shapes:
#   - _hallucinate_sentence(): 'inserted sentence from note N: "<sentence>"' -- a whole
#     sentence spliced in from a similar note.
#   - _swap_entity(): '"<original>" was swapped for "<replacement>" (from note N)' -- an
#     NER-tagged entity/term (name, abbreviation, heading) swapped for a same-category one.
#   - _flip_negative() [extra]: 'I have negated "<original>" to "<flipped>"' -- a negated
#     clinical statement flipped to its affirmed opposite (e.g. "denies X" -> "reports X").
#   - _swap_number() [extra]: 'I have edited number "<original>" to "<new>" in "<sentence>"'
#     -- a dosage/day-count/vitals digit perturbed within its sentence.
# Confirmed by inspecting prim57/bad notes labels lib/*.txt directly and cross-referencing
# against the generator functions that wrote them. Omission labels have neither prefix, just
# the plain removed sentence. Stripping this boilerplate before comparison keeps the label
# text down to the same bare clinical content a checker module's predicted "error" string is
# expected to contain -- for a swap or negation flip, both sides are kept (space-joined)
# since a checker might flag either side as the error; for a number edit, the sentence the
# number lives in is kept since that's the actual erroneous text a checker would echo back.
NOTE_PREFIX_RE = re.compile(r"^\s*inserted sentence from note\s+\d+\s*:\s*", re.IGNORECASE)
SWAP_RE = re.compile(
    r'^\s*"(?P<original>.*?)"\s+was swapped for\s+"(?P<replacement>.*?)"\s*\(from note\s+\d+\)\s*$',
    re.IGNORECASE | re.DOTALL,
)
NEGATION_FLIP_RE = re.compile(
    r'^\s*I have negated\s+"(?P<original>.*?)"\s+to\s+"(?P<flipped>.*?)"\s*$',
    re.IGNORECASE | re.DOTALL,
)
NUMBER_EDIT_RE = re.compile(
    r'^\s*I have edited number\s+"(?P<original>.*?)"\s+to\s+"(?P<new>.*?)"\s+in\s+"(?P<sentence>.*?)"\s*$',
    re.IGNORECASE | re.DOTALL,
)
PUNCTUATION_RE = re.compile(r"[^\w\s]")
WHITESPACE_RE = re.compile(r"\s+")

# TEMP: "X was swapped for Y (from note N)" labels are excluded from evaluation entirely
# for now, rather than matched via SWAP_RE above -- remove this filter (in clean_labels)
# once swap-type labels are ready to be scored again.
SKIP_SWAP_LABELS = True


class Evaluate:
    """Compares a checker module's predicted (type, error) pairs against ground-truth
    label files and accumulates precision/recall/F1/accuracy across every compare() call
    -- both overall and broken down per error type (hallucination, omission, ...).

    Usage (matches main.py):
        evaluator = Evaluate(LABELS_DIR, module_name)
        evaluator.compare(errors, filename, elapsed)   # once per file
        evaluator.results()                             # once, after all files
    """

    def __init__(self, labels_dir, module_name):
        self.labels_dir = labels_dir
        self.module_name = module_name

        self.log_dir = os.environ.get("RESULTS_DIR", "Logs")
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"evaluate_{self.module_name}_{timestamp}"
        self.log_path = os.path.join(self.log_dir, f"{stem}.log")
        self.json_path = os.path.join(self.log_dir, f"{stem}.json")

        self.records = []
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.file_count = 0  # rounds -- one compare() call per file
        # Running TP/FP/FN per error type (e.g. "hallucination", "omission"), discovered
        # dynamically from whatever types show up in compare() -- not hardcoded, so a new
        # checker's error type gets its own breakdown automatically.
        self.type_totals = {}

        # Severity/detail_type are label METADATA on a matched error, not a
        # separate thing to detect the way type is -- so these track two
        # different questions rather than mirroring type_totals's TP/FP/FN
        # shape:
        #   1. "Of every label at severity/detail_type X, how many did we
        #      catch at all (any TP, regardless of what severity/detail_type
        #      the checker itself assigned)?" -- self.severity_label_totals /
        #      self.detail_type_label_totals, {value: {"total", "caught"}}.
        #      This is the "are we catching the risky ones" question.
        #   2. "When we DO catch something, how often do we correctly judge
        #      HOW bad it is / what kind it is?" -- self.severity_pairs /
        #      self.detail_type_pairs, [(predicted, label), ...] over every
        #      TP where the checker's prediction carried that metadata (a
        #      4-tuple (type, severity, detail_type, detail) error, not the
        #      2-tuple every other checker in this project returns).
        self.severity_label_totals = {}
        self.detail_type_label_totals = {}
        self.severity_pairs = []
        self.detail_type_pairs = []
        # Same recall question as severity_label_totals, but split by label
        # type too -- {error_type: {severity: {"total", "caught"}}} -- since
        # "are we catching the risky ones" can have a very different answer
        # for hallucination vs omission (a checker can be strong on one and
        # near-blind on the other, which a combined-across-types number
        # hides).
        self.severity_by_type_totals = {}
        # SOAP-section breakdown (Modules/high_risk_checker.py only -- the
        # only checker that reports a 5th "section" field on its errors,
        # via its own deterministic PMH:/DH:/SH:/Imp:/Plan: header scan).
        # {error_type: {section: {"tp", "fp"}}} -- built from PREDICTIONS
        # only (every flagged error, matched or not), since a section is a
        # property of where a prediction landed in the SOAP note, not
        # something a ground-truth label carries on its own. Omissions
        # never get a section (they describe content missing FROM the note,
        # so there's nowhere in it to point to) and always land under the
        # "n/a" bucket.
        self.section_totals = {}
        # Of every TP (caught) label at severity "critical", how many
        # landed in each section -- answers "which sections are we actually
        # catching the worst errors in".
        self.critical_caught_by_section = {}

    # ------------------------------------------------------------------
    # Label loading / cleaning
    # ------------------------------------------------------------------

    def clean_labels(self, filename):
        """Loads the ground-truth label file for filename and returns a list of
        {"type", "detail", "raw_detail"} dicts, with "detail" cleaned for comparison
        (note-injection prefix/swap boilerplate stripped, punctuation stripped, lowercased).

        TEMP: swap-type labels ('"X" was swapped for "Y" (from note N)') are dropped
        entirely here while SKIP_SWAP_LABELS is True -- see the module-level comment.
        """
        path = os.path.join(self.labels_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            raw_labels = json.load(f)

        if SKIP_SWAP_LABELS:
            raw_labels = [
                item for item in raw_labels
                if "was swapped for" not in str(item.get("detail", "")).lower()
            ]

        return [
            {
                "type": str(item.get("type", "")).strip().lower(),
                "detail": self._clean_text(item.get("detail", "")),
                "raw_detail": item.get("detail", ""),
                # None (not "") for labels from before severity/detail_type existed
                # (e.g. prim57/old/bad notes labels lib) -- distinguishes "no
                # severity on this label" from "empty string severity" everywhere
                # below that reads these two fields.
                "severity": (str(item["severity"]).strip().lower() if item.get("severity") else None),
                "detail_type": (str(item["detail_type"]).strip().lower() if item.get("detail_type") else None),
            }
            for item in raw_labels
        ]

    @staticmethod
    def _clean_text(text):
        """Strips generator boilerplate -- the "inserted sentence from note N:" prefix,
        the "X was swapped for Y (from note N)" wrapper (keeping both swapped terms),
        the "I have negated X to Y" wrapper (keeping both negated terms), or the "I have
        edited number X to Y in <sentence>" wrapper (keeping the sentence) -- and all
        punctuation, collapses whitespace, and lowercases. Used on both label details and
        predicted error text so the two sides compare on equal footing."""
        text = str(text)

        swap_match = SWAP_RE.match(text)
        negation_match = NEGATION_FLIP_RE.match(text)
        number_match = NUMBER_EDIT_RE.match(text)
        if swap_match:
            text = f"{swap_match.group('original')} {swap_match.group('replacement')}"
        elif negation_match:
            text = f"{negation_match.group('original')} {negation_match.group('flipped')}"
        elif number_match:
            text = number_match.group("sentence")
        else:
            text = NOTE_PREFIX_RE.sub("", text)

        text = PUNCTUATION_RE.sub(" ", text)
        text = WHITESPACE_RE.sub(" ", text).strip().lower()
        return text

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(self, errors, filename, elapsed=None):
        """Compares predicted (type, error) pairs -- or, from a checker that
        also grades severity/detail_type (e.g. Modules/high_risk_checker.py),
        (type, severity, detail_type, error) 4-tuples -- against the cleaned
        ground-truth labels for filename, accumulates TP/FP/FN into the
        running totals (overall, per error type, and per label severity/
        detail_type), logs the outcome of every prediction and every missed
        label, and returns the per-file record.

        A prediction is a true positive if its type matches a not-yet-matched
        label of the same type and the cleaned error text is a substring of
        the cleaned label detail, or vice versa -- severity/detail_type, when
        present, are metadata on that match, not part of the matching key
        itself (see self.severity_label_totals's comment in __init__ for why).
        """
        labels = self.clean_labels(filename)
        matched = [False] * len(labels)

        flagged = []  # (type, p_severity, p_detail_type, p_section, raw_error, is_true_positive)
        tp = 0
        fp = 0
        file_type_counts = {}  # {error_type: {"tp":, "fp":, "fn":}} for this file only

        def _bump(error_type, key):
            bucket = file_type_counts.setdefault(error_type, {"tp": 0, "fp": 0, "fn": 0})
            bucket[key] += 1

        for error in errors:
            if len(error) == 5:
                error_type, p_severity, p_detail_type, error_text, p_section = error
                p_severity = str(p_severity).strip().lower() if p_severity else None
                p_detail_type = str(p_detail_type).strip().lower() if p_detail_type else None
                p_section = str(p_section).strip().lower() if p_section else "n/a"
            elif len(error) == 4:
                error_type, p_severity, p_detail_type, error_text = error
                p_severity = str(p_severity).strip().lower() if p_severity else None
                p_detail_type = str(p_detail_type).strip().lower() if p_detail_type else None
                p_section = None
            else:
                error_type, error_text = error
                p_severity = None
                p_detail_type = None
                p_section = None

            p_type = str(error_type).strip().lower()
            p_clean = self._clean_text(error_text)

            match_index = None
            for i, label in enumerate(labels):
                if matched[i]:
                    continue
                if p_type != label["type"]:
                    continue
                if p_clean and (p_clean in label["detail"] or label["detail"] in p_clean):
                    match_index = i
                    break

            if p_section is not None:
                bucket = self.section_totals.setdefault(p_type, {}).setdefault(p_section, {"tp": 0, "fp": 0})

            if match_index is not None:
                matched[match_index] = True
                tp += 1
                _bump(p_type, "tp")
                flagged.append((error_type, p_severity, p_detail_type, p_section, error_text, True))
                if p_section is not None:
                    bucket["tp"] += 1

                label = labels[match_index]
                if p_severity and label["severity"]:
                    self.severity_pairs.append((p_severity, label["severity"]))
                if p_detail_type and label["detail_type"]:
                    self.detail_type_pairs.append((p_detail_type, label["detail_type"]))
                if label["severity"] == "critical" and p_section is not None:
                    section_bucket = self.critical_caught_by_section.setdefault(p_section, 0)
                    self.critical_caught_by_section[p_section] = section_bucket + 1
            else:
                fp += 1
                _bump(p_type, "fp")
                flagged.append((error_type, p_severity, p_detail_type, p_section, error_text, False))
                if p_section is not None:
                    bucket["fp"] += 1

        missed = [labels[i] for i, was_matched in enumerate(matched) if not was_matched]
        fn = len(missed)
        for label in missed:
            _bump(label["type"], "fn")

        # Every label's severity/detail_type counts toward "total" here, caught
        # or not -- that's what makes by_label_severity/by_label_detail_type's
        # recall answer "are we catching the risky ones", independent of
        # whether the checker even reports severity/detail_type on its own
        # predictions.
        for i, label in enumerate(labels):
            if label["severity"]:
                bucket = self.severity_label_totals.setdefault(label["severity"], {"total": 0, "caught": 0})
                bucket["total"] += 1
                if matched[i]:
                    bucket["caught"] += 1

                type_buckets = self.severity_by_type_totals.setdefault(label["type"], {})
                type_bucket = type_buckets.setdefault(label["severity"], {"total": 0, "caught": 0})
                type_bucket["total"] += 1
                if matched[i]:
                    type_bucket["caught"] += 1
            if label["detail_type"]:
                bucket = self.detail_type_label_totals.setdefault(label["detail_type"], {"total": 0, "caught": 0})
                bucket["total"] += 1
                if matched[i]:
                    bucket["caught"] += 1

        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.file_count += 1
        for error_type, counts in file_type_counts.items():
            bucket = self.type_totals.setdefault(error_type, {"tp": 0, "fp": 0, "fn": 0})
            bucket["tp"] += counts["tp"]
            bucket["fp"] += counts["fp"]
            bucket["fn"] += counts["fn"]

        record = {
            "filename": filename,
            "elapsed": elapsed,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            **self._scores(tp, fp, fn),
            "by_type": {
                error_type: {**counts, **self._scores(**counts)}
                for error_type, counts in file_type_counts.items()
            },
        }
        self.records.append(record)

        self._log_compare(filename, flagged, missed)

        return record

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def results(self):
        """Writes per-file, per-type, and overall precision/recall/F1/accuracy to the
        log file, and dumps every record plus those scores to a JSON file (same naming
        scheme as the log, in LOG_DIR). Returns {"overall": ..., "by_type": {...}}."""
        overall = self._scores(self.total_tp, self.total_fp, self.total_fn)

        self._log("\n=== Evaluation Results ===")
        for r in self.records:
            self._log(
                f"{r['filename']}: precision={r['precision']:.2f} recall={r['recall']:.2f} "
                f"f1={r['f1']:.2f} accuracy={r['accuracy']:.2f} (TP={r['tp']} FP={r['fp']} FN={r['fn']})"
            )

        self._log(
            f"\nOverall (all types): precision={overall['precision']:.2f} recall={overall['recall']:.2f} "
            f"f1={overall['f1']:.2f} accuracy={overall['accuracy']:.2f} "
            f"(TP={self.total_tp} FP={self.total_fp} FN={self.total_fn})"
        )

        # "Flags" = every prediction the checker raised, matched or not
        # (TP+FP) -- i.e. how much output a human reviewer has to wade
        # through per file/round, independent of whether it was right.
        total_flags = self.total_tp + self.total_fp
        avg_flags_per_round = total_flags / self.file_count if self.file_count else 0.0
        self._log(
            f"\nAverage flags per round: {avg_flags_per_round:.1f} "
            f"({total_flags} flags / {self.file_count} files)"
        )

        by_type_scores = {}
        if self.type_totals:
            self._log("\nBy type:")
            for error_type in sorted(self.type_totals):
                counts = self.type_totals[error_type]
                scores = self._scores(**counts)
                by_type_scores[error_type] = {**counts, **scores}
                self._log(
                    f"  {error_type}: precision={scores['precision']:.2f} recall={scores['recall']:.2f} "
                    f"f1={scores['f1']:.2f} accuracy={scores['accuracy']:.2f} "
                    f"(TP={counts['tp']} FP={counts['fp']} FN={counts['fn']})"
                )

        elapsed_values = [r["elapsed"] for r in self.records if r.get("elapsed") is not None]
        if elapsed_values:
            self._log(f"\nAverage elapsed time: {sum(elapsed_values) / len(elapsed_values):.2f}s")

        severity_scores = self._label_metadata_scores(
            "By severity (recall -- are we catching the risky ones?):",
            self.severity_label_totals, self.severity_pairs, numeric_rank=SEVERITY_RANK,
        )
        detail_type_scores = self._label_metadata_scores(
            "By detail type (recall -- are we catching this corruption category?):",
            self.detail_type_label_totals, self.detail_type_pairs,
        )

        severity_by_type_scores = {}
        if self.severity_by_type_totals:
            self._log("\nBy severity, split by label type:")
            for error_type in sorted(self.severity_by_type_totals):
                self._log(f"  {error_type}:")
                by_value = {}
                for severity in sorted(self.severity_by_type_totals[error_type]):
                    counts = self.severity_by_type_totals[error_type][severity]
                    recall = counts["caught"] / counts["total"] if counts["total"] else 0.0
                    by_value[severity] = {**counts, "recall": recall}
                    self._log(f"    {severity}: {counts['caught']}/{counts['total']} caught (recall={recall:.2f})")
                severity_by_type_scores[error_type] = by_value

        if self.section_totals:
            self._log("\nBy SOAP section (predictions -- omissions have no section, they")
            self._log("describe content missing FROM the note, always bucketed under \"n/a\"):")
            for error_type in sorted(self.section_totals):
                self._log(f"  {error_type}:")
                for section in sorted(self.section_totals[error_type]):
                    counts = self.section_totals[error_type][section]
                    self._log(f"    {section}: TP={counts['tp']} FP={counts['fp']}")

        if self.critical_caught_by_section:
            self._log("\nCritical-severity labels caught (TP), by section:")
            for section in sorted(self.critical_caught_by_section):
                self._log(f"  {section}: {self.critical_caught_by_section[section]}")

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "module": self.module_name,
                    "overall": {
                        "tp": self.total_tp,
                        "fp": self.total_fp,
                        "fn": self.total_fn,
                        **overall,
                    },
                    "by_type": by_type_scores,
                    "by_severity": severity_scores,
                    "by_severity_by_type": severity_by_type_scores,
                    "by_detail_type": detail_type_scores,
                    "by_section": self.section_totals,
                    "critical_caught_by_section": self.critical_caught_by_section,
                    "avg_flags_per_round": avg_flags_per_round,
                    "file_count": self.file_count,
                    "records": self.records,
                },
                f,
                indent=2,
            )

        return {
            "overall": overall,
            "by_type": by_type_scores,
            "avg_flags_per_round": avg_flags_per_round,
            "by_severity": severity_scores,
            "by_severity_by_type": severity_by_type_scores,
            "by_detail_type": detail_type_scores,
            "by_section": self.section_totals,
            "critical_caught_by_section": self.critical_caught_by_section,
        }

    def _label_metadata_scores(self, heading, label_totals, pairs, numeric_rank=None):
        """Shared by severity and detail_type (see __init__'s comment for the
        two questions this answers): logs + returns
        {"by_value": {value: {"total", "caught", "recall"}}, "accuracy":
        (fraction of TPs where the checker's own predicted value matched the
        label's), "mae": mean absolute rank distance (numeric_rank only,
        e.g. SEVERITY_RANK -- omitted for detail_type, which has no natural
        ordering)}."""
        by_value = {}
        if label_totals:
            self._log(f"\n{heading}")
            for value in sorted(label_totals):
                counts = label_totals[value]
                recall = counts["caught"] / counts["total"] if counts["total"] else 0.0
                by_value[value] = {**counts, "recall": recall}
                self._log(f"  {value}: {counts['caught']}/{counts['total']} caught (recall={recall:.2f})")

        accuracy = None
        mae = None
        if pairs:
            correct = sum(1 for predicted, label in pairs if predicted == label)
            accuracy = correct / len(pairs)
            self._log(f"  accuracy on matched errors (predicted value == label value): {accuracy:.2f} ({correct}/{len(pairs)})")
            if numeric_rank:
                diffs = [
                    abs(numeric_rank.get(predicted, 0) - numeric_rank.get(label, 0))
                    for predicted, label in pairs
                ]
                mae = sum(diffs) / len(diffs)
                self._log(f"  mean absolute error (severity rank): {mae:.2f}")

        return {"by_value": by_value, "accuracy": accuracy, "mae": mae}

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_compare(self, filename, flagged, missed):
        """Logs which errors were flagged for filename (TRUE/FALSE against the
        labels), plus every ground-truth label that was missed entirely -- an FN,
        printed the same way as the TRUE/FALSE predictions above it."""
        self._log(f"\n{filename}")
        if flagged:
            for error_type, severity, detail_type, section, error_text, is_true in flagged:
                verdict = "TRUE" if is_true else "FALSE"
                if severity or detail_type:
                    self._log(f"  [{verdict}] {error_type} [{severity or '?'}/{detail_type or '?'}/{section or '?'}]: {error_text}")
                else:
                    self._log(f"  [{verdict}] {error_type}: {error_text}")
        else:
            self._log("  flagged - none")

        if missed:
            for label in missed:
                severity = label["severity"] or "?"
                detail_type = label["detail_type"] or "?"
                self._log(f"  [FN] {label['type']} [{severity}/{detail_type}]: {label['raw_detail']}")
        else:
            self._log("  FN - none")

    def _log(self, message):
        #print(message)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    @staticmethod
    def _scores(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
        return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}
