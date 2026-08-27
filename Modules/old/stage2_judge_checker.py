import os
import sys

# This file lives under Modules/old/, two levels below the repo root (where
# stage2_inference_judge.py lives) -- one more dirname() than a file directly
# under Modules/ would need.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import stage2_inference_judge

from Modules.base import CheckerModule, split_sentences


class Stage2JudgeChecker(CheckerModule):
    """Wraps stage2_inference_judge.judge_claim (the paper's Stage-2
    Inference-Aware Judge, see stage2_inference_judge.py's module docstring)
    as a CheckerModule: splits the SOAP note
    into per-sentence claims -- the same split_sentences() convention, with
    the same validated MIN_SENTENCE_WORDS=4 short-fragment filter, every
    other checker here uses (see Modules/base.py) -- and judges EACH claim
    individually against the transcript, one Gemini call per claim. That's
    the paper's actual per-claim methodology (Section 4.3), unlike
    AI_checker.py's single whole-note call (the paper's naive Stage-1 judge,
    which this Stage-2 protocol is a direct response to).

    Only ever returns "hallucination"-type errors. The per-claim protocol
    judges claims that exist in the note against the transcript -- it has no
    way to notice something the note doesn't mention at all, so it
    structurally cannot detect omissions. Evaluate.results() will correctly
    show 0 recall on omission-type labels for this checker; that's expected,
    not a bug, and worth calling out in any report built from these results.

    Each HALLUCINATED verdict is returned as a (type, severity, detail_type,
    detail) 4-tuple -- the same shape, and the same Modules/risk_taxonomy.py
    vocabulary, Modules/high_risk_checker.py already uses and
    Evaluate.compare() already knows how to score (severity/detail_type
    recall, not just TP/FP/FN on the bare claim text). The prompt in
    stage2_inference_judge.py asks the model for these fields directly; "detail" here is
    still the ORIGINAL claim string this checker passed in, not whatever the
    model echoed back -- matching against ground truth needs it exact, and
    trusting a model's copy-back for that is a needless risk when the exact
    string is already sitting right here.
    """

    def __init__(self):
        # Per-claim tier/verdict/reason detail from the most recent check()
        # call -- the CheckerModule interface only returns flagged errors
        # plus a total elapsed time (to stay compatible with Evaluate/run_checker_modules.py
        # and every other checker here), but a batch report wants the full
        # per-claim breakdown (tier distribution, SUPPORTED claims too, not
        # just HALLUCINATED ones) too. Simpler than inventing a second return
        # shape just for this one checker; callers that want it (e.g.
        # batch_runs/stage2_judge_batch.py) read it right after calling check().
        self.last_claims = []

    def check(self, transcript, soap_note):
        claims = split_sentences(soap_note)
        errors = []
        total_elapsed = 0.0
        self.last_claims = []
        for claim in claims:
            result = stage2_inference_judge.judge_claim(transcript, claim)
            total_elapsed += result.get("elapsed", 0.0)
            self.last_claims.append({
                "claim": claim,
                "tier": result.get("tier"),
                "verdict": result.get("verdict"),
                "reason": result.get("reason"),
                "severity": result.get("severity"),
                "detail_type": result.get("detail_type"),
                "model_detail": result.get("detail"),
                "elapsed": result.get("elapsed"),
            })
            if result.get("verdict") == "HALLUCINATED":
                errors.append(("hallucination", result.get("severity"), result.get("detail_type"), claim))
        return tuple(errors), total_elapsed
