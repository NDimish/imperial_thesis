import os
import re
import sys

# This file lives under Modules/old/, two levels below the repo root (where
# stage2_inference_judge.py lives) -- one more dirname() than a file directly
# under Modules/ would need.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import stage2_inference_judge

from Modules.base import CheckerModule

# Deliberately NOT Modules.base.split_sentences -- that function's
# MIN_SENTENCE_WORDS=4 filter is tuned for disfluent TRANSCRIPT speech
# ("Yeah.", "Okay.") and was confirmed (10-file run, 2026-08-31) to silently
# drop real corrupted SOAP-note facts before the judge ever saw them: "blood
# in vomit." and "Opening bowels x60/day." are both 3 tokens, both fell
# below the 4-word floor, and neither one appeared ANYWHERE in that run's
# per-claim judgment list -- not judged wrong, never judged at all. SOAP-note
# shorthand is genuinely this terse; 2 still clears blank lines and stray
# single tokens without re-introducing that gap.
_SOAP_CLAIM_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
MIN_SOAP_CLAIM_WORDS = 2


def _split_soap_claims(soap_note):
    return [
        s.strip() for s in _SOAP_CLAIM_SPLIT_RE.split(soap_note.strip())
        if s and len(s.strip().split()) >= MIN_SOAP_CLAIM_WORDS
    ]


class Stage2JudgeChecker(CheckerModule):
    """Wraps stage2_inference_judge's Stage-2 Inference-Aware Judge (see
    that module's docstring) as a CheckerModule, in BOTH directions:

    Hallucination -- judge_claim(): splits the SOAP note into per-claim
    fragments (see _split_soap_claims above) and judges each individually
    against the transcript, one Gemini call per claim -- the paper's actual
    per-claim methodology (Section 4.3).

    Omission -- extract_checkable_facts() + judge_omission(): NOT part of
    the paper (it only covers hallucination) -- extends the same
    criteria-and-protocol approach to the other direction. One call extracts
    every clinically checkable fact from the transcript in SOAP-note
    register (not the transcript's own disfluent phrasing -- see
    stage2_inference_judge.py's module docstring for why register matters
    for matching against ground-truth labels), then one call per extracted
    fact judges whether the note covers it.

    Each verdict is returned as a (type, severity, detail_type, detail)
    4-tuple -- the same shape, and the same Modules/risk_taxonomy.py
    vocabulary, Modules/high_risk_checker.py already uses and
    Evaluate.compare() already knows how to score. "detail" is always the
    ORIGINAL claim/fact string this checker passed in, not whatever the
    model echoed back -- matching against ground truth needs it exact.
    """

    def __init__(self):
        # Per-claim/per-fact tier/verdict/reason detail from the most
        # recent check() call -- the CheckerModule interface only returns
        # flagged errors plus a total elapsed time, but a batch report wants
        # the full breakdown (including COVERED/SUPPORTED items, not just
        # flagged ones) too. Kept as two separate lists (hallucination vs
        # omission) rather than one merged list, since their result dicts
        # have different shapes (tier vs no tier).
        self.last_claims = []
        self.last_facts = []

    def check(self, transcript, soap_note):
        errors = []
        total_elapsed = 0.0

        claims = _split_soap_claims(soap_note)
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

        facts = stage2_inference_judge.extract_checkable_facts(transcript)
        self.last_facts = []
        for fact in facts:
            result = stage2_inference_judge.judge_omission(soap_note, fact)
            total_elapsed += result.get("elapsed", 0.0)
            self.last_facts.append({
                "fact": fact,
                "verdict": result.get("verdict"),
                "reason": result.get("reason"),
                "severity": result.get("severity"),
                "detail_type": result.get("detail_type"),
                "model_detail": result.get("detail"),
                "elapsed": result.get("elapsed"),
            })
            if result.get("verdict") == "OMITTED":
                errors.append(("omission", result.get("severity"), result.get("detail_type"), fact))

        return tuple(errors), total_elapsed
