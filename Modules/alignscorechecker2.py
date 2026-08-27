"""AlignScoreChecker2 -- adds transcript Q&A bridging on top of
Modules/alignscore_checker.py's claim-splitting fix (see that module's
docstring for the fix this builds on, and extra/explainer/
checker_audit_report.html for the original glued-claim failure).

Why this exists: after the splitter fix, a real 20-file run's dominant
remaining false-positive class was a true, correctly-summarized SOAP fact
that's a NEGATIVE (or terse affirmative) doctor-question answer in the
transcript -- "any arm weakness?" / "No." -- and never appears anywhere in
the transcript as an actual declarative sentence for AlignScore's
entailment model to match the SOAP claim against. Confirmed pattern from
that run: "No arm or leg weakness", "no dysphagia, no dysphasia", "No rash
seen on arm", "no Pmhx of CVA/TIA", "Nil smoking, social EtOH" were ALL
true SOAP facts flagged as false-positive hallucinations, because the
model has to infer the Q+A-to-assertion bridge entirely on its own from a
raw, undifferentiated transcript, with no help from any structure.

Fix: before handing the transcript to AlignScore as CONTEXT, resolve every
doctor-question / immediate-patient-reply pair into an explicit synthetic
assertion sentence ("No arm weakness.") and APPEND it to the context --
never replacing the original text, so a wrong resolution can only add a
slightly-off extra sentence, never destroy real content. Same adjacency-
pairing idea as Modules/high_risk_checker.py's Adjacency Pair Context
Propagation (see that module's docstring above its own AFFIRMATIVE_LEXICON),
reused here as a plain TEXT transform instead of a UMLS-concept transform,
since AlignScore just needs supporting text to match against, not a
structured concept. Kept as a local, trimmed copy of that lexicon/logic
rather than importing it -- same reasoning high_risk_checker.py itself
gives for keeping its own denylist copies instead of importing across
checkers (independent dependency chains).

Not yet validated at scale -- this is a first attempt at the fix, tested
on a small file count. Threshold is inherited unchanged from
Modules/old/alignscore_checker.py; re-sweep before treating this as final.
"""

import os
import re
import sys
import time

from alignscore import AlignScore

from Modules.old.alignscore_checker import CKPT_PATH, MODEL, THRESHOLD, split_full_stop_sentences
from Modules.base import CheckerModule

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Medical condensor"))
from base import split_turns  # noqa: E402 -- path insert must run first

# Trimmed local copy of Modules/high_risk_checker.py's AFFIRMATIVE_LEXICON/
# NEGATIVE_LEXICON -- the handful of phrases confirmed there to matter most
# ("no", "not that i know of", "i don't have any", ...), not the full list.
# Strict LEADING-phrase match only (never substring), same reasoning as
# high_risk_checker.py's _classify_reply: "not sure" must not match "no".
AFFIRMATIVE_LEXICON = ["yes", "yeah", "yep", "i do", "i have", "a bit", "definitely", "correct"]
NEGATIVE_LEXICON = [
    "no", "nope", "nah", "never", "none",
    "i haven't", "i've not", "not really", "don't think so",
    "not that i know of", "nothing that i know of", "not aware of any",
    "i don't have", "don't have", "i do not have", "do not have",
    "no known", "not known",
]
FILLER_PREFIX_RE = re.compile(r"^[\s,.\-]*\b(um+|uh+|h?mm+|well|so|okay|ok)\b[\s,.\-]*", re.IGNORECASE)
FILLER_ONLY_RE = re.compile(r"^(um+|uh+|h?mm+|well|so|okay|ok)$", re.IGNORECASE)
CLAUSE_SPLIT_RE = re.compile(r"[.,;]\s+|\band\b", re.IGNORECASE)


def _split_reply_clauses(text):
    raw = [c.strip() for c in CLAUSE_SPLIT_RE.split(text) if c.strip()]
    return [c for c in raw if not FILLER_ONLY_RE.match(c.rstrip(".!?"))]


def _classify_reply(clause):
    """True=affirmative, False=negative, None=ambiguous -- caller skips the
    pair rather than guessing (see _resolve_qa_pairs)."""
    stripped = FILLER_PREFIX_RE.sub("", clause.strip()).strip().lower().rstrip(".!?").strip()
    for phrase in NEGATIVE_LEXICON:
        if stripped == phrase or stripped.startswith(phrase + " ") or stripped.startswith(phrase + ","):
            return False
    for phrase in AFFIRMATIVE_LEXICON:
        if stripped == phrase or stripped.startswith(phrase + " ") or stripped.startswith(phrase + ","):
            return True
    return None


# Strips a question down to its noun-phrase body ("Do you have any arm
# weakness?" -> "arm weakness") using a small set of interrogative openers
# this dataset's doctor turns actually use -- a confirmed-pattern list, not
# a general parser. Only covers "you"/"the patient" as the question's
# subject since every doctor question in this dataset is addressed
# directly to the patient.
QUESTION_STRIP_RE = re.compile(
    r"^(?:so\s+)?(?:do|does|did|have|has|had|is|are|was|were|can|could|will|would)\s+"
    r"(?:you|the patient|they)\s+"
    r"(?:have|had|get|got|notice|noticed|experience|experienced|feel|felt|any)?\s*",
    re.IGNORECASE,
)
LEADING_ANY_RE = re.compile(r"^any\s+", re.IGNORECASE)


def _question_body(question_text):
    """Falls back to the raw question (minus "?") when no opener pattern
    matches -- even an unstripped question still carries useful lexical
    overlap for AlignScore's own scoring, better than dropping the pair
    entirely rather than guess at a rewrite that might be wrong."""
    body = question_text.strip().rstrip("?").strip()
    stripped = QUESTION_STRIP_RE.sub("", body)
    stripped = LEADING_ANY_RE.sub("", stripped)
    stripped = stripped.strip()
    return stripped or body


def _resolve_qa_pairs(text):
    """Returns synthetic assertion sentences bridging each doctor-question /
    immediate-patient-reply pair in text into an explicit declarative
    statement -- see module docstring. Purely additive: callers append
    these to the original transcript text, never replace it.

    Only the FIRST reply clause is used per question (unlike
    high_risk_checker.py's fuller positional multi-clause pairing) -- this
    is a simpler, single-pair-at-a-time version since the goal here is
    just to give AlignScore more supporting text, not to build a complete
    concept-level record; a question with an ambiguous or multi-part reply
    is skipped rather than guessed at.
    """
    assertions = []
    pending_question = None

    for speaker, line_text in split_turns(text):
        if not line_text.strip():
            continue
        speaker_norm = (speaker or "").lower()
        is_doctor = speaker_norm == "d"
        is_patient = speaker_norm == "p"

        if is_doctor and line_text.strip().endswith("?"):
            pending_question = line_text.strip()
            continue

        if is_patient and pending_question:
            clauses = _split_reply_clauses(line_text)
            reply = _classify_reply(clauses[0]) if clauses else None
            body = _question_body(pending_question)
            if reply is False and body:
                assertions.append(f"No {body}.")
            elif reply is True and body:
                assertions.append(f"{body[0].upper()}{body[1:]}.")
            pending_question = None
            continue

        if is_doctor:
            pending_question = None

    return assertions


class AlignScoreChecker2(CheckerModule):
    """AlignScoreChecker with Q&A bridging: same claim splitter as
    Modules/alignscore_checker.py, but the transcript passed to AlignScore
    as CONTEXT has synthetic bridged assertions appended for every
    doctor-question/patient-reply pair (see module docstring). Hallucination
    direction only, same scope as Modules/alignscore_checker.py.
    """

    def __init__(self, ckpt_path=None, model=None, threshold=None, device="cpu"):
        ckpt_path = ckpt_path or CKPT_PATH
        if not ckpt_path:
            raise RuntimeError(
                "AlignScore is not configured. Set CKPT_PATH in alignscore_checker.py "
                "to a downloaded AlignScore checkpoint (see yuh-zha/AlignScore)."
            )
        self._scorer = AlignScore(
            model=model or MODEL,
            batch_size=32,
            device=device,
            ckpt_path=ckpt_path,
            evaluation_mode="nli_sp",
        )
        self._threshold = THRESHOLD if threshold is None else threshold

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        bridged = _resolve_qa_pairs(transcript)
        context = transcript if not bridged else transcript + "\n" + "\n".join(bridged)

        claims = split_full_stop_sentences(soap_note)
        errors = self._flag(claims=claims, context=context, error_type="hallucination")

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed

    def _flag(self, claims, context, error_type):
        if not claims:
            return []
        scores = self._scorer.score(contexts=[context] * len(claims), claims=claims)
        return [(error_type, sentence) for sentence, score in zip(claims, scores) if score < self._threshold]
