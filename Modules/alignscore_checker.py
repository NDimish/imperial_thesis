import re
import time

from alignscore import AlignScore

from Modules.base import MIN_SENTENCE_WORDS, CheckerModule

# Set to a downloaded AlignScore checkpoint (see yuh-zha/AlignScore -- checkpoints
# are on HuggingFace, not bundled with the package).
CKPT_PATH = r"C:\Users\natha\OneDrive\Documents\Uni\Impreial\modules\Thesis\code\data\AlignScore-base.ckpt"
MODEL = "roberta-base"

# 0.25, not the earlier 0.02: found in a test.py session comparing AlignScore
# against 6 hand-picked candidates on prim1 (3 genuinely true -- "watery
# diarrhea"=0.863, "asthma"=0.581, "abdominal pain" (a paraphrase of "pain in
# my lower abdomen")=0.302 -- and 3 false/unrelated -- "chest pain"=0.211,
# "broken arm"=0.077, "gallavanting horses"=0.077). 0.25 sits cleanly in the
# gap between the lowest true score (0.302) and the highest false score
# (0.211). Still only 6 probes on 1 file, same caveat as every other
# threshold in this project -- re-sweep across more files/real SOAP
# sentences (not hand-picked probes) before treating this as final. The
# earlier 0.02 was calibrated on a different task (whole SOAP-note sentences
# vs whole transcript, 5-file sweep against real corruption-pipeline
# labels), not on short candidate phrases like these -- the two aren't
# directly comparable.
THRESHOLD = 0.25

# Only a "." (literal full stop) counts as a sentence boundary here --
# deliberately narrower than Modules.base.split_sentences (which also
# accepts !/?/newline boundaries, see that module's own docstring for why).
# Dropping anything that isn't a complete, full-stop-terminated statement
# means questions, terse turn fragments ("d: Okay"), and mid-thought line
# breaks never become candidate claims at all.
FULL_STOP_SENTENCE_RE = re.compile(r"[^.]*\.")


def split_full_stop_sentences(text):
    """Splits text into sentences that literally end with a period, dropping
    fragments under MIN_SENTENCE_WORDS words (same validated fix as
    Modules.base.split_sentences -- see its docstring for why short
    fragments are guaranteed false positives)."""
    sentences = []
    for match in FULL_STOP_SENTENCE_RE.finditer(text):
        sentence = match.group().strip()
        if sentence and len(sentence.split()) >= MIN_SENTENCE_WORDS:
            sentences.append(sentence)
    return sentences


class AlignScoreChecker(CheckerModule):
    """Flags hallucinated SOAP sentences using AlignScore, scored one
    full-stop-terminated sentence at a time against the whole transcript.

    Hallucination direction only for now (SOAP sentence = claim, transcript =
    context) -- the omission direction (transcript sentence vs SOAP note) is
    not run here.

    Not usable until CKPT_PATH points at a downloaded AlignScore checkpoint --
    raises immediately on construction if it isn't configured.
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

        claims = split_full_stop_sentences(soap_note)
        errors = self._flag(claims=claims, context=transcript, error_type="hallucination")

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed

    def _flag(self, claims, context, error_type):
        if not claims:
            return []
        scores = self._scorer.score(contexts=[context] * len(claims), claims=claims)
        return [(error_type, sentence) for sentence, score in zip(claims, scores) if score < self._threshold]
