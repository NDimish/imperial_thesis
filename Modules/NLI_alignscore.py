"""NLIAlignScoreChecker -- direct MedNLI-tuned classifier alternative to
AlignScoreChecker2, swapping AlignScore's own RoBERTa-based unified
alignment model for a plain HuggingFace NLI classifier fine-tuned on the
clinical-domain MedNLI dataset (Romanov & Shivade, EMNLP 2018, built from
MIMIC-III) -- pritamdeka/PubMedBERT-MNLI-MedNLI by default.

Why this exists: AlignScore's own checkpoint is trained entirely on
general-domain data (4.7M examples from NLI/QA/paraphrase/fact-verification/
information-retrieval/semantic-textual-similarity/summarization -- confirmed
directly against the AlignScore paper and github.com/yuh-zha/AlignScore, no
MedNLI or any clinical-domain data anywhere in that mix). This checker tests
whether a model that HAS actually seen clinical-domain NLI data does better
at hallucination detection on real clinical transcripts than AlignScore's
general-domain checkpoint does.

Reuses this project's own already-built claim/sentence machinery rather
than re-deriving it:
  - split_full_stop_sentences (Modules/old/alignscore_checker.py) splits
    the SOAP note into per-sentence claims.
  - _resolve_qa_pairs (Modules/alignscorechecker2.py) bridges doctor-
    question/patient-reply pairs into synthetic assertion sentences before
    scoring -- see that module's docstring for the false-positive pattern
    this fixes ("any arm weakness?" / "No." never appearing as a
    declarative sentence for an NLI model to match against).

Architecture difference from AlignScore, and why: AlignScore's own
contribution is a chunking/aggregation mechanism that lets it score a claim
against an arbitrarily long context in one call (splits context into
~350-token chunks, aggregates chunk-level scores). A plain HuggingFace NLI
checkpoint like this one has no such mechanism -- it's a short premise/
hypothesis pair classifier, the same shape MedNLI itself was built as (one
sentence vs one sentence). Feeding it a whole multi-turn transcript as one
"premise" would badly exceed BERT's usual window and isn't the input shape
it was fine-tuned on. Instead, each SOAP claim is scored (as the NLI
hypothesis) against EVERY individual transcript turn/bridged-assertion
sentence separately (as the premise), and the BEST (highest-entailment)
match is kept -- "is this claim supported by ANY sentence in the
transcript", the same best-supporting-span pattern already used elsewhere
in this project (e.g. Medical condensor/kdbe_check.py's cosine_coverage
nearest-neighbor matching), rather than one long-context match.

THRESHOLD below is a neutral, UNCALIBRATED 0.5 (the natural "entailment
more likely than not" boundary for a 3-way softmax) -- NOT swept against
real labels the way AlignScore's own 0.30 was (see
Modules/old/alignscore_checker.py's threshold-sweep comment). Re-sweep
before treating this as final.
"""
import os
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Modules.alignscorechecker2 import _resolve_qa_pairs
from Modules.base import CheckerModule
from Modules.old.alignscore_checker import split_full_stop_sentences

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Medical condensor"))
from base import split_turns  # noqa: E402 -- path insert must run first

MODEL_NAME = "pritamdeka/PubMedBERT-MNLI-MedNLI"
THRESHOLD = 0.5
BATCH_SIZE = 16
# PubMedBERT/BERT-family default window is 512 tokens; premise and
# hypothesis here are both always short (one transcript turn, one SOAP
# sentence), so 256 leaves real headroom without wasting compute padding
# every batch out to a length nothing here needs.
MAX_LENGTH = 256


class NLIAlignScoreChecker(CheckerModule):
    """AlignScore-shaped checker (same claim-splitting + Q&A bridging as
    AlignScoreChecker2) backed by a MedNLI-fine-tuned classifier instead of
    AlignScore's own general-domain checkpoint. Hallucination direction
    only, same scope as AlignScoreChecker/AlignScoreChecker2 -- this
    checker has no omission-direction mode."""

    def __init__(self, model_name=None, threshold=None, device="cpu"):
        model_name = model_name or MODEL_NAME
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        # num_labels=3 passed explicitly even though the checkpoint's own
        # config already specifies 3 -- matches the model card's own
        # documented loading example exactly, kept for fidelity with it.
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        self._device = device
        self._model.to(device)
        self._model.eval()

        # Reads the entailment class index from the model's own config
        # rather than hardcoding it -- confirmed directly for the default
        # checkpoint (id2label: {0: "contradiction", 1: "entailment",
        # 2: "neutral"}), but resolving it at runtime keeps this correct if
        # model_name is ever swapped for a checkpoint with a different
        # label order.
        id2label = self._model.config.id2label
        entailment_indices = [i for i, label in id2label.items() if str(label).lower() == "entailment"]
        if not entailment_indices:
            raise RuntimeError(
                f"{model_name}'s config has no 'entailment' label in id2label ({id2label}) -- "
                "can't score support without knowing which output index is entailment."
            )
        self._entailment_idx = entailment_indices[0]

        self._threshold = THRESHOLD if threshold is None else threshold

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        candidates = self._transcript_candidates(transcript)
        claims = split_full_stop_sentences(soap_note)
        errors = self._flag(claims, candidates, error_type="hallucination")

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed

    def _transcript_candidates(self, transcript):
        """The set of short premise candidates a claim is scored against --
        every non-empty turn/line, plus the Q&A-bridged synthetic
        assertions (see module docstring) appended the same additive way
        AlignScoreChecker2 appends them to its context (never replacing
        real transcript text, only adding to the candidate pool)."""
        turns = [line_text.strip() for _speaker, line_text in split_turns(transcript) if line_text.strip()]
        bridged = _resolve_qa_pairs(transcript)
        return turns + bridged

    def _flag(self, claims, candidates, error_type):
        if not claims or not candidates:
            return []

        errors = []
        for claim in claims:
            best_score = self._best_entailment_score(claim, candidates)
            if best_score < self._threshold:
                errors.append((error_type, claim))
        return errors

    def _best_entailment_score(self, claim, candidates):
        """Scores `claim` (as the NLI hypothesis) against every candidate
        premise sentence, batched, and returns the single highest
        entailment probability found -- see module docstring for why "best
        matching sentence" replaces AlignScore's own long-context
        chunking here."""
        best = 0.0
        for i in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[i : i + BATCH_SIZE]
            encoded = self._tokenizer(
                batch, [claim] * len(batch),
                truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, self._entailment_idx]
            batch_best = float(probs.max().item())
            if batch_best > best:
                best = batch_best
        return best
