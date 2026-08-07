import re
from abc import ABC, abstractmethod

TURN_PATTERN = re.compile(r"^(d|p|speaker\s*\d+)\s*:\s*(.*)$", re.IGNORECASE)
# "Speaker 0:"/"Speaker 1:" -- confirmed directly on extra/temp/prim1.txt (a
# newer transcript that doesn't use this project's usual "d:"/"p:" tagging)
# that turns are still one-per-line, just with a different speaker token,
# and that speaker 0/1 map consistently to doctor/patient the same way "d:"
# always opens every other transcript in this dataset: Speaker 0 asks "how
# can I help you this morning?", Speaker 1 reports symptoms. Normalized to
# "d"/"p" here -- the single shared parsing point every checker/condenser
# already imports split_turns from -- rather than duplicating a second
# speaker-role convention through every caller that currently does
# `speaker.lower() == "d"`/`"p"`. A speaker number other than 0/1 (a
# hypothetical third party) is left as its own normalized token rather than
# guessed into doctor or patient, so callers checking specifically for "d"/
# "p" simply treat it as an unrecognized speaker, same as an unmatched line.
_SPEAKER_NUMBER_ROLE = {"0": "d", "1": "p"}


def _normalize_speaker(raw):
    raw = raw.strip().lower()
    if raw in ("d", "p"):
        return raw
    match = re.match(r"speaker\s*(\d+)", raw)
    if match:
        return _SPEAKER_NUMBER_ROLE.get(match.group(1), raw.replace(" ", ""))
    return raw
# <UNSURE>...</UNSURE> wraps text the transcriber wasn't fully confident about --
# strip the tags but keep the enclosed words. <UNIN/> and <INAUDIBLE_SPEECH/>
# both mark a stretch of genuinely unintelligible speech -- self-closing, no
# content to keep. Left unstripped, the literal tag names get fed straight
# into NLP models -- confirmed via direct QuickUMLS inspection that the word
# "UNSURE" itself coincidentally matches a real UMLS concept (T033, Finding)
# at similarity=1.0, which is exactly why greeting/sign-off turns kept
# surviving the clinical-relevance filter no matter how the matching
# thresholds were tuned. <INAUDIBLE_SPEECH/> was found the same way later --
# present in 56 of 57 transcript files and, until this fix, getting tagged as
# a literal "ENTITY" by scispacy's NER (confirmed directly on prim28.txt).
UNSURE_TAG_PATTERN = re.compile(r"<UNSURE>(.*?)</UNSURE>", re.IGNORECASE | re.DOTALL)
UNIN_TAG_PATTERN = re.compile(r"<UNIN\s*/>", re.IGNORECASE)
INAUDIBLE_SPEECH_TAG_PATTERN = re.compile(r"<INAUDIBLE_SPEECH\s*/>", re.IGNORECASE)

# en_core_sci_sm (used by SciSpacyCondenser and NegspacyCondenser) exposes only
# one flat entity label ("ENTITY") -- there's no semantic-type filter available
# the way QuickUMLS has accepted_semtypes. Confirmed via direct inspection that
# it tags pure discourse markers as entities ("Hello", "Yeah", "okay", "Great",
# "I wish") -- it's trained on formal scientific literature, not spoken
# dialogue, and its notion of an entity-shaped span doesn't transfer. Denylist
# these by matched surface form, same fix pattern as QuickUMLS's "start"/"well".
GENERIC_ENTITY_DENYLIST = {
    "hello", "hi", "hey", "yeah", "yep", "yes", "no", "okay", "ok",
    "good morning", "morning", "great", "fine", "well", "sure", "right",
    "i wish", "thanks", "thank you", "bye", "sir",
}


def is_generic_entity(text):
    """True if text is a discourse-marker false positive rather than a real
    clinical mention (see GENERIC_ENTITY_DENYLIST)."""
    return text.strip().lower() in GENERIC_ENTITY_DENYLIST


class CondenserModule(ABC):
    """Base class for all transcript fluff-removal (condenser) modules."""

    @abstractmethod
    def condense(self, transcript):
        """Removes non-clinically-relevant content from a transcript.

        Returns (condensed_transcript, elapsed).
        """
        raise NotImplementedError


def clean_transcript(text):
    """Strips transcription-annotation tags (<UNSURE>...</UNSURE>, <UNIN/>,
    <INAUDIBLE_SPEECH/>) from raw transcript text, keeping the enclosed words
    for <UNSURE> and dropping the two self-closing tags entirely. Collapses
    the extra spacing left behind, without merging across lines (turn
    boundaries)."""
    text = UNSURE_TAG_PATTERN.sub(r"\1", text)
    text = UNIN_TAG_PATTERN.sub("", text)
    text = INAUDIBLE_SPEECH_TAG_PATTERN.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def split_turns(transcript):
    """Splits a d:/p: (or Speaker 0:/Speaker 1:) tagged transcript into a
    list of (speaker, text) tuples, normalized to "d"/"p" either way."""
    turns = []
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        match = TURN_PATTERN.match(line)
        if match:
            turns.append((_normalize_speaker(match.group(1)), match.group(2)))
        else:
            turns.append((None, line))
    return turns


def join_turns(turns):
    """Rejoins (speaker, text) tuples back into d:/p: tagged transcript text."""
    lines = [f"{speaker}: {text}" if speaker else text for speaker, text in turns]
    return "\n".join(lines)
