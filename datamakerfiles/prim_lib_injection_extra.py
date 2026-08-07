import argparse
import json
import os
import random
import re
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textflint.generation.transformation.UT.swap_named_ent import SwapNamedEnt
from textflint.input.component.sample import SASample

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "Medical condensor"))
from base import clean_transcript  # noqa: E402 -- path insert must run first
from Modules.medspacy_umls_checker import MedspacyUmlsChecker  # noqa: E402

NOTES_CLEANED_DIR = "prim57/notes cleaned"
TRANSCRIPTS_DIR = "prim57/cleaned transcripts"
BAD_NOTES_DIR = "prim57/bad notes lib extra"
BAD_NOTES_LABELS_DIR = "prim57/bad notes labels lib extra"

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

# See prim_lib_injection.py's identical comment/fix -- _hallucinate_sentence
# below has the same bug (and same fix) as that file's version.
MAX_TRUTH_CHECK_ATTEMPTS = 6

_truth_checker = None
_transcript_concepts_cache = {}


def _get_truth_checker():
    global _truth_checker
    if _truth_checker is None:
        _truth_checker = MedspacyUmlsChecker()
    return _truth_checker


def _target_transcript_concepts(note_num):
    if note_num not in _transcript_concepts_cache:
        path = os.path.join(TRANSCRIPTS_DIR, f"prim{note_num}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                text = clean_transcript(f.read())
            _transcript_concepts_cache[note_num] = _get_truth_checker()._extract_concepts(text)
        else:
            _transcript_concepts_cache[note_num] = {}
    return _transcript_concepts_cache[note_num]


def _is_already_true(sentence, target_num):
    """True if every clinical concept in sentence is already supported (same
    CUI, same assertion polarity) by note target_num's own real transcript.
    See prim_lib_injection.py's version for the full rationale."""
    sentence_concepts = _get_truth_checker()._extract_concepts(sentence)
    if not sentence_concepts:
        return False
    target_concepts = _target_transcript_concepts(target_num)
    for cui, info in sentence_concepts.items():
        t_info = target_concepts.get(cui)
        if t_info is None:
            return False
        if not (set(info["mentions"]) & set(t_info["mentions"])):
            return False
    return True

# Cue -> replacement applied to flip a negated clinical statement into an affirmed
# one (word-boundary, case-insensitive, first match in the sentence only). Chosen to
# read as plausible clinical shorthand after substitution, e.g. "Nil smoking" ->
# " smoking", "denies chest pain" -> "reports chest pain", "no blood in stool" ->
# " blood in stool", "without complications" -> "with complications".
NEGATION_PATTERNS = [
    (re.compile(r"\bdenies\b", re.IGNORECASE), "reports"),
    (re.compile(r"\bdenied\b", re.IGNORECASE), "reported"),
    (re.compile(r"\bwithout\b", re.IGNORECASE), "with"),
    (re.compile(r"\babsent\b", re.IGNORECASE), "present"),
    (re.compile(r"\bnegative for\b", re.IGNORECASE), "positive for"),
    (re.compile(r"\bnil\b", re.IGNORECASE), ""),
    (re.compile(r"\bnot\b", re.IGNORECASE), ""),
    (re.compile(r"\bno\b", re.IGNORECASE), ""),
    (re.compile(r"n't\b", re.IGNORECASE), ""),
]
EXTRA_WHITESPACE_RE = re.compile(r"\s{2,}")

# Matches a bare integer so it can be perturbed regardless of any unit that follows
# it (mg, mcg, ml, mmol, units, /7, x3, etc.) -- units are left untouched, only the
# quantity changes, mirroring real transcription/dosage errors.
NUMBER_RE = re.compile(r"\d+")

_swap_ent = SwapNamedEnt()


def read_notes(count=None):
    """Reads prim57/notes cleaned/prim{N}.txt into {note_num: text}."""
    files = sorted(
        f for f in os.listdir(NOTES_CLEANED_DIR) if f.startswith("prim") and f.endswith(".txt")
    )
    notes = {}
    for filename in files:
        note_num = int(filename[len("prim"):-len(".txt")])
        with open(os.path.join(NOTES_CLEANED_DIR, filename), "r", encoding="utf-8") as f:
            notes[note_num] = f.read()
    if count is not None:
        notes = {num: text for num, text in notes.items() if num <= count}
    return notes


def get_similar_notes(notes):
    """Finds each note's 1-3 most similar notes via TF-IDF cosine similarity (no AI).

    Returns {note_num: [similar_note_num, ...]}.
    """
    note_nums = sorted(notes)
    texts = [notes[num] for num in note_nums]

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    similarity = cosine_similarity(matrix)

    similar = {}
    for i, note_num in enumerate(note_nums):
        scores = sorted(
            ((similarity[i][j], note_nums[j]) for j in range(len(note_nums)) if j != i),
            reverse=True,
        )
        top = [num for _, num in scores[:3]]
        similar[note_num] = top or [note_nums[(i + 1) % len(note_nums)]]

    return similar


def _split_sentences(text):
    return [s for s in SENTENCE_SPLIT_PATTERN.split(text.strip()) if s]


def _entities_in(text):
    """Returns (sample, indices, entities, categories) of named entities in text.

    Uses TextFlint's NER pipeline + entity decomposition (PERSON/LOCATION/ORGANIZATION).
    """
    sample = SASample({"x": text, "y": "neutral"})
    ner_info = sample.get_ner("x")
    indices, entities, categories = _swap_ent.decompose_entities_info(ner_info)
    return sample, indices, entities, categories


def _swap_entity(current_text, similar_notes, target_num=None):
    """Hallucination: swaps one named entity for a same-category entity from a similar note.

    See prim_lib_injection.py's identical fix -- a same-category entity pulled
    from a similar note can coincidentally already be true for the target
    patient (confirmed directly via a full-dataset audit), so candidates are
    filtered through the same truthfulness check _hallucinate_sentence uses
    before one is picked.
    """
    sample, indices, entities, categories = _entities_in(current_text)
    if not indices:
        return None

    i = random.randrange(len(indices))
    category = categories[i]
    original_entity = entities[i]

    pool = []
    for similar_num, similar_text in similar_notes.items():
        _, _, s_entities, s_categories = _entities_in(similar_text)
        for entity, entity_category in zip(s_entities, s_categories):
            if entity_category == category and entity != original_entity:
                pool.append((similar_num, entity))

    if target_num is not None:
        pool = [
            (similar_num, entity) for similar_num, entity in pool
            if not _is_already_true(entity, target_num)
        ]

    if not pool:
        return None

    source_num, replacement = random.choice(pool)
    new_sample = sample.unequal_replace_field_at_indices("x", [indices[i]], [replacement])
    bad_text = new_sample.get_text("x")

    detail = f'"{original_entity}" was swapped for "{replacement}" (from note {source_num})'
    return bad_text, {"type": "hallucination", "detail": detail}


def _hallucinate_sentence(current_text, similar_notes, target_num=None):
    """Hallucination: splices a whole sentence from a similar note into the
    target note -- but only if that sentence isn't already true for this
    patient (see _is_already_true). Retries up to MAX_TRUTH_CHECK_ATTEMPTS
    times before giving up. See prim_lib_injection.py's version for the
    full rationale."""
    if not similar_notes:
        return None

    for _ in range(MAX_TRUTH_CHECK_ATTEMPTS):
        similar_num = random.choice(list(similar_notes.keys()))
        sentences = _split_sentences(similar_notes[similar_num])
        if not sentences:
            continue
        sentence = random.choice(sentences)

        if target_num is not None and _is_already_true(sentence, target_num):
            continue

        target_sentences = _split_sentences(current_text)
        insert_at = random.randrange(len(target_sentences) + 1)
        target_sentences.insert(insert_at, sentence)
        bad_text = " ".join(target_sentences)

        detail = f'inserted sentence from note {similar_num}: "{sentence}"'
        return bad_text, {"type": "hallucination", "detail": detail}

    return None


def _omit_sentence(current_text):
    """Omission: deletes one sentence from the note (keeps at least one sentence)."""
    sentences = _split_sentences(current_text)
    if len(sentences) <= 1:
        return None

    idx = random.randrange(len(sentences))
    removed = sentences.pop(idx)
    bad_text = " ".join(sentences)

    return bad_text, {"type": "omission", "detail": removed}


def _flip_negative(current_text, target_num=None):
    """Hallucination: flips one negated clinical statement to its affirmed opposite,
    e.g. "denies diabetes" -> "reports diabetes", "no blood in stool" -> "blood in
    stool". Picks the first negation cue found across a random sentence order, so
    which sentence/cue gets flipped varies between calls.

    Assumes the negated original was true and the flip therefore isn't -- true for
    most sentences, but a full-dataset audit found real exceptions (e.g. prim55:
    "Vision not disturbed." flipped to "Vision disturbed.", except that patient's
    own transcript genuinely does affirm disturbed vision, so the flip produced a
    statement that isn't actually a hallucination). Same _is_already_true check
    _hallucinate_sentence/_swap_entity use, applied to the flipped sentence before
    it's accepted; rejects and tries the next cue/sentence instead.
    """
    sentences = _split_sentences(current_text)
    if not sentences:
        return None

    order = list(range(len(sentences)))
    random.shuffle(order)

    for idx in order:
        sentence = sentences[idx]
        for pattern, replacement in NEGATION_PATTERNS:
            if pattern.search(sentence):
                new_sentence = pattern.sub(replacement, sentence, count=1)
                new_sentence = EXTRA_WHITESPACE_RE.sub(" ", new_sentence).strip()
                original_sentence = sentence.strip()
                if new_sentence == original_sentence or not new_sentence:
                    continue
                if target_num is not None and _is_already_true(new_sentence, target_num):
                    continue  # flip landed on something already true -- not a real hallucination
                sentences[idx] = new_sentence
                bad_text = " ".join(sentences)
                detail = f'I have negated "{original_sentence}" to "{new_sentence}"'
                return bad_text, {"type": "hallucination", "detail": detail}

    return None


def _perturb_number(num_str):
    """Returns a plausible transcription-error variant of an integer string: a
    factor-of-10 shift (300 -> 30, 30 -> 300) or a single changed digit (300 -> 350),
    picked at random from whichever variants are valid for this number."""
    value = int(num_str)
    variants = []

    if num_str.endswith("0") and value != 0:
        divided = str(value // 10)
        if divided != num_str:
            variants.append(divided)

    multiplied = str(value * 10)
    if multiplied != num_str:
        variants.append(multiplied)

    if len(num_str) >= 2:
        digits = list(num_str)
        pos = random.randrange(len(digits))
        choices = [d for d in "0123456789" if d != digits[pos]]
        digits[pos] = random.choice(choices)
        if not (pos == 0 and digits[0] == "0"):
            changed = "".join(digits)
            if changed != num_str:
                variants.append(changed)

    if not variants:
        return num_str
    return random.choice(variants)


def _swap_number(current_text):
    """Hallucination: perturbs one number in the note (dosage, day count, vitals,
    etc.), e.g. "300 mg" -> "30 mg". Units and surrounding text are left untouched --
    only the digits change.
    """
    sentences = _split_sentences(current_text)
    if not sentences:
        return None

    order = list(range(len(sentences)))
    random.shuffle(order)

    for idx in order:
        sentence = sentences[idx]
        matches = list(NUMBER_RE.finditer(sentence))
        if not matches:
            continue

        random.shuffle(matches)
        for match in matches:
            original = match.group()
            new_number = _perturb_number(original)
            if new_number == original:
                continue

            new_sentence = sentence[:match.start()] + new_number + sentence[match.end():]
            sentences[idx] = new_sentence
            bad_text = " ".join(sentences)
            detail = f'I have edited number "{original}" to "{new_number}" in "{new_sentence.strip()}"'
            return bad_text, {"type": "hallucination", "detail": detail}

    return None


def make_bad_note(target_note, similar_notes, target_num=None):
    """Corrupts target_note with 1-5 issues, mixing in content from similar_notes,
    flipped negations, and perturbed numbers.

    similar_notes: {similar_note_num: text}
    target_num: target_note's own note number, so _hallucinate_sentence,
        _swap_entity, and _flip_negative can check candidate content against
        ITS OWN transcript before accepting it (see _is_already_true).

    Returns (bad_note_text, issues) where issues is a list of
    {"type": "hallucination"|"omission", "detail": ...} dicts.
    """
    current_text = target_note
    issues = []

    num_issues = random.randint(1, 5)
    actions = [_swap_entity, _hallucinate_sentence, _omit_sentence, _flip_negative, _swap_number]
    needs_similar = {_swap_entity, _hallucinate_sentence}
    needs_target_only = {_flip_negative}

    attempts = 0
    while len(issues) < num_issues and attempts < num_issues * 4:
        attempts += 1
        action = random.choice(actions)

        if action in needs_similar:
            result = action(current_text, similar_notes, target_num)
        elif action in needs_target_only:
            result = action(current_text, target_num)
        else:
            result = action(current_text)

        if result is None:
            continue

        current_text, issue = result
        issues.append(issue)

    return current_text, issues


def write_bad_note_and_labels(note_num, bad_note, issues):
    os.makedirs(BAD_NOTES_DIR, exist_ok=True)
    os.makedirs(BAD_NOTES_LABELS_DIR, exist_ok=True)

    with open(os.path.join(BAD_NOTES_DIR, f"prim{note_num}.txt"), "w", encoding="utf-8") as f:
        f.write(bad_note)

    with open(os.path.join(BAD_NOTES_LABELS_DIR, f"prim{note_num}.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(issues, indent=2))


def main(count=None):
    print("Reading cleaned notes...")
    notes = read_notes(count)
    print(f"Loaded {len(notes)} notes.")

    print("Computing note similarity (TF-IDF cosine similarity, no AI)...")
    similar = get_similar_notes(notes)
    print("Got similarity pairings.")

    total = len(notes)
    for i, (note_num, target_note) in enumerate(sorted(notes.items()), start=1):
        similar_nums = similar.get(note_num, [])
        similar_notes = {n: notes[n] for n in similar_nums if n in notes}

        print(f"[{i}/{total}] prim{note_num}.txt: corrupting (similar to {similar_nums})...")
        bad_note, issues = make_bad_note(target_note, similar_notes, note_num)
        write_bad_note_and_labels(note_num, bad_note, issues)

        print(f"[{i}/{total}] prim{note_num}.txt: done — {len(issues)} issue(s) injected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Corrupt prim57 notes with hallucinations/omissions using TF-IDF "
        "similarity, TextFlint's NER-based entity swapping, negation flips, and number "
        "swaps (no AI)."
    )
    parser.add_argument(
        "count", nargs="?", type=int, default=None, help="Process only the first N notes (default: all)"
    )
    args = parser.parse_args()

    print("started Library-Based Bad Note Injection (extra: negation flips + number swaps)")
    main(args.count)
