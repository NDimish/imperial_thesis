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
from Modules.risk_taxonomy import DRUG_SEMTYPES, classify_severity  # noqa: E402

NOTES_CLEANED_DIR = "prim57/notes cleaned"
TRANSCRIPTS_DIR = "prim57/cleaned transcripts"
BAD_NOTES_DIR = "prim57/bad notes lib"
BAD_NOTES_LABELS_DIR = "prim57/bad notes labels lib"

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

# Cue -> replacement applied to flip a negated clinical statement into an affirmed
# one (word-boundary, case-insensitive, first match in the sentence only). Ported
# from datamakerfiles/prim_lib_injection_extra.py -- folded into the base
# generator too so severity grading has real variation to work with: a flipped
# "no diabetes"/"denies penicillin allergy" is exactly the kind of high-severity
# error Modules/risk_taxonomy.py's rule is built to catch, and the original
# generator's swap/insert/omit actions alone couldn't produce one.
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

# Matches a bare integer so it can be perturbed regardless of any unit that
# follows it (mg, mcg, ml, mmol, units, /7, x3, etc.) -- also ported from
# prim_lib_injection_extra.py, same reasoning as NEGATION_PATTERNS above.
NUMBER_RE = re.compile(r"\d+")

_swap_ent = SwapNamedEnt()


def _is_drug_term(term):
    """True if term itself resolves to a UMLS drug concept (Pharmacologic
    Substance / Antibiotic / Clinical Drug) via the shared QuickUMLS matcher
    -- used to tell a "drug switch" (critical severity) apart from a generic
    "entity swap" (person/location/org name, low severity) for the SAME
    underlying textflint entity-swap mechanism."""
    if not term or not term.strip():
        return False
    matcher = _get_truth_checker()._matcher
    matches = matcher.match(term, best_match=True, ignore_syntax=False)
    return any(m.get("semtypes", set()) & DRUG_SEMTYPES for group in matches for m in group)

# _hallucinate_sentence() below picks sentences from a TF-IDF-similar note
# specifically so injected content reads plausibly for the target patient --
# which turned out to also make it likely to coincidentally restate
# something already TRUE for that patient (shared boilerplate like "no
# fever", "fluids, paracetamol"). Confirmed directly: a full-dataset audit
# of this function's 58 already-injected sentences found 3 where every
# clinical concept was independently verified true by the target's own
# transcript (not a hallucination at all, just mislabeled by provenance
# instead of by truthfulness), and 16 more partially true. _is_already_true
# below reuses MedspacyUmlsChecker's own concept+assertion extraction (see
# Modules/medspacy_umls_checker.py) to check a candidate sentence against
# the target's real transcript before accepting it, so the same tool that
# exposed this labeling gap is what closes it.
MAX_TRUTH_CHECK_ATTEMPTS = 6

_truth_checker = None
_transcript_concepts_cache = {}


def _get_truth_checker():
    global _truth_checker
    if _truth_checker is None:
        _truth_checker = MedspacyUmlsChecker()
    return _truth_checker


def _target_transcript_concepts(note_num):
    """Lazily extracts and caches note_num's own transcript concepts, so
    repeated truthfulness checks against the same target note don't re-run
    UMLS matching + ConText from scratch every time."""
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
    CUI, same assertion polarity) by note target_num's own real transcript --
    i.e. splicing this sentence in wouldn't actually introduce any
    hallucinated content, just restate something the patient's own
    transcript already establishes. A sentence with no extractable clinical
    concepts (plan numbering, formatting fragments) can't be judged this way
    -- treated as foreign/unverifiable and kept, same as before this check
    existed."""
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

    Same false-hallucination risk _hallucinate_sentence was found to have (see
    _is_already_true's docstring): a same-category entity pulled from a similar
    note can coincidentally already be true for the target patient -- confirmed
    directly via a full-dataset audit, e.g. prim44 swapping in "Lung Ca" (from a
    similar note) where prim44's own transcript already establishes lung cancer
    (family history), so the swap wasn't actually a hallucination for that
    patient. Pool candidates are now filtered the same way _hallucinate_sentence
    filters its candidate sentences, before picking one.
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

    # detail is the actual corrupted sentence, not a provenance note about
    # where the swap came from (see classify_severity's docstring / module
    # comment: detail_type carries the corruption category, detail carries
    # only the plain sentence text).
    detail = next((s for s in _split_sentences(bad_text) if replacement in s), bad_text)
    detail_type = "drug switch" if (_is_drug_term(original_entity) or _is_drug_term(replacement)) else "entity swap"
    severity = classify_severity(detail_type, original_entity, replacement, detail)
    return bad_text, {"type": "hallucination", "severity": severity, "detail_type": detail_type, "detail": detail}


def _hallucinate_sentence(current_text, similar_notes, target_num=None):
    """Hallucination: splices a whole sentence from a similar note into the
    target note -- but only if that sentence isn't already true for this
    patient (see _is_already_true). Retries with a different similar-note
    sentence, up to MAX_TRUTH_CHECK_ATTEMPTS times, rather than accepting
    the first pick. target_num=None (e.g. no matching transcript on disk)
    skips the check and falls back to the old accept-first-pick behavior."""
    if not similar_notes:
        return None

    for _ in range(MAX_TRUTH_CHECK_ATTEMPTS):
        similar_num = random.choice(list(similar_notes.keys()))
        sentences = _split_sentences(similar_notes[similar_num])
        if not sentences:
            continue
        sentence = random.choice(sentences)

        if target_num is not None and _is_already_true(sentence, target_num):
            continue  # already true for this patient -- not a real hallucination, try another

        target_sentences = _split_sentences(current_text)
        insert_at = random.randrange(len(target_sentences) + 1)
        target_sentences.insert(insert_at, sentence)
        bad_text = " ".join(target_sentences)

        # detail is just the inserted sentence itself -- no "from note N"
        # provenance text (see _swap_entity's identical comment).
        detail_type = "inserted sentence"
        severity = classify_severity(detail_type, sentence)
        return bad_text, {"type": "hallucination", "severity": severity, "detail_type": detail_type, "detail": sentence}

    return None


def _omit_sentence(current_text):
    """Omission: deletes one sentence from the note (keeps at least one sentence)."""
    sentences = _split_sentences(current_text)
    if len(sentences) <= 1:
        return None

    idx = random.randrange(len(sentences))
    removed = sentences.pop(idx)
    bad_text = " ".join(sentences)

    detail_type = "omitted detail"
    severity = classify_severity(detail_type, removed)
    return bad_text, {"type": "omission", "severity": severity, "detail_type": detail_type, "detail": removed}


def _flip_negative(current_text):
    """Hallucination: flips one negated clinical statement to its affirmed
    opposite, e.g. "denies diabetes" -> "reports diabetes", "no blood in
    stool" -> "blood in stool". Ported from
    datamakerfiles/prim_lib_injection_extra.py -- see the NEGATION_PATTERNS
    comment above for why this is in the base generator now, not just extra.
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
                if new_sentence == sentence.strip() or not new_sentence:
                    continue
                sentences[idx] = new_sentence
                bad_text = " ".join(sentences)
                detail_type = "negation flip"
                severity = classify_severity(detail_type, new_sentence, sentence)
                return bad_text, {"type": "hallucination", "severity": severity, "detail_type": detail_type, "detail": new_sentence}

    return None


def _perturb_number(num_str):
    """Returns a plausible transcription-error variant of an integer string: a
    factor-of-10 shift (300 -> 30, 30 -> 300) or a single changed digit (300 ->
    350), picked at random from whichever variants are valid for this number.
    Ported from datamakerfiles/prim_lib_injection_extra.py."""
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
    """Hallucination: perturbs one number in the note (dosage, day count,
    vitals, etc.), e.g. "300 mg" -> "30 mg". Ported from
    datamakerfiles/prim_lib_injection_extra.py."""
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
            # Skip plan/list-item numbering ("...\n4.") -- the number is the
            # very last token before the sentence's own trailing period, with
            # no clinical unit or context around it. Confirmed directly: real
            # runs were corrupting bureaucratic step numbers ("Simple
            # analgesia\n5." -> "...50.") instead of an actual clinical value,
            # producing an uncatchable, meaningless "number edit" label.
            if sentence[match.end():] == ".":
                continue

            original = match.group()
            new_number = _perturb_number(original)
            if new_number == original:
                continue

            new_sentence = sentence[:match.start()] + new_number + sentence[match.end():]
            sentences[idx] = new_sentence
            bad_text = " ".join(sentences)
            detail_type = "number edit"
            severity = classify_severity(detail_type, new_sentence)
            return bad_text, {"type": "hallucination", "severity": severity, "detail_type": detail_type, "detail": new_sentence.strip()}

    return None


def make_bad_note(target_note, similar_notes, target_num=None):
    """Corrupts target_note with 1-5 issues, mixing in content from
    similar_notes, flipped negations, and perturbed numbers.

    similar_notes: {similar_note_num: text}
    target_num: target_note's own note number, so _hallucinate_sentence and
        _swap_entity can check candidate content against ITS OWN transcript
        (see _is_already_true) before accepting it.

    Returns (bad_note_text, issues) where issues is a list of
    {"type": "hallucination"|"omission", "severity": ..., "detail_type": ...,
    "detail": ...} dicts -- see Modules/risk_taxonomy.py for the
    severity/detail_type vocabulary.
    """
    current_text = target_note
    issues = []
    # Confirmed directly: without dedup, the same negation-flip (or other
    # action) landing on the same sentence twice across separate attempts
    # produced two IDENTICAL label entries -- and since Evaluate only lets
    # one prediction satisfy one not-yet-matched label, the second copy can
    # never be matched even when the checker genuinely caught it, showing up
    # as a false "miss" that was really just an unwinnable duplicate label.
    seen = set()

    num_issues = random.randint(1, 5)
    actions = [_swap_entity, _hallucinate_sentence, _omit_sentence, _flip_negative, _swap_number]
    needs_similar_and_target = {_swap_entity, _hallucinate_sentence}

    attempts = 0
    while len(issues) < num_issues and attempts < num_issues * 6:
        attempts += 1
        action = random.choice(actions)

        if action in needs_similar_and_target:
            result = action(current_text, similar_notes, target_num)
        else:
            result = action(current_text)

        if result is None:
            continue

        new_text, issue = result
        key = (issue.get("type"), issue.get("detail_type"), issue.get("detail"))
        if key in seen:
            continue  # duplicate corruption -- discard and try a different action instead
        seen.add(key)

        current_text = new_text
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
        "similarity + TextFlint's NER-based entity swapping (no AI)."
    )
    parser.add_argument(
        "count", nargs="?", type=int, default=None, help="Process only the first N notes (default: all)"
    )
    args = parser.parse_args()

    print("started Library-Based Bad Note Injection")
    main(args.count)
