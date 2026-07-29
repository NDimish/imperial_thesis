import argparse
import json
import os
import random
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textflint.generation.transformation.UT.swap_named_ent import SwapNamedEnt
from textflint.input.component.sample import SASample

NOTES_CLEANED_DIR = "prim57/notes cleaned"
BAD_NOTES_DIR = "prim57/bad notes lib"
BAD_NOTES_LABELS_DIR = "prim57/bad notes labels lib"

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

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


def _swap_entity(current_text, similar_notes):
    """Hallucination: swaps one named entity for a same-category entity from a similar note."""
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

    if not pool:
        return None

    source_num, replacement = random.choice(pool)
    new_sample = sample.unequal_replace_field_at_indices("x", [indices[i]], [replacement])
    bad_text = new_sample.get_text("x")

    detail = f'"{original_entity}" was swapped for "{replacement}" (from note {source_num})'
    return bad_text, {"type": "hallucination", "detail": detail}


def _hallucinate_sentence(current_text, similar_notes):
    """Hallucination: splices a whole sentence from a similar note into the target note."""
    if not similar_notes:
        return None

    similar_num = random.choice(list(similar_notes.keys()))
    sentences = _split_sentences(similar_notes[similar_num])
    if not sentences:
        return None
    sentence = random.choice(sentences)

    target_sentences = _split_sentences(current_text)
    insert_at = random.randrange(len(target_sentences) + 1)
    target_sentences.insert(insert_at, sentence)
    bad_text = " ".join(target_sentences)

    detail = f'inserted sentence from note {similar_num}: "{sentence}"'
    return bad_text, {"type": "hallucination", "detail": detail}


def _omit_sentence(current_text):
    """Omission: deletes one sentence from the note (keeps at least one sentence)."""
    sentences = _split_sentences(current_text)
    if len(sentences) <= 1:
        return None

    idx = random.randrange(len(sentences))
    removed = sentences.pop(idx)
    bad_text = " ".join(sentences)

    return bad_text, {"type": "omission", "detail": removed}


def make_bad_note(target_note, similar_notes):
    """Corrupts target_note with 1-4 issues, mixing in content from similar_notes.

    similar_notes: {similar_note_num: text}

    Returns (bad_note_text, issues) where issues is a list of
    {"type": "hallucination"|"omission", "detail": ...} dicts.
    """
    current_text = target_note
    issues = []

    num_issues = random.randint(1, 4)
    actions = [_swap_entity, _hallucinate_sentence, _omit_sentence]

    attempts = 0
    while len(issues) < num_issues and attempts < num_issues * 4:
        attempts += 1
        action = random.choice(actions)

        if action is _omit_sentence:
            result = _omit_sentence(current_text)
        else:
            result = action(current_text, similar_notes)

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
        bad_note, issues = make_bad_note(target_note, similar_notes)
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
