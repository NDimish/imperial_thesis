import argparse
import json
import os
import sys
import time

from google import genai

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from secrets_config import GEMINI_API_KEY

NOTES_CLEANED_DIR = "prim57/notes cleaned"
BAD_NOTES_DIR = "prim57/bad notes"
BAD_NOTES_LABELS_DIR = "prim57/bad notes labels"

MODEL = "gemini-3.5-flash-lite"
RPM = 15  # gemini-3.5-flash-lite requests-per-minute limit
MIN_INTERVAL_SECONDS = 90 / RPM

_last_call_at = 0.0


def _throttle():
    """Sleeps just enough to keep calls to MODEL within its RPM limit."""
    global _last_call_at
    elapsed = time.monotonic() - _last_call_at
    if elapsed < MIN_INTERVAL_SECONDS:
        time.sleep(MIN_INTERVAL_SECONDS - elapsed)
    _last_call_at = time.monotonic()

SIMILARITY_PROMPT_TEMPLATE = (
    "You are comparing {count} clinical consultation notes to find which ones are clinically "
    "similar to each other (similar presenting complaint, diagnosis, symptoms, or patient "
    "demographics).\n\n"
    "For each note, identify the notes most similar to it — at least 1 and at most 3, ranked by "
    "similarity, excluding the note itself.\n\n"
    "Respond with ONLY a JSON object mapping each note number (as a string) to a JSON array of "
    'the most similar note numbers (integers), e.g. {{"1": [2, 3], "2": [1]}}. Include an entry '
    "for every note number.\n\n"
    "{notes}\n"
)

INJECTION_PROMPT_TEMPLATE = (
    "You are generating synthetic corrupted clinical notes for testing an error-detection system.\n\n"
    "Below is a TARGET NOTE and one or more SIMILAR NOTES from different patients.\n\n"
    "Produce a corrupted version of the TARGET NOTE by introducing between 1 and 4 total issues, "
    "drawn from:\n"
    "- Hallucination: take a sentence, name, age, or medical detail (drug, diagnosis, symptom) "
    "from one of the SIMILAR NOTES and swap it into the TARGET NOTE as if it belonged to this "
    "patient. The inserted content must be lifted from a SIMILAR NOTE, not invented from nothing.\n"
    "- Omission: delete a sentence or piece of clinically relevant information that is already "
    "present in the TARGET NOTE.\n\n"
    "Respond with ONLY a JSON object with exactly two keys:\n"
    '  "bad_note": the full corrupted note text\n'
    '  "issues": a JSON array of objects, one per issue introduced, each with keys "type" (one of '
    '"hallucination", "omission") and "detail", where "detail" depends on "type":\n'
    '    - "hallucination": the sentence/detail swapped into the TARGET NOTE, noting which similar '
    "note it came from\n"
    '    - "omission": the exact sentence removed from the TARGET NOTE\n\n'
    "TARGET NOTE (note {note_num}):\n{target_note}\n\n"
    "SIMILAR NOTES:\n{similar_notes}\n"
)


def _clean_json_text(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return cleaned


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


def get_similar_notes(client, notes):
    """Asks Gemini which notes are most similar to each other.

    Returns {note_num: [similar_note_num, ...]}, 1-3 entries per note.
    """
    joined = "\n\n".join(f"NOTE {num}:\n{text}" for num, text in sorted(notes.items()))
    prompt = SIMILARITY_PROMPT_TEMPLATE.format(count=len(notes), notes=joined)

    _throttle()
    response = client.models.generate_content(model=MODEL, contents=prompt)
    data = json.loads(_clean_json_text(response.text), strict=False)

    return {int(num): [int(similar) for similar in similars] for num, similars in data.items()}


def make_bad_note(client, note_num, target_note, similar_notes):
    """Corrupts target_note with 1-4 hallucinations/omissions pulled from similar_notes.

    similar_notes: {similar_note_num: text}

    Returns (bad_note_text, issues) where issues is a list of
    {"type": "hallucination"|"omission", "detail": ...} dicts.
    """
    similar_text = "\n\n".join(f"NOTE {num}:\n{text}" for num, text in similar_notes.items())
    prompt = INJECTION_PROMPT_TEMPLATE.format(
        note_num=note_num, target_note=target_note, similar_notes=similar_text
    )

    _throttle()
    response = client.models.generate_content(model=MODEL, contents=prompt)
    data = json.loads(_clean_json_text(response.text), strict=False)

    return data["bad_note"], data["issues"]


def write_bad_note_and_labels(note_num, bad_note, issues):
    os.makedirs(BAD_NOTES_DIR, exist_ok=True)
    os.makedirs(BAD_NOTES_LABELS_DIR, exist_ok=True)

    with open(os.path.join(BAD_NOTES_DIR, f"prim{note_num}.txt"), "w", encoding="utf-8") as f:
        f.write(bad_note)

    with open(os.path.join(BAD_NOTES_LABELS_DIR, f"prim{note_num}.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(issues, indent=2))


def main(count=None):
    client = genai.Client(api_key=GEMINI_API_KEY)

    print("Reading cleaned notes...")
    notes = read_notes(count)
    print(f"Loaded {len(notes)} notes.")

    print("Asking Gemini for similar-note pairings (1 request)...")
    similar = get_similar_notes(client, notes)
    print("Got similarity pairings.")

    total = len(notes)
    for i, (note_num, target_note) in enumerate(sorted(notes.items()), start=1):
        similar_nums = (similar.get(note_num) or [n for n in notes if n != note_num][:1])[:3]
        similar_notes = {n: notes[n] for n in similar_nums if n in notes and n != note_num}

        print(f"[{i}/{total}] prim{note_num}.txt: corrupting (similar to {similar_nums})...")
        bad_note, issues = make_bad_note(client, note_num, target_note, similar_notes)
        write_bad_note_and_labels(note_num, bad_note, issues)

        print(f"[{i}/{total}] prim{note_num}.txt: done — {len(issues)} issue(s) injected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Corrupt prim57 notes with hallucinations/omissions borrowed from similar notes."
    )
    parser.add_argument(
        "count", nargs="?", type=int, default=None, help="Process only the first N notes (default: all)"
    )
    args = parser.parse_args()

    print("started Bad Note Injection")
    main(args.count)
