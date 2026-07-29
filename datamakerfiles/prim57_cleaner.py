import json
import os
import re

TRANSCRIPTS_DIR = "prim57/transcripts"
NOTES_DIR = "prim57/notes"
CLEANED_TRANSCRIPTS_DIR = "prim57/cleaned transcripts"
NOTES_CLEANED_DIR = "prim57/notes cleaned"

INTERVAL_PATTERN = re.compile(
    r'xmin = ([\d.eE+-]+)\s*\n\s*xmax = [\d.eE+-]+\s*\n\s*text = "(.*)"\s*$',
    re.MULTILINE,
)


def read_transcript(path):
    """Reads a TextGrid file and returns [(xmin, text), ...] for its non-empty intervals."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    turns = []
    for match in INTERVAL_PATTERN.finditer(content):
        xmin = float(match.group(1))
        text = match.group(2).strip()
        if text:
            turns.append((xmin, text))
    return turns


def read_json(path):
    """Reads a consultation note JSON file and returns the parsed dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_transcripts(consultation_ids):
    """Loops through each consultation's doctor/patient TextGrid pair and merges them.

    Turns from both files are combined and sorted by start time (xmin), then
    tagged "d:" or "p:", so the result reads as a single chronological dialogue.

    Returns {consultation_id: combined_text}.
    """
    aggregated = {}
    for consultation_id in consultation_ids:
        doctor_path = os.path.join(TRANSCRIPTS_DIR, f"{consultation_id}_doctor.TextGrid")
        patient_path = os.path.join(TRANSCRIPTS_DIR, f"{consultation_id}_patient.TextGrid")

        doctor_turns = [(xmin, "d", text) for xmin, text in read_transcript(doctor_path)]
        patient_turns = [(xmin, "p", text) for xmin, text in read_transcript(patient_path)]

        turns = sorted(doctor_turns + patient_turns, key=lambda turn: turn[0])

        aggregated[consultation_id] = "\n".join(f"{speaker}: {text}" for _, speaker, text in turns)

    return aggregated


def read_notes(consultation_ids):
    """Loops through each consultation's note JSON and extracts its free-text note.

    Returns {consultation_id: note_text}.
    """
    notes = {}
    for consultation_id in consultation_ids:
        path = os.path.join(NOTES_DIR, f"{consultation_id}.json")
        notes[consultation_id] = read_json(path)["note"]
    return notes


def write_transcript_file(index, text):
    os.makedirs(CLEANED_TRANSCRIPTS_DIR, exist_ok=True)
    path = os.path.join(CLEANED_TRANSCRIPTS_DIR, f"prim{index}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_notes_file(index, text):
    os.makedirs(NOTES_CLEANED_DIR, exist_ok=True)
    path = os.path.join(NOTES_CLEANED_DIR, f"prim{index}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    consultation_ids = sorted(
        f[: -len(".json")] for f in os.listdir(NOTES_DIR) if f.endswith(".json")
    )

    transcripts = aggregate_transcripts(consultation_ids)
    notes = read_notes(consultation_ids)

    for index, consultation_id in enumerate(consultation_ids, start=1):
        write_transcript_file(index, transcripts[consultation_id])
        write_notes_file(index, notes[consultation_id])

    print(f"Cleaned {len(consultation_ids)} consultations into '{CLEANED_TRANSCRIPTS_DIR}' and '{NOTES_CLEANED_DIR}'.")


if __name__ == "__main__":
    main()
