import argparse
import json
import os

from google import genai

from secrets_config import GEMINI_API_KEY

TRANSCRIPT_DIR = "Clean Transcripts"
INPUT_DIR = "Input"
OG_SOAP_DIR = "OG_SOAP"
OUTPUT_DIR = "Output"
LABELS_DIR = "Labels"

MODEL = "gemini-3.5-flash-lite"
API_KEY = GEMINI_API_KEY

SOAP_PROMPT_TEMPLATE = (
    "You are a clinical scribe. Read the following medical consultation TRANSCRIPT "
    "and write an accurate SOAP note (Subjective, Objective, Assessment, Plan) that "
    "reflects only what is said in the transcript. Do not add information that isn't "
    "supported by the transcript.\n\n"
    "TRANSCRIPT:\n{transcript}\n\n"
    "Respond with ONLY the SOAP note text."
)

OMISSION_DESCRIPTION = (
    "delete a sentence or piece of clinically relevant information that is already "
    "present in the ORIGINAL SOAP NOTE"
)
HALLUCINATION_DESCRIPTION = "add information to the SOAP note that is not supported by the transcript"
LASA_DESCRIPTION = (
    "swap a drug, dosage, or clinical term for a similarly named/sounding one"
)

ERROR_INJECTION_PROMPT_TEMPLATE = (
    "You are generating synthetic test data for evaluating a clinical documentation "
    "checker. You will be given a TRANSCRIPT and an accurate SOAP note written from it.\n\n"
    "Produce a corrupted version of the SOAP note by introducing errors as follows:\n"
    "{error_requirements}\n\n"
    "Respond with ONLY a JSON object with exactly two keys:\n"
    '  "corrupted_soap": the full corrupted SOAP note text\n'
    '  "errors": a JSON array of objects, one per error introduced, each with keys "type" '
    '(one of "omission", "hallucination", "lasa") and "error", where "error" depends on "type":\n'
    '    - "lasa": just the confused term used in the corrupted SOAP note, exactly as it appears there — '
    "this may be a single word or a multi-word phrase, whatever the confusable unit is\n"
    '    - "omission": the exact sentence removed from the ORIGINAL SOAP NOTE\n'
    '    - "hallucination": the exact sentence added to the corrupted SOAP note that is not supported by the transcript\n\n'
    "TRANSCRIPT:\n{transcript}\n\n"
    "ORIGINAL SOAP NOTE:\n{soap_note}\n"
)

LASA_SCAN_PROMPT_TEMPLATE = (
    "You are a clinical documentation auditor. You will be given a SOAP note.\n\n"
    "Identify every drug name, dosage, or clinical term in the SOAP note that is a known "
    "LASA (Look-Alike/Sound-Alike) term — i.e. a term commonly confused with a similarly "
    "named/sounding drug or clinical term, as catalogued in resources like the ISMP List of "
    "Confused Drug Names. List every such term you find, regardless of whether it appears "
    "correct in this note.\n\n"
    "Respond with ONLY a JSON array of objects, one per term found, each with exactly two keys:\n"
    '  "type": "lasa"\n'
    '  "error": the term itself, exactly as it appears in the SOAP note — this may be a single word '
    "or a multi-word phrase, whatever the confusable unit is\n"
    "If there are none, respond with an empty JSON array: []\n\n"
    "SOAP NOTE:\n{soap_note}\n"
)


def _clean_json_text(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return cleaned


def generate_soap_note(client, transcript):
    """Writes an accurate SOAP note from a transcript."""
    prompt = SOAP_PROMPT_TEMPLATE.format(transcript=transcript)
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text.strip()


def _build_error_requirements(min_omissions, min_hallucinations, min_lasa):
    """Builds the error-requirements section of the injection prompt.

    If any minimum is set, requires at least that many of each specified type.
    Otherwise falls back to a random number (1 to 4) of errors from any category.
    """
    requirements = []
    if min_omissions:
        requirements.append(f"- At least {min_omissions} omission error(s): {OMISSION_DESCRIPTION}.")
    if min_hallucinations:
        requirements.append(f"- At least {min_hallucinations} hallucination error(s): {HALLUCINATION_DESCRIPTION}.")
    if min_lasa:
        requirements.append(f"- At least {min_lasa} LASA error(s): {LASA_DESCRIPTION}.")

    if requirements:
        requirements.append("You may introduce additional errors from any of these categories beyond the minimums above.")
        return "\n".join(requirements)

    return (
        "Introduce a random number (1 to 4) of realistic errors, drawn from these categories:\n"
        f"- Omission: {OMISSION_DESCRIPTION}.\n"
        f"- Hallucination: {HALLUCINATION_DESCRIPTION}.\n"
        f"- LASA (Look-Alike/Sound-Alike): {LASA_DESCRIPTION}."
    )


def _expand_lasa_errors(errors):
    """Every intentional LASA swap is also a hallucination: the confused word was never said."""
    expanded = list(errors)
    for item in errors:
        if str(item.get("type", "")).strip().lower() == "lasa":
            expanded.append({"type": "hallucination", "error": item.get("error", "")})
    return expanded


def find_lasa_words(client, soap_note):
    """Lists every LASA-candidate term found in a SOAP note (correct or not)."""
    prompt = LASA_SCAN_PROMPT_TEMPLATE.format(soap_note=soap_note)
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return json.loads(_clean_json_text(response.text))


def _merge_lasa_words(errors, lasa_words):
    """Appends any LASA words not already present in errors (case-insensitive, by type+word)."""
    seen = {
        (str(item.get("type", "")).strip().lower(), str(item.get("error", "")).strip().lower())
        for item in errors
    }
    merged = list(errors)
    for item in lasa_words:
        key = (str(item.get("type", "")).strip().lower(), str(item.get("error", "")).strip().lower())
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def inject_errors(client, transcript, soap_note, min_omissions=0, min_hallucinations=0, min_lasa=0):
    """Corrupts a SOAP note with omissions/hallucinations/LASA errors.

    min_omissions, min_hallucinations, min_lasa: minimum number of each error
    type to inject. If all are 0, a random number of errors from any category
    are introduced instead.

    Returns (corrupted_soap_text, errors) where errors is a list of
    {"type": ..., "error": ...} dicts describing what was injected. Every LASA
    error is also duplicated as a hallucination error, since the confused word
    is unsupported by the transcript.
    """
    error_requirements = _build_error_requirements(min_omissions, min_hallucinations, min_lasa)
    prompt = ERROR_INJECTION_PROMPT_TEMPLATE.format(
        error_requirements=error_requirements, transcript=transcript, soap_note=soap_note
    )
    response = client.models.generate_content(model=MODEL, contents=prompt)
    data = json.loads(_clean_json_text(response.text))
    return data["corrupted_soap"], _expand_lasa_errors(data["errors"])


def write_file(directory, filename, text):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main(count, min_omissions=0, min_hallucinations=0, min_lasa=0):
    client = genai.Client(api_key=API_KEY)

    transcript_files = sorted(os.listdir(TRANSCRIPT_DIR))[:count]

    for filename in transcript_files:
        path = os.path.join(TRANSCRIPT_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            transcript = f.read()

        soap_note = generate_soap_note(client, transcript)
        corrupted_soap, errors = inject_errors(
            client, transcript, soap_note, min_omissions, min_hallucinations, min_lasa
        )

        lasa_words = find_lasa_words(client, corrupted_soap)
        errors = _merge_lasa_words(errors, lasa_words)

        write_file(INPUT_DIR, filename, transcript)
        write_file(OG_SOAP_DIR, filename, soap_note)
        write_file(OUTPUT_DIR, filename, corrupted_soap)
        write_file(LABELS_DIR, filename, json.dumps(errors, indent=2))

        print(f"{filename}: injected {len(errors)} error(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic Input/OG_SOAP/Output/Labels data from clean transcripts."
    )
    parser.add_argument("x", type=int, help="Number of transcript files to process")
    parser.add_argument(
        "--omissions", type=int, default=0, help="Minimum number of omission errors to inject per file"
    )
    parser.add_argument(
        "--hallucinations", type=int, default=0, help="Minimum number of hallucination errors to inject per file"
    )
    parser.add_argument("--lasa", type=int, default=0, help="Minimum number of LASA errors to inject per file")
    args = parser.parse_args()

    print("started Soap Maker")
    main(args.x, args.omissions, args.hallucinations, args.lasa)
