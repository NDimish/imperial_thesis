import json
import time

from google import genai

from Modules.base import CheckerModule
from secrets_config import GEMINI_API_KEY

RPM = 5  # gemini-3.6-flash requests-per-minute limit
MIN_INTERVAL_SECONDS = 90 / RPM

_last_call_at = 0.0


def _throttle():
    """Sleeps just enough to keep Gemini calls within the RPM limit."""
    global _last_call_at
    elapsed = time.monotonic() - _last_call_at
    if elapsed < MIN_INTERVAL_SECONDS:
        time.sleep(MIN_INTERVAL_SECONDS - elapsed)
    _last_call_at = time.monotonic()


class AIChecker(CheckerModule):
    """Compares a transcript against its SOAP note using Gemini to flag omissions, hallucinations, and LASA errors."""

    PROMPT_TEMPLATE = (
        "You are a clinical documentation auditor performing a safety-critical review, not a casual "
        "proofread. You will be given a medical consultation TRANSCRIPT and a SOAP note written from "
        "that transcript.\n\n"
        "Research on LLM clinical safety (e.g. the MedSafe-Dx benchmark) has found that models often fail "
        "by being falsely reassuring: they sound confident while quietly under-reporting real problems. "
        "For this review, a missed error is a worse outcome than flagging a borderline case that turns out "
        "to be fine — do not default to an empty result out of caution or uncertainty. If something looks "
        "like it could plausibly be an omission, hallucination, or LASA error, include it.\n\n"
        "Before answering, work through the transcript and the SOAP note fact by fact — every symptom, "
        "medication, dosage, vital sign, and plan item — and check whether each one is accurately "
        "reflected in the other document. Then compile the complete list of errors you found.\n\n"
        "Compare the SOAP note against the transcript and identify every:\n"
        "- Omission: clinically relevant information present in the transcript but missing from the SOAP note. "
        "Omissions matter because the SOAP note is the clinical record used for continuity of care, billing, "
        "and legal documentation — a missing symptom, medication, allergy, vital sign, or plan detail can "
        "directly lead to a misdiagnosis or the wrong treatment for the patient.\n"
        "- Hallucination: information present in the SOAP note that is not supported by the transcript.\n"
        "- LASA (Look-Alike/Sound-Alike) error: a drug, dosage, or clinical term in the SOAP note that has been "
        "confused with a similarly named/sounding one from the transcript. LASA errors are a well-documented "
        "cause of real-world medication mistakes (e.g. hydralazine vs. hydroxyzine, Celebrex vs. Celexa, "
        "clonidine vs. clonazepam). The ISMP (Institute for Safe Medication Practices) List of Confused Drug "
        "Names is the standard reference for common LASA pairs — use it as a guide for what counts as a "
        "plausible confusion.\n\n"
        "Respond with ONLY a JSON array of objects, one per error found, each with exactly two keys:\n"
        '  "type": one of "omission", "hallucination", "lasa"\n'
        '  "error": depends on "type":\n'
        '    - "lasa": just the confused term itself, exactly as it appears in the SOAP note — this may be '
        "a single word (e.g. a drug name) or a multi-word phrase (e.g. a dosage, a two-word drug name, or a "
        "clinical term), whatever the confusable unit is\n"
        '    - "omission": the exact sentence from the transcript that is missing from the SOAP note\n'
        '    - "hallucination": the exact sentence from the SOAP note that is not supported by the transcript\n'
        "If there are no errors, respond with an empty JSON array: []\n"
        "Output must be valid JSON and nothing else — no markdown code fences, no commentary, no preamble.\n\n"
        "TRANSCRIPT:\n{transcript}\n\n"
        "SOAP NOTE:\n{soap_note}\n"
    )

    LASA_PROMPT_TEMPLATE = (
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
        "If there are none, respond with an empty JSON array: []\n"
        "Output must be valid JSON and nothing else — no markdown code fences, no commentary, no preamble.\n\n"
        "SOAP NOTE:\n{soap_note}\n"
    )

    MODEL = "gemini-3.6-flash"
    API_KEY = GEMINI_API_KEY

    def __init__(self, api_key=None, model=None):
        self._client = genai.Client(api_key=api_key or self.API_KEY)
        self._model = model or self.MODEL

    def check(self, transcript, soap_note):
        """Runs the prompt against Gemini and returns (((type, error), ...), elapsed)."""
        prompt = self.PROMPT_TEMPLATE.format(transcript=transcript, soap_note=soap_note)

        _throttle()
        start = time.perf_counter()
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        elapsed = time.perf_counter() - start

        errors = self._parse_response(response.text)

        error_pairs = tuple((error.get("type", "Unknown"), error.get("error", "")) for error in errors)

        #lasa_pairs, lasa_elapsed = self.find_lasa_words(soap_note)

        return error_pairs,elapsed#error_pairs + lasa_pairs, elapsed + lasa_elapsed

    def find_lasa_words(self, soap_note):
        """Lists every LASA-candidate term found in a SOAP note (no transcript needed).

        Returns (((type, error), ...), elapsed), same shape as check().
        """
        prompt = self.LASA_PROMPT_TEMPLATE.format(soap_note=soap_note)

        _throttle()
        start = time.perf_counter()
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        elapsed = time.perf_counter() - start

        errors = self._parse_response(response.text)

        error_pairs = tuple((error.get("type", "Unknown"), error.get("error", "")) for error in errors)

        return error_pairs, elapsed

    @staticmethod
    def _parse_response(text):
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return [{"type": "ParseError", "error": f"Could not parse model response: {text[:200]}"}]
