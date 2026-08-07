"""Mirrors the "Stage 2: Inference-Aware Judge" framework from:

    Augnito Research, "Beyond Literal Summarization: Redefining Hallucination
    for Medical SOAP Note Evaluation" (arXiv:2604.14829, April 2026).

The paper's core finding: a naive "flag anything not explicitly stated in the
transcript" judge (their Stage 1) produces a 35.2% mean hallucination rate,
nearly 3x the human-annotator baseline of 10.4% -- because it can't tell a
genuinely fabricated claim from a paraphrase, a trade/generic drug-name swap,
or a medically reasonable clinical inference. Their Stage 2 judge, given
explicit criteria for what counts as SUPPORTED beyond verbatim restatement,
closes that gap to 9.1% -- within 1.3 points of the human baseline. This
script implements Stage 2's exact criteria (Section 4.3): the Supported Claim
Criteria, the non-negotiable Synonym Rule, the Five-Step Chain of Thought
protocol, and the hard-floor Retained Hallucination Conditions, plus the
paper's five-tier claim classification (Section 3.2, Table 1) as the output
schema -- via a single LLM call per claim, since this project already has a
configured Gemini key/client for exactly this kind of judge call (see
Modules/AI_checker.py).

Note: the paper's Stage 2 also uses a retrieval-augmented knowledge layer
(SNOMED CT / ICD-10 / drug-equivalence lookups fed to the judge as extra
context per claim, Section 4.3 "Clinical Knowledge Retrieval") -- not
reproduced here. This project's shared UMLS/QuickUMLS matcher (Medical
condensor/umls_matching.py) could fill that role directly if this needs
extending later; left out for now to keep this a single-call script matching
what was asked for.
"""
import json
import os
import time

from google import genai

from secrets_config import GEMINI_API_KEY

# --- Global config -- edit these, not the functions below ---
API_KEY = GEMINI_API_KEY  # see secrets_config.py; same key Modules/AI_checker.py uses
MODEL = "gemini-3.6-flash"
INPUT_DIR = "prim57/cleaned transcripts"

# Same RPM throttle as Modules/AI_checker.py -- same API key/quota.
RPM = 5
MIN_INTERVAL_SECONDS = 90 / RPM
_last_call_at = 0.0


def _throttle():
    global _last_call_at
    elapsed = time.monotonic() - _last_call_at
    if elapsed < MIN_INTERVAL_SECONDS:
        time.sleep(MIN_INTERVAL_SECONDS - elapsed)
    _last_call_at = time.monotonic()


# Ported directly from the paper's Section 4.3: Supported Claim Criteria,
# Synonym Rule, Five-Step Chain of Thought, and Retained Hallucination
# Conditions, plus the Section 3.2 / Table 1 five-tier classification as the
# required output schema. The paper's own "Representative prompt excerpt
# (Stage 2)" text is folded in near the top almost verbatim.
PROMPT_TEMPLATE = """You are a clinical documentation expert and medical AI evaluator, applying \
the Inference-Aware Judge framework from Augnito Research's "Beyond Literal \
Summarization: Redefining Hallucination for Medical SOAP Note Evaluation" \
(arXiv:2604.14829).

When assessing a SOAP note claim against the source transcript, apply the \
following criteria: a claim is SUPPORTED if it is (a) directly stated in the \
transcript, (b) a paraphrase or medical terminology equivalent of stated \
information, or (c) a medically reasonable inference that a trained clinician \
would draw from the presented symptom picture. A claim is HALLUCINATED only \
if it (a) introduces information with no basis in the transcript and cannot \
be reasonably inferred, or (b) directly contradicts information stated in \
the transcript.

TRANSCRIPT (the source of facts):
{transcript}

CLAIM (a statement from a generated SOAP note, to classify against the \
transcript above):
{claim}

Supported Claim Criteria -- the claim is SUPPORTED, and must NOT be flagged, \
under ANY of the following conditions:
  - It is a direct statement, verbatim or near-verbatim, from the transcript.
  - It is a paraphrase or terminology upgrade of a stated fact (e.g. "throat \
irritation" reported as "pharyngeal sensitivity").
  - It is a trade name / generic drug name equivalent of a medication \
mentioned in the transcript (or vice versa).
  - It is a medically reasonable inference from symptoms, history, or \
findings -- a diagnosis need NOT be explicitly named by the patient or \
physician in the transcript to be valid.
  - It reflects a plan goal or intention consistent with a described \
treatment action.
  - It is a standard clinical summary statement appropriate given the \
overall picture (e.g. "hemodynamically stable" is supported if vitals \
described in the transcript are normal).

Synonym Rule (non-negotiable): trade names and generic drug names are to be \
treated as equivalent under ALL circumstances. If the transcript contains a \
trade name and the claim uses the generic name, or vice versa, the claim is \
SUPPORTED -- this rule overrides any apparent lack of verbatim match.

Apply this five-step protocol, in order, with HALLUCINATED as the verdict of \
last resort:
  1. State the claim.
  2. Scan the entire transcript for a direct mention, synonym, translated \
equivalent, or paraphrase of the claim.
  3. If found -> SUPPORTED. Stop.
  4. If not found, ask: is this a reasonable clinical inference, a standard \
summary statement, or a terminology upgrade? If yes -> SUPPORTED. Stop.
  5. Otherwise -> HALLUCINATED. State the specific reason.

Retained Hallucination Conditions (hard safety floor -- these are ALWAYS \
HALLUCINATED regardless of the criteria above, no exceptions):
  - A specific medication, dose, frequency, or duration with no equivalent \
anywhere in the transcript.
  - A diagnosis that is neither stated nor inferable from the symptom picture.
  - A procedure or test not mentioned anywhere in the transcript.
  - Any detail that directly contradicts a fact stated in the transcript.

Classify the claim into exactly one tier from this table:
  Tier 1  Direct Statement                        -> SUPPORTED
  Tier 2a Paraphrasing                             -> SUPPORTED
  Tier 2b Trade name / generic name equivalence    -> SUPPORTED
  Tier 3  Medically reasonable inference           -> SUPPORTED
  Tier 4  Speculated overreach (no inferential basis) -> HALLUCINATED
  Tier 5  Contradiction                            -> HALLUCINATED

Respond with ONLY a JSON object, no markdown code fences, no commentary:
{{"tier": "1|2a|2b|3|4|5", "verdict": "SUPPORTED|HALLUCINATED", "reason": \
"<one sentence: cite the specific transcript evidence used, or explain what's \
missing/contradicted>"}}
"""


def judge_claim(transcript, claim, api_key=None, model=None):
    """Runs the Stage-2 Inference-Aware Judge protocol on one claim against
    one transcript via a single Gemini call. Returns a dict with keys
    tier/verdict/reason (or verdict="PARSE_ERROR" if the model's response
    wasn't valid JSON)."""
    client = genai.Client(api_key=api_key or API_KEY)
    prompt = PROMPT_TEMPLATE.format(transcript=transcript, claim=claim)

    _throttle()
    start = time.perf_counter()
    response = client.models.generate_content(model=model or MODEL, contents=prompt)
    elapsed = time.perf_counter() - start

    result = _parse_response(response.text)
    result["elapsed"] = elapsed
    return result


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
        return {"tier": "?", "verdict": "PARSE_ERROR", "reason": text[:300]}


if __name__ == "__main__":
    transcript_name = input("Enter transcript filename (e.g. prim1.txt): ").strip()
    path = os.path.join(INPUT_DIR, transcript_name)
    with open(path, "r", encoding="utf-8") as f:
        source_transcript = f.read()
    print(f"Loaded {transcript_name} ({len(source_transcript)} chars).")

    x = "x"
    while x != "Q":
        x = input("Enter a target claim (or press Q to exit): ")
        if x == "Q":
            break

        result = judge_claim(source_transcript, x)
        print(f"Tier:    {result.get('tier')}")
        print(f"Verdict: {result.get('verdict')}")
        print(f"Reason:  {result.get('reason')}")
        print(f"Time:    {result.get('elapsed', 0):.2f}s\n")
