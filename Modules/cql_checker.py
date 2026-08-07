"""CQL-style structured-resource checker: maps UMLS concepts extracted from
each document into synthetic FHIR resources (Condition, MedicationRequest,
Procedure, Observation), then runs a small set of deterministic boolean
logic gates -- written in the antecedent/consequent style real CQL "define"
statements use -- to check whether the SOAP note's structured content is
internally consistent and clinically supported.

This is a different kind of check from Modules/metamap_cui_checker.py's or
Modules/medspacy_umls_checker.py's plain CUI set-difference: those ask "does
this concept appear in both documents"; this asks "given the concepts that
DO appear, do they satisfy a cross-resource-type clinical constraint" --
e.g. a MedicationRequest for an antibiotic with no Condition anywhere to
justify it, or a documented Condition with no Observation/MedicationRequest/
Procedure anywhere addressing it. A pure CUI diff cannot catch either of
these: the individual concepts can each be genuinely present in both
documents while the RELATIONSHIP between them is still wrong. This checker
is meant to run alongside a CUI-diff checker, not replace one -- see the
accompanying HTML write-up's pipeline recommendation.

Resource mapping: each matched UMLS concept is assigned to one or more FHIR
resource *types* by its semantic type (TUI), using the same
Medical condensor/umls_matching.py ACCEPTED_SEMTYPES vocabulary the rest of
this project already validated for clinical relevance:
  Condition       T033 T037 T046 T047 T048 T184 T191   (findings/diseases)
  MedicationRequest  T121 T195 T200                     (drugs; T195 also
                                                          tagged "_antibiotic")
  Procedure       T061                                  (therapeutic/preventive)
  Observation     T023 T029 T031 T034 T040 T041 T059 T060 T201
                                                         (body/lab/function)
A concept can land in more than one bucket (e.g. a lab-result finding could
plausibly read as either Condition or Observation) -- deliberately permissive
here, since a rule only fires on ABSENCE of a whole resource type, so a
concept counted in an extra bucket can only make a rule less likely to fire,
never more.

Rule engine: real CQL (Clinical Quality Language, the HL7/CMS standard used
to author eCQMs) needs a CQL-to-ELM translation service (Java) and an
execution engine that runs the translated ELM against a FHIR-shaped patient
bundle -- typically the JS cql-execution + cql-exec-fhir packages via
Node.js. That toolchain is deliberately NOT wired into this Python project's
dependency management (it would mean bundling a Node runtime + npm install +
a separate translation service just for two illustrative rules). Instead:
  - RULES below is written in CQL's own define-statement style purely as
    documentation of the logic being evaluated (this is the same boolean
    antecedent/consequent shape a real CQL "Result" define would express).
  - _run_builtin_rules() evaluates that exact logic natively in Python
    against the synthetic FHIR bundles built above. This is what runs by
    default, and is what every result in this project's evaluation logs
    comes from.
  - _run_via_node_bridge() is included as the wiring for a genuine
    cql-execution engine, IF a caller sets CQL_EXECUTION_BRIDGE_DIR to a
    Node.js project with cql-execution/cql-exec-fhir installed and a
    run_rules.js entrypoint (reads the JSON bundle pair from stdin, returns
    a JSON array of violations on stdout). Falls back to the built-in
    evaluator automatically if the bridge isn't configured or errors out,
    so a missing Node install never breaks a run.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Medical condensor"))
from base import split_turns  # noqa: E402 -- path insert must run first
from umls_matching import get_matcher, is_common_word  # noqa: E402 -- path insert must run first

from Modules.base import CheckerModule

# Copied from Modules/medspacy_umls_checker.py's JUNK_CONCEPT_DENYLIST -- see
# Modules/metamap_cui_checker.py's copy for why this is duplicated rather
# than imported (keeps this checker's dependency chain independent).
#
# "patient", "diagnosed", "history", "complaints" added after a direct
# synthetic-case audit of this module's two rules found both silently dead
# on any conventionally-phrased note: "diagnosed"/T060 and "patient"/T031
# both land in the Observation bucket (see SEMTYPE_RESOURCE_MAP), and those
# two words alone appear in nearly every SOAP note -- so
# "condition_without_plan"'s "PlanModeled" check came back true almost
# unconditionally, never actually testing for a real Observation/
# MedicationRequest/Procedure. "history" (T033) and "complaints" (T033)
# similarly polluted the Condition bucket with a generic collision rather
# than an asserted finding -- e.g. "no complaints at all" (a negation, which
# this checker has no mechanism to read) still counted as a documented
# Condition. Confirmed via two targeted synthetic cases (an antibiotic with
# no real condition anywhere, and a diabetes diagnosis with no real plan
# anywhere) that both rules correctly fire once these four are excluded.
JUNK_CONCEPT_DENYLIST = {
    "hand", "controll", "control", "other things", "move", "mind", "said",
    "close", "test", "able", "times", "life", "difficult", "little",
    "stage", "recap", "keen", "remember", "feels", "much", "normal",
    "nice", "listened", "sampled", "quite often", "examined", "therex",
    "etests", "couplet", "coinfection",
    "plan", "patient", "diagnosed", "history", "complaints",
}


def _is_junk_concept(term):
    return term.strip().lower() in JUNK_CONCEPT_DENYLIST


SEMTYPE_RESOURCE_MAP = {
    "T033": "Condition", "T037": "Condition", "T046": "Condition",
    "T047": "Condition", "T048": "Condition", "T184": "Condition", "T191": "Condition",
    "T121": "MedicationRequest", "T195": "MedicationRequest", "T200": "MedicationRequest",
    "T061": "Procedure",
    "T023": "Observation", "T029": "Observation", "T031": "Observation",
    "T034": "Observation", "T040": "Observation", "T041": "Observation",
    "T059": "Observation", "T060": "Observation", "T201": "Observation",
}
ANTIBIOTIC_SEMTYPES = {"T195"}
RESOURCE_TYPES = ("Condition", "MedicationRequest", "Procedure", "Observation")

# A concept's Observation semtypes span both purely descriptive types (T023
# Body Part, T029 Body Location, T031 Body Substance, T040 Organism
# Function, T041 Mental Process, T201 Clinical Attribute -- "arm", "sleep",
# "mood") and genuinely actionable ones (T034 Lab/Test Result, T059 Lab
# Procedure, T060 Diagnostic Procedure). "condition_without_plan" (rule 2
# below) needs the latter only -- direct synthetic-case testing found a note
# that mentions any body part or symptom-adjacent word at all (nearly every
# note) trivially satisfied "PlanModeled" against the full Observation
# bucket, without any lab, test, procedure, or medication ever having been
# documented. PLAN_EVIDENCE_SEMTYPES is the narrower "was something actually
# done or ordered" set that fixes that: Procedure and MedicationRequest
# concepts always qualify (see _build_bundle), plus only the actionable
# subset of Observation.
PLAN_EVIDENCE_SEMTYPES = {"T034", "T059", "T060"}

# Optional bridge to a real cql-execution (npm) engine -- see module docstring.
CQL_EXECUTION_BRIDGE_DIR = os.environ.get("CQL_EXECUTION_BRIDGE_DIR")


class CqlChecker(CheckerModule):
    """See module docstring for the full extraction -> FHIR-mapping ->
    boolean-gate pipeline this implements."""

    # Written in CQL's own define-statement style -- documentation of the
    # exact logic _run_builtin_rules() evaluates natively, and the payload a
    # real cql-execution bridge (_run_via_node_bridge) would be asked to run
    # instead. error_type is chosen to match this project's existing label
    # vocabulary (see Modules/evaluate.py) rather than introducing a
    # separate "rule_violation" category the gold labels have no examples
    # of: an unsupported MedicationRequest is a structural hallucination
    # (something asserted with nothing backing it); a Condition with no
    # modeled plan is a structural omission (something established but never
    # followed through on).
    RULES = (
        {
            "name": "antibiotic_without_condition",
            "cql": (
                'define "AntibioticOrdered": exists(\n'
                '  [MedicationRequest] M where M.code.coding.code in "Antibiotic value set")\n'
                'define "ConditionDocumented": exists([Condition])\n'
                'define "Result": "AntibioticOrdered" and not "ConditionDocumented"'
            ),
            "error_type": "hallucination",
            "message": "antibiotic ordered with no documented condition anywhere to justify it",
        },
        {
            "name": "condition_without_plan",
            "cql": (
                'define "ConditionDocumented": exists([Condition])\n'
                'define "PlanModeled": exists([MedicationRequest]) or exists([Procedure])\n'
                '  or exists([Observation] O where O.code.coding.code in "Lab/Diagnostic value set")\n'
                'define "Result": "ConditionDocumented" and not "PlanModeled"'
            ),
            "error_type": "omission",
            "message": "condition documented with no medication, procedure, or lab/diagnostic result addressing it",
        },
    )

    def __init__(self, quickumls_install_dir=None):
        self._matcher = get_matcher(quickumls_install_dir)

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        transcript_bundle = self._build_bundle(transcript)
        note_bundle = self._build_bundle(soap_note)

        if CQL_EXECUTION_BRIDGE_DIR:
            errors = self._run_via_node_bridge(transcript_bundle, note_bundle)
        else:
            errors = self._run_builtin_rules(transcript_bundle, note_bundle)

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed

    # ------------------------------------------------------------------
    # Concept extraction -> synthetic FHIR resource bundling
    # ------------------------------------------------------------------

    def _build_bundle(self, text):
        """Returns {"Condition": {cui: {term, sentence}}, ..., "_antibiotic":
        {...}, "_plan_evidence": {...}} -- one line/turn (see split_turns) at
        a time, same line-scoped extraction MedspacyUmlsChecker/
        MetaMapCuiChecker use, so a matched concept keeps the sentence it
        actually came from for a human-/evaluator-readable error message.

        "_plan_evidence" is the narrower "something was actually done or
        ordered" subset the "condition_without_plan" rule needs -- see
        PLAN_EVIDENCE_SEMTYPES's comment for why the full Observation bucket
        (built for general use, e.g. any future rule that wants "was a body
        system mentioned at all") is too broad for that specific check."""
        bundle = {rtype: {} for rtype in RESOURCE_TYPES}
        bundle["_antibiotic"] = {}
        bundle["_plan_evidence"] = {}

        for _speaker, line_text in split_turns(text):
            if not line_text.strip():
                continue
            matches = self._matcher.match(line_text, best_match=True, ignore_syntax=False)
            for group in matches:
                for m in group:
                    term, ngram, cui = m["term"], m["ngram"], m["cui"]
                    if is_common_word(term) or is_common_word(ngram):
                        continue
                    if _is_junk_concept(term) or _is_junk_concept(ngram):
                        continue

                    semtypes = m.get("semtypes", set())
                    entry = {"term": term, "sentence": line_text.strip()}
                    for rtype in RESOURCE_TYPES:
                        if any(SEMTYPE_RESOURCE_MAP.get(t) == rtype for t in semtypes):
                            bundle[rtype].setdefault(cui, entry)
                    if semtypes & ANTIBIOTIC_SEMTYPES:
                        bundle["_antibiotic"].setdefault(cui, entry)
                    is_med_or_procedure = any(
                        SEMTYPE_RESOURCE_MAP.get(t) in ("MedicationRequest", "Procedure") for t in semtypes
                    )
                    if semtypes & PLAN_EVIDENCE_SEMTYPES or is_med_or_procedure:
                        bundle["_plan_evidence"].setdefault(cui, entry)

        return bundle

    # ------------------------------------------------------------------
    # Built-in boolean rule evaluator (default -- see module docstring)
    # ------------------------------------------------------------------

    def _run_builtin_rules(self, transcript_bundle, note_bundle):
        errors = []

        antibiotic_rule = self.RULES[0]
        if note_bundle["_antibiotic"] and not note_bundle["Condition"] and not transcript_bundle["Condition"]:
            entry = next(iter(note_bundle["_antibiotic"].values()))
            errors.append((antibiotic_rule["error_type"], self._describe(antibiotic_rule, entry)))

        plan_rule = self.RULES[1]
        if note_bundle["Condition"] and not note_bundle["_plan_evidence"]:
            entry = next(iter(note_bundle["Condition"].values()))
            errors.append((plan_rule["error_type"], self._describe(plan_rule, entry)))

        return errors

    @staticmethod
    def _describe(rule, entry):
        return f"{entry['sentence']} ({entry['term']}: CQL \"{rule['name']}\" -- {rule['message']})"

    # ------------------------------------------------------------------
    # Optional real cql-execution bridge (see module docstring)
    # ------------------------------------------------------------------

    def _run_via_node_bridge(self, transcript_bundle, note_bundle):
        try:
            payload = json.dumps({
                "transcript": self._bundle_to_fhir(transcript_bundle),
                "note": self._bundle_to_fhir(note_bundle),
                "rules": [{"name": r["name"], "cql": r["cql"]} for r in self.RULES],
            })
            result = subprocess.run(
                ["node", "run_rules.js"],
                input=payload, capture_output=True, text=True,
                cwd=CQL_EXECUTION_BRIDGE_DIR, timeout=60, check=True,
            )
            violations = json.loads(result.stdout)
            return [(v["error_type"], v["message"]) for v in violations]
        except Exception as e:
            print(f"CqlChecker: Node cql-execution bridge failed ({e}); falling back to built-in rule evaluator.")
            return self._run_builtin_rules(transcript_bundle, note_bundle)

    @staticmethod
    def _bundle_to_fhir(bundle):
        """Minimal, illustrative FHIR Bundle JSON -- enough structure for a
        real cql-execution + cql-exec-fhir engine to run [ResourceType]
        retrieves against, not a spec-complete FHIR resource."""
        entries = []
        for rtype in RESOURCE_TYPES:
            for cui, info in bundle[rtype].items():
                entries.append({"resource": {
                    "resourceType": rtype,
                    "code": {"coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/umls",
                        "code": cui,
                        "display": info["term"],
                    }]},
                    "text": {"div": info["sentence"]},
                }})
        return {"resourceType": "Bundle", "entry": entries}
