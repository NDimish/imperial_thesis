import medspacy
from medspacy.ner import TargetRule

def find_exact_clinical_issues(transcript_text, soap_text):
    # 1. Load pipeline with target matcher & ConText negation rules
    nlp = medspacy.load()
    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    
    # Define clinical targets (in production, use QuickUMLS for full UMLS/SNOMED CT)
    target_matcher.add([
        TargetRule("diarrhea", "CONDITION"),
        TargetRule("fever", "CONDITION"),
        TargetRule("steroid cream", "MEDICATION"),
        TargetRule("asthma", "CONDITION"),
        TargetRule("vomiting", "CONDITION"),
    ])

    doc_trans = nlp(transcript_text)
    doc_soap = nlp(soap_text)

    # 2. Extract structured tuples: (lemma_text, label, is_negated)
    def extract_tuples(doc):
        tuples = {}
        for ent in doc.ents:
            key = ent.text.lower()
            tuples[key] = {
                "text": ent.text,
                "label": ent.label_,
                "is_negated": ent._.is_negated,
                "span": ent.sent.text.strip()
            }
        return tuples

    trans_ents = extract_tuples(doc_trans)
    soap_ents = extract_tuples(doc_soap)

    issues = []

    # 3. Check for Exact Hallucinations (Present in SOAP, Absent in Transcript)
    for key, soap_data in soap_ents.items():
        if key not in trans_ents:
            issues.append({
                "issue_type": "HALLUCINATION",
                "concept": soap_data["text"],
                "category": soap_data["label"],
                "detail": f"'{soap_data['text']}' was added to the SOAP note but never mentioned in the transcript.",
                "soap_context": soap_data["span"]
            })

    # 4. Check for Exact Omissions & Contradictions
    for key, trans_data in trans_ents.items():
        if key not in soap_ents:
            issues.append({
                "issue_type": "OMISSION",
                "concept": trans_data["text"],
                "category": trans_data["label"],
                "detail": f"'{trans_data['text']}' was reported in the transcript but omitted from the SOAP note.",
                "transcript_context": trans_data["span"]
            })
        else:
            # Concept exists in both -> Check for Attribute Mismatches (e.g., Negation)
            soap_data = soap_ents[key]
            if trans_data["is_negated"] != soap_data["is_negated"]:
                issues.append({
                    "issue_type": "CONTRADICTION",
                    "concept": trans_data["text"],
                    "category": trans_data["label"],
                    "detail": f"Negation status flip for '{trans_data['text']}'. Transcript negated={trans_data['is_negated']}, SOAP negated={soap_data['is_negated']}.",
                    "transcript_context": trans_data["span"],
                    "soap_context": soap_data["span"]
                })

    return issues

# --- Run on Example ---


INPUT_DIR = "prim57/cleaned transcripts"
OUTPUT_DIR = "prim57/bad notes lib"
import os

# Example Usage
if __name__ == "__main__":


    x = "x"

    while x != "Q":
        x = input("Enter a target SOAP record (or press Q to exit): ")
        if x == "Q":
            break

        path = os.path.join(INPUT_DIR, x)
        with open(path, "r", encoding="utf-8") as f:
            transcript = f.read()
        path = os.path.join(OUTPUT_DIR, x)
        with open(path, "r", encoding="utf-8") as ft:
            soap_note = ft.read()
        

        issues = find_exact_clinical_issues(transcript, soap_note)

        for issue in issues:
            print(f"[{issue['issue_type']}] Concept: '{issue['concept']}' ({issue['category']})")
            print(f"  Reason: {issue['detail']}\n")