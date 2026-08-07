import argparse
import os
import sys

import matplotlib.pyplot as plt

from Modules.evaluate import Evaluate

sys.path.insert(0, "Medical condensor")
from base import clean_transcript
from medspacy_condenser import MedspacyCondenser
# from negspacy_condenser import NegspacyCondenser
# from quickumls_condenser import QuickUMLSCondenser
# from scispacy_condenser import SciSpacyCondenser

INPUT_DIR = "prim57/cleaned transcripts"  # "Input"
OUTPUT_DIR = "prim57/bad notes lib"  # "Output"
LABELS_DIR = "prim57/bad notes labels lib"  # "Labels"
RESULTS_DIR = os.environ.get("RESULTS_DIR", "Logs")
PLOT_PATH = os.path.join(RESULTS_DIR, "main_trends.png")

# Which NLP condenser module runs on each transcript before it reaches the checker
# modules below. Swap this to MedspacyCondenser / SciSpacyCondenser /
# NegspacyCondenser / QuickUMLSCondenser (needs QUICKUMLS_INSTALL_DIR configured).
CONDENSER = MedspacyCondenser

RUNS_PER_MODULE = 1

# Fixed categorical order (validated palette) -- one color per module, never cycled.
MODULE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def load_checker_modules():
    """Imports and instantiates each checker module.

    Skips (and reports) any that fail to load -- e.g. AlignScoreChecker until
    CKPT_PATH is configured with a downloaded AlignScore checkpoint.
    """
    modules = []

    # AIChecker disabled for this batch: out of scope (this pass is about
    # tuning AlignScore/FactKB/Kdbe specifically), it's a Gemini API call
    # subject to transient outages (hit a 503 "high demand" error mid-run),
    # and its results aren't directly comparable to the other checkers'
    # anyway since it's a free-form LLM call, not a deterministic scorer with
    # a tunable threshold. Re-enable by uncommenting.
    # try:
    #     from Modules.AI_checker import AIChecker
    #     modules.append(AIChecker())
    # except Exception as e:
    #     print(f"Skipping AIChecker: {e}")

    # Hashed out for a HighRiskChecker-only run -- re-enable by uncommenting.
    # try:
    #     from Modules.alignscore_checker import AlignScoreChecker
    #     modules.append(AlignScoreChecker())
    # except Exception as e:
    #     print(f"Skipping AlignScoreChecker: {e}")

    try:
        from Modules.high_risk_checker import HighRiskChecker
        modules.append(HighRiskChecker())
    except Exception as e:
        print(f"Skipping HighRiskChecker: {e}")

    # SummaCChecker disabled: a real 5-file run measured it at ~1,829s/file
    # average (one file took 71 minutes) -- a full 57-file run would take
    # roughly a day for this checker alone. That's a real performance bug
    # (almost certainly re-loading or re-running its model per sentence
    # rather than batching), not something to wait out. Left importable but
    # commented out here until that's actually fixed; re-enable by
    # uncommenting once it's been profiled and sped up.
    # try:
    #     from Modules.summac_checker import SummaCChecker
    #     modules.append(SummaCChecker())
    # except Exception as e:
    #     print(f"Skipping SummaCChecker: {e}")

    # Hashed out for an AlignScoreChecker-only run (per request) -- re-enable
    # by uncommenting.
    # try:
    #     from Modules.factkb_checker import FactKBChecker
    #     modules.append(FactKBChecker())
    # except Exception as e:
    #     print(f"Skipping FactKBChecker: {e}")

    # try:
    #     from Modules.old.kdbe_checker import KdbeChecker
    #     modules.append(KdbeChecker())
    # except Exception as e:
    #     print(f"Skipping KdbeChecker: {e}")

    # try:
    #     from Modules.embedkde_checker import EmbedKdeChecker
    #     modules.append(EmbedKdeChecker())
    # except Exception as e:
    #     print(f"Skipping EmbedKdeChecker: {e}")

    # try:
    #     from Modules.concept_checker import ConceptChecker
    #     modules.append(ConceptChecker())
    # except Exception as e:
    #     print(f"Skipping ConceptChecker: {e}")

    # try:
    #     from Modules.deterministic_checker import DeterministicChecker
    #     modules.append(DeterministicChecker())
    # except Exception as e:
    #     print(f"Skipping DeterministicChecker: {e}")

    # try:
    #     from Modules.medspacy_umls_checker import MedspacyUmlsChecker
    #     modules.append(MedspacyUmlsChecker())
    # except Exception as e:
    #     print(f"Skipping MedspacyUmlsChecker: {e}")

    # try:
    #     from Modules.metamap_cui_checker import MetaMapCuiChecker
    #     modules.append(MetaMapCuiChecker())
    # except Exception as e:
    #     print(f"Skipping MetaMapCuiChecker: {e}")

    # try:
    #     from Modules.cql_checker import CqlChecker
    #     modules.append(CqlChecker())
    # except Exception as e:
    #     print(f"Skipping CqlChecker: {e}")

    return modules


def read_input_file(filename, condenser):
    """Reads a single transcript file from the Input folder and runs it through
    the configured NLP condenser module (see CONDENSER at the top of this file)
    before returning it."""
    path = os.path.join(INPUT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        transcript = clean_transcript(f.read())

    #condensed, _ = condenser.condense(transcript)
    return transcript


def read_output_file(filename):
    """Reads a single SOAP record file from the Output folder."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()




def main(limit=None):
    input_files = sorted(os.listdir(INPUT_DIR))
    output_files = sorted(os.listdir(OUTPUT_DIR))

    file_count = len(input_files) if limit is None else min(limit, len(input_files))

    modules = load_checker_modules()
    print(f"Loaded {len(modules)} checker module(s).")

    condenser = CONDENSER()
    print(f"Using {condenser.__class__.__name__} to condense transcripts.")


    for module in modules:
        module_name = module.__class__.__name__
        print(f"\n=== Running {module_name} ({RUNS_PER_MODULE} runs x {file_count} files) ===")
        evaluator = Evaluate(LABELS_DIR, module_name)

        for run in range(1, RUNS_PER_MODULE + 1):
            for i in range(file_count):
                input_filename = input_files[i]
                output_filename = output_files[i]

                transcript = read_input_file(input_filename, condenser)
                soap_note = read_output_file(output_filename)

                try:
                    errors, elapsed = module.check(transcript, soap_note)
                except Exception as e:
                    # A single file/module failure (e.g. a transient upstream
                    # API outage) used to kill the entire run, including
                    # every checker after this one that hadn't run yet.
                    # Skip just this file instead -- report it loudly so a
                    # real pattern of failures doesn't pass unnoticed, but
                    # let the other files and checkers still complete.
                    print(f"\n=== {module_name} on {input_filename} (run {run}/{RUNS_PER_MODULE}) -- FAILED ===")
                    print(f"  {type(e).__name__}: {e}")
                    continue

                print(f"\n=== {module_name} on {input_filename} (run {run}/{RUNS_PER_MODULE}) ===")
                for error in errors:
                    # HighRiskChecker (and any future severity-aware checker) returns
                    # (type, severity, detail_type, detail, section) 5-tuples instead
                    # of the (type, detail) 2-tuples every other checker here returns.
                    if len(error) == 5:
                        error_type, severity, detail_type, detail, section = error
                        print(f"{error_type} [{severity}/{detail_type}/{section}]: {detail}")
                    elif len(error) == 4:
                        error_type, severity, detail_type, detail = error
                        print(f"{error_type} [{severity}/{detail_type}]: {detail}")
                    else:
                        error_type, detail = error
                        print(f"{error_type}: {detail}")
                print(f"Time to complete: {elapsed:.2f}s")

                evaluator.compare(errors, input_filename, elapsed)


        evaluator.results()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOAP note checkers over transcript/output file pairs.")
    parser.add_argument(
        "limit",
        nargs="?",
        type=int,
        default=None,
        help="Number of file pairs to process per run, starting from the first (default: all files)",
    )
    args = parser.parse_args()

    print("started Main")

    main(args.limit)
