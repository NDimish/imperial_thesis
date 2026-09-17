# Deterministic Post-Market Monitoring for Ambient AI Scribes

MSc thesis project (Imperial College London) proposing an automated, deterministic
safety layer that sits between an ambient AI scribe (a clinical dictation tool that
turns a doctor-patient transcript into a SOAP note) and the electronic health
record, flagging likely **hallucinations** (fabricated content in the SOAP note)
and **omissions** (content said in the transcript but missing from the note)
before they reach the chart -- without training a model, calling an LLM judge, or
sending patient data to a third-party API.

The full write-up, including the regulatory background (SaMD/AIaMD), the risk
taxonomy, every method's derivation and results, and the LSEPI/policy sections,
is in [`Thesis/main.tex`](Thesis/main.tex) (compiled: `Thesis/main.pdf`). This
README covers the codebase, not the argument -- read the thesis for that.

## What's actually in here

Four deterministic candidate checkers were implemented and evaluated against a
hand-labelled, augmented version of the [PriMock57](https://github.com/babylonhealth/primock57)
consultation corpus, plus a fifth, narrower module (NoteExtractor) that sidesteps
the cross-document verification problem entirely:

| Checker | Module | Idea | Thesis section |
|---|---|---|---|
| medspaCy / UMLS / ConText | `Modules/medspacy_umls_checker.py` | Rule-based clinical NLP (negation/context-aware entity matching) | §9.2 |
| EmbedKDECheck | `Modules/embedkde_checker.py` | Kernel density estimation over PCA-reduced word embeddings -- flags text sitting in a low-density region of the other document's vocabulary | §9.3 |
| AlignScore | `Modules/alignscorechecker2.py`, `Modules/NLI_alignscore.py` | Trained NLI-style alignment model scoring claim vs. context support | §9.4 |
| SummaC | `Modules/summac_checker.py` | Document-level NLI, sentence-pair entailment matrix | §9.5 |
| NoteExtractor | `Modules/note_extractor.py` | In-document only: extracts negations/medicines/inferences from the note itself, no transcript needed | §9.6 |

**Headline result** (full 57-file set, see `Thesis/figures/Results/Results.tex` /
§10 for the complete table): every cross-document verification method has
precision under 25% except NoteExtractor (88.6%) -- the thesis's core finding is
that verifying spoken, disfluent dialogue against a structured note is a harder,
differently-shaped problem than the written-document-vs-written-document task
every one of these methods was originally benchmarked on, and that an
in-document extraction approach sidesteps that register mismatch entirely.

## Repo layout

```
project/
|-- run_checker_modules.py   run every SOAP checker over prim57, log P/R/F1
|-- run_condensers.py        run every NLP condenser, plot trend comparisons
|-- run_note_extractor.py    run NoteExtractor over prim57's real notes
|-- run_single_pair.py       run every module over one transcript/SOAP pair
|-- umls_checker.py          gold-standard eval of the shared QuickUMLS matcher
|-- stage2_inference_judge.py  LLM-judge baseline (Augnito Stage-2 protocol), for comparison only
|-- gemini_rate_limiter.py   cross-process-safe throttle for Gemini-backed checkers
|-- graph_maker.py           rebuilds every data-driven thesis figure from data_jsons/
|-- secrets_config.py        API keys (gitignored -- create your own, see below)
|-- requirements.txt
|
|-- Modules/                 the checkers themselves + shared Evaluate/CheckerModule base
|-- Medical condensor/       NLP "condensers" that trim a transcript before checking (§9.2.1)
|-- batch_runs/              per-checker scripts for running/logging a fixed file subset
|-- datamakerfiles/          builds prim57's injected hallucination/omission labels
|-- data_jsons/              small JSON exports behind every regenerable thesis figure
|-- graph_maker_output/      SVG/PDF/PNG output of graph_maker.py
|-- slurm/                   HPC cluster batch scripts (Imperial RCS)
|-- prim57/                  the working dataset -- see below
|-- google api/              browser extension + local Flask server (the deployable prototype)
`-- Thesis/                  LaTeX source (main.tex) and compiled PDF
```

A few working directories exist locally but are **gitignored** and won't appear
on a fresh clone: `Logs/` (every run's output -- recreated automatically),
`Clean Transcripts/` (larger raw transcript pool datamakerfiles draws from),
`figures/`, `extra/`, `assets/`, `generator files/` (scratch/dev-only outputs).
`secrets_config.py` is also gitignored since it holds API keys.

## Setup

1. **Python 3.11** (a conda env or venv both work -- the SLURM scripts default
   to a conda env named `soap-checker`). Python 3.12 breaks the medspaCy/PyRuSH
   install below (see the comment block at the top of `requirements.txt`).
2. Install the bulk of dependencies:
   ```
   pip install -r requirements.txt
   ```
3. **Three packages need installing separately**, bypassing stale upstream
   version constraints:
   ```
   pip install --no-deps medspacy==1.3.1 PyRuSH==1.0.13
   pip install --no-deps git+https://github.com/yuh-zha/AlignScore.git@a0936d5afee642a46b22f6c02a163478447aa493
   ```
   (`en_core_sci_sm` is already pinned as a direct URL in `requirements.txt`,
   so step 2 installs it -- no separate download needed.)
4. **QuickUMLS** (used by most checkers/condensers via `Medical condensor/umls_matching.py`).
   Install the `quickumls` package and build/download a local UMLS installation
   separately (free, but requires UMLS licence clearance), then edit
   `QUICKUMLS_INSTALL_DIR` near the top of `Medical condensor/umls_matching.py`
   to point at it -- it's a hardcoded constant, not an env var, so this needs
   editing on every new machine.
5. **GPU.** `SummaCChecker` defaults to `device="cuda"`; pass `device="cpu"`
   yourself if you don't have one.
6. **Secrets.** Create `secrets_config.py` at the project root with any API
   keys the AI/Gemini-backed checkers need (it's gitignored, never committed).
7. **Output location (optional).** Every script writes logs/JSON under `Logs/`
   by default; override with `export RESULTS_DIR=/path/to/results`.

## Running things

Every full-dataset script takes an optional file-count limit (random sample;
omit or use `57` for the whole set):

```
python run_checker_modules.py [limit]   # medspaCy/UMLS, EmbedKDE, AlignScore, SummaC
python run_condensers.py [limit]        # SciSpacy/Negspacy/QuickUMLS/medspaCy condensers
python run_note_extractor.py [limit]    # NoteExtractor
python run_single_pair.py <file>        # every module, one transcript/SOAP pair
```

Per-checker fixed-sample runs (used for the thesis's own logged results) live
in `batch_runs/`, e.g. `python batch_runs/kde_batch57.py`.

Rebuild the thesis's data-driven figures (bar/line charts backed by this
project's own logged results, output as SVG+PDF by default):

```
python graph_maker.py                # every figure
python graph_maker.py fig09          # just one
python graph_maker.py --png          # also write a high-dpi PNG
```

## Dataset: `prim57/`

Built on [PriMock57](https://github.com/babylonhealth/primock57) (57 simulated
UK primary-care consultations: audio, transcript, and a clinician-written SOAP
note). `datamakerfiles/prim_lib_injection.py` and `prim_lib_injection_extra.py`
inject six categories of synthetic corruption (negation flip, number edit,
inserted/hallucinated sentence, omitted detail, entity swap, drug-name switch)
into the ground-truth notes to build labelled hallucination/omission pairs for
evaluation -- see `Thesis/figures/Data figures/` and §8.1.1 for the full
augmentation methodology. Key subfolders: `cleaned transcripts/` (input),
`bad notes lib/` (corrupted SOAP notes, the checkers' input) paired with
`bad notes labels lib/` (the ground truth), and `labels_extraction/` /
`labels_inferences/` (NoteExtractor's separate, independent label set).

## Browser extension / deployable prototype

`google api/` contains the Manifest V3 Chrome extension and local Flask server
that demonstrate this as a working tool against a real ambient-scribe UI
(DiSCaScribe) -- see [`google api/README.md`](google%20api/README.md) for setup
and `Thesis/` §11 / Appendix B for the full write-up.
