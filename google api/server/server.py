"""Local capture server for the DiSCaScribe browser extension (oogle
api/extension). Two jobs:

  1. /api/check -- the extension posts the four SOAP fields it just read off
     the Clinical Note tab; this runs Modules/note_extractor.py's
     extract_from_sections over them and returns the checklist items the
     floating widget renders (medicine / negation / inference / number).

  2. /events -- the extension posts the AI's original note draft and the
     clinician's edited/saved version (plus the transcript) whenever the
     clinician hits Save; this runs Modules/edit_diff_checker.py's diff_edit
     to size the edit and judge each changed span critical/non-critical,
     appends one row to data/events.csv (the durable, timestamped log), and
     writes a companion JSON file with the full text for the dashboard's
     drill-down view.

Everything here is local-only (SQLite-free, no auth beyond a static dev API
key) -- see oogle api/ambient-ai-capture-plan.html section 3 for why writes
are routed through the extension's background service worker rather than a
content script, and why this is a research/monitoring prototype, not
something to point at real patient data without institutional sign-off.

Run: python server.py  (serves http://localhost:5000, dashboard at /dashboard)
"""
import csv
import datetime
import json
import os
import sys
import uuid

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from Modules.edit_diff_checker import diff_edit  # noqa: E402
from Modules.note_extractor import extract_from_sections  # noqa: E402

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
CSV_PATH = os.path.join(DATA_DIR, "events.csv")

# Matches every detail_type Modules/edit_diff_checker.py's _classify_span can
# return -- kept here (not imported) since the CSV/dashboard only need the
# label strings, not the classifier itself.
DETAIL_TYPES = ["drug switch", "negation flip", "number edit", "inserted sentence", "omitted detail", "reworded"]
TYPE_SLUGS = [dt.replace(" ", "_") for dt in DETAIL_TYPES]
TYPE_COLUMNS = [f"type_{slug}" for slug in TYPE_SLUGS]

CSV_FIELDS = [
    "timestamp", "session_id", "session_file", "url", "version",
    "transcript_chars", "note_before_chars", "note_after_chars",
    "similarity", "edit_ratio", "num_changes", "num_critical", "critical",
] + TYPE_COLUMNS

API_KEY = os.environ.get("CAPTURE_API_KEY", "dev-only-key-replace-me")
SOAP_SECTIONS = ("Subjective", "Objective", "Assessment", "Plan")
CHECKLIST_COLORS = {
    "medicine": "#d64545",
    "number": "#c97a1f",
    "negation": "#3a6ea5",
    "inference": "#7a5ea8",
}


def _type_counts(changes):
    """{"type_drug_switch": n, ...} -- one count per DETAIL_TYPES entry,
    for either a CSV row (this function's normal caller) or a schema-
    migration backfill (_ensure_storage)."""
    counts = {col: 0 for col in TYPE_COLUMNS}
    for c in changes:
        col = f"type_{c['detail_type'].replace(' ', '_')}"
        if col in counts:
            counts[col] += 1
    return counts


def _ensure_storage():
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
        return

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fields = set(reader.fieldnames or [])
        rows = list(reader)
    if existing_fields >= set(CSV_FIELDS):
        return  # already current

    # Schema upgrade (added the per-type breakdown columns) -- backfill the
    # new columns on old rows from each row's own session JSON (which
    # already has the full change list) rather than zero-filling, so a
    # dashboard reload after this update doesn't lose history that's still
    # sitting right there in data/sessions/.
    for row in rows:
        if set(TYPE_COLUMNS) <= set(k for k, v in row.items() if v not in (None, "")):
            continue
        session_path = os.path.join(SESSIONS_DIR, row.get("session_file", ""))
        changes = []
        if os.path.isfile(session_path):
            with open(session_path, encoding="utf-8") as sf:
                changes = json.load(sf).get("result", {}).get("changes", [])
        row.update(_type_counts(changes))

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, 0) for k in CSV_FIELDS})


def _require_api_key():
    return request.headers.get("X-API-Key") == API_KEY


def _join_note(sections):
    """Renders the four SOAP fields back into one plain-text note (in fixed
    SOAP order) for edit_diff_checker.diff_edit, which compares whole-note
    strings -- keeps note_before/note_after comparable even if the
    clinician reordered nothing (SOAP order is fixed by the UI itself)."""
    parts = []
    for name in SOAP_SECTIONS:
        text = (sections or {}).get(name, "") or ""
        if text.strip():
            parts.append(f"{name}:\n{text.strip()}")
    return "\n\n".join(parts)


def _run_note_extractor(sections):
    """Wraps Modules/note_extractor.py's extract_from_sections() into the
    common shape every entry in CHECKLIST_MODULES returns: a list of dicts
    with type/section/text/highlight/color (item ids are assigned once,
    after every module's output is concatenated, in api_check() below)."""
    items = []
    for item_type, section, detail, highlight in extract_from_sections(sections):
        items.append(
            {
                "type": item_type,
                "section": section,
                "text": detail,
                "highlight": highlight or detail,
                "color": CHECKLIST_COLORS.get(item_type, "#666666"),
            }
        )
    return items


# Each entry: a function taking the {"Subjective": "...", ...} dict the
# extension posts, returning a list of checklist-item dicts in the shape
# _run_note_extractor produces above. api_check() runs every entry and
# concatenates the results, so adding a checklist module is "write a
# wrapper function + add it to this list" and removing one is "comment its
# line out" -- the same one-list-of-functions idiom main.py's
# load_checker_modules() uses for the offline checker pipeline, just
# without that function's per-import try/except, since these modules only
# ever take sections in and return items out (none of them have main.py's
# optional heavy dependencies, e.g. AlignScore's checkpoint file, that can
# fail to import). See "oogle api/code-walkthrough.html" for the full
# add-a-module walkthrough.
CHECKLIST_MODULES = [
    _run_note_extractor,
]


@app.route("/api/check", methods=["POST"])
def api_check():
    if not _require_api_key():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    sections = data.get("sections") or {}

    items = []
    for module_fn in CHECKLIST_MODULES:
        items.extend(module_fn(sections))
    for i, item in enumerate(items):
        item["id"] = f"{item['type'][0]}{i}"
    return jsonify({"items": items})


@app.route("/events", methods=["POST"])
def capture_event():
    if not _require_api_key():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(silent=True) or {}

    session_id = data.get("sessionId") or str(uuid.uuid4())
    transcript = data.get("transcript") or ""
    sections_before = data.get("sectionsBefore") or {}
    sections_after = data.get("sectionsAfter") or {}
    note_before = _join_note(sections_before)
    note_after = _join_note(sections_after)

    result = diff_edit(note_before, note_after)

    now = datetime.datetime.now(datetime.timezone.utc)
    timestamp = now.isoformat()
    session_file = f"{now.strftime('%Y%m%dT%H%M%S')}_{session_id}.json"

    _ensure_storage()
    with open(os.path.join(SESSIONS_DIR, session_file), "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "session_id": session_id,
                "url": data.get("url", ""),
                "version": data.get("version", ""),
                "transcript": transcript,
                "note_before": note_before,
                "note_after": note_after,
                "sections_before": sections_before,
                "sections_after": sections_after,
                "result": result,
            },
            f,
            indent=2,
        )

    row = {
        "timestamp": timestamp,
        "session_id": session_id,
        "session_file": session_file,
        "url": data.get("url", ""),
        "version": data.get("version", ""),
        "transcript_chars": len(transcript),
        "note_before_chars": len(note_before),
        "note_after_chars": len(note_after),
        "similarity": round(result["similarity"], 4),
        "edit_ratio": round(result["edit_ratio"], 4),
        "num_changes": len(result["changes"]),
        "num_critical": sum(1 for c in result["changes"] if c["critical"]),
        "critical": result["critical"],
        **_type_counts(result["changes"]),
    }
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)

    return jsonify({"status": "ok", "critical": result["critical"], "edit_ratio": result["edit_ratio"]})


def _read_events():
    _ensure_storage()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _daily_aggregates(events):
    """Groups events by UTC calendar date -> {date, count, avg_edit_ratio,
    critical, non_critical, types: {slug: count, ...}}, sorted oldest-to-
    newest -- the trend/control chart data the dashboard's SVG charts draw
    from. Both the edit-rate trend and the critical/non-critical split are
    §7 of oogle api/ambient-ai-capture-plan.html's strongest drift signal;
    the per-type breakdown is that section's follow-up point -- "edit-
    category drift, not just edit volume" (a flat edit rate with a rising
    share of drug/negation corrections is a worse sign than a rising rate
    made of stylistic rewording)."""
    by_date = {}
    for e in events:
        date = e["timestamp"][:10]
        bucket = by_date.setdefault(
            date,
            {"date": date, "ratios": [], "critical": 0, "non_critical": 0, "types": {slug: 0 for slug in TYPE_SLUGS}},
        )
        try:
            bucket["ratios"].append(float(e["edit_ratio"]))
        except (TypeError, ValueError):
            pass
        if e["critical"] == "True":
            bucket["critical"] += 1
        else:
            bucket["non_critical"] += 1
        for slug in TYPE_SLUGS:
            try:
                bucket["types"][slug] += int(e.get(f"type_{slug}") or 0)
            except (TypeError, ValueError):
                pass
    out = []
    for date in sorted(by_date):
        b = by_date[date]
        out.append(
            {
                "date": date,
                "count": len(b["ratios"]),
                "avg_edit_ratio": round(sum(b["ratios"]) / len(b["ratios"]), 4) if b["ratios"] else 0,
                "critical": b["critical"],
                "non_critical": b["non_critical"],
                "types": b["types"],
            }
        )
    return out


@app.route("/dashboard")
def dashboard():
    events = _read_events()
    total = len(events)
    avg_ratio = round(sum(float(e["edit_ratio"]) for e in events) / total, 4) if total else 0
    critical_count = sum(1 for e in events if e["critical"] == "True")
    pct_critical = round(100 * critical_count / total, 1) if total else 0
    # Every session partitions into exactly one of these three buckets:
    # nothing changed at all, changes but none critical (a "low" edit --
    # tidying, not a real correction), or at least one critical change.
    no_edit_count = sum(1 for e in events if int(e.get("num_changes") or 0) == 0)
    low_edit_count = sum(
        1 for e in events if int(e.get("num_changes") or 0) > 0 and e["critical"] != "True"
    )
    stats = {
        "total": total,
        "avg_ratio": avg_ratio,
        "critical_count": critical_count,
        "pct_critical": pct_critical,
        "no_edit_count": no_edit_count,
        "low_edit_count": low_edit_count,
    }
    return render_template(
        "dashboard.html",
        events=list(reversed(events)),
        stats=stats,
        daily=_daily_aggregates(events),
    )


@app.route("/dashboard/session/<session_file>")
def dashboard_session(session_file):
    safe_name = os.path.basename(session_file)
    path = os.path.join(SESSIONS_DIR, safe_name)
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    with open(path, encoding="utf-8") as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    _ensure_storage()
    app.run(port=5000, debug=True)
