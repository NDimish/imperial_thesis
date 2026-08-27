# DiSCaScribe capture pipeline

Three pieces, matching `ambient-ai-capture-plan.html`'s architecture:

- **`extension/`** -- a Chrome extension that runs on
  `https://discascribe.vercel.app/consultations/*`. It reads the Clinical
  Note tab's four SOAP fields and the Capture tab's transcript, asks the
  local server what's worth double-checking, and shows it as a floating
  checklist. It also records each edit pass (the note when you enter Edit
  mode vs. the note when you hit Save) and ships both to the local server.
- **`server/`** -- a local Flask server (`server.py`) that runs the
  checklist extraction and the before/after diff, logs every edit pass to
  a timestamped CSV, and serves a dashboard at `/dashboard`.
- **`ambient-ai-capture-plan.html`** -- the original design doc this was
  built from.

Nothing here talks to the internet except your own machine (`localhost:5000`)
and the DiSCaScribe page itself.

## 1. Run the server

The project's medspaCy/QuickUMLS stack lives in the `soap-checker` conda
environment, which is what the server imports `Modules/note_extractor.py`
and `Modules/edit_diff_checker.py` from. Flask has already been installed
into that environment as part of building this.

```
conda activate soap-checker
python "oogle api/server/server.py"
```

This starts the server on `http://localhost:5000` and serves the dashboard
at `http://localhost:5000/dashboard`. Leave it running while you use the
extension -- the extension has nothing to talk to without it.

Data is written to `oogle api/server/data/` (gitignored):
`events.csv` (one row per edit pass -- the durable, timestamped log you
asked for) and `sessions/*.json` (the full transcript/note/diff behind each
row, for the dashboard's drill-down).

## 2. Load the extension

1. Open `chrome://extensions` in Chrome.
2. Turn on **Developer mode** (top-right toggle).
3. Click **Load unpacked** and select the `oogle api/extension/` folder.
4. Open `https://discascribe.vercel.app/consultations/...` -- a small
   purple circle should appear bottom-right of the page once the server is
   also running.

Click the extension's toolbar icon any time to flip the master on/off
switch, or to jump straight to the dashboard.

## 3. Using it

- **The floating bubble** shows a count of unchecked items. Click it to
  expand the checklist, grouped by SOAP section and color-coded: red =
  medicine, orange = number (dose/frequency/duration/vital), blue =
  negation, purple = inference/impression.
- **Click a checklist item** to jump to it -- it switches to the Clinical
  Note tab, scrolls to and selects/highlights the matching text, and turns
  the item's row green once you've looked at it. Items are shown compactly
  as `Type: the flagged text` rather than the full sentence. A checked item
  stays green across a normal refresh as long as that specific finding
  didn't change; if the note has nothing to flag at all (e.g. no note has
  loaded yet), the panel says "No SOAP note detected" rather than leaving
  the previous checklist showing.
- **The &#8635; reset button** in the panel header re-pulls the checklist
  from the current note and clears every green check, if you want a clean
  slate without waiting for a Save.
- **The toggle inside the panel header** (and the one in the toolbar popup)
  both write to the same on/off flag -- flipping either one takes effect on
  every open tab immediately, no reload needed.
- **Every time you hit "Save"** (the edit-mode save action, not "Approve &
  file"), the extension ships the note as it was when you entered Edit mode
  and the note as you just saved it, plus the transcript, to the server.
  The server runs `Modules/edit_diff_checker.py` on the two versions and
  marks each changed line critical or not (a drug removed, a negation
  flipped, or a dose/frequency/duration changed -> critical; a pure
  rewording -> not).

## 4. Things worth knowing before you rely on this

- **The "Edit" and "Approve & file" buttons weren't inspected directly** --
  only "Save"/"Discard" (title-matched exactly) and the tab buttons were.
  The checklist widget and the Save-triggered capture don't depend on
  Edit/Approve & file at all, so this doesn't block anything working, but
  if either button ever needs its own hook, grab its outerHTML the same way
  the other elements were (right-click -> Inspect -> Copy outerHTML) and
  it's a small addition.
- **The API key is a static dev placeholder** (`dev-only-key-replace-me`,
  in both `extension/background.js` and `server.py`'s `CAPTURE_API_KEY` env
  var default) -- fine for one machine testing against itself, not fine
  once this is anything more than that.
- **This has not been run against the live page** -- it's built entirely
  from the outerHTML snippets you pasted and a syntax/logic check of every
  module in isolation (Modules/note_extractor.py's new number category and
  Modules/edit_diff_checker.py's critical/non-critical judgment were both
  run against realistic sample notes; the server's two endpoints and the
  dashboard were smoke-tested end-to-end with curl). The first real session
  on the live page is the actual test -- if a selector's off, the widget
  will say "Can't reach the local server" (wrong -- that message specifically
  means the server's down or unreachable) or simply show nothing, and the
  browser DevTools console (on the DiSCaScribe tab) is the place to look
  first.
- **"Stimulated Recall"** (the recall/rating tab) is ignored entirely --
  only "Capture" (transcript) and "Clinical Note" are read.
