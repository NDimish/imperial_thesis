// Content script for discascribe.vercel.app/consultations/* -- reads the
// Clinical Note tab's four SOAP fields (both its edit-mode <textarea>s and
// its view-mode <section>s -- both are kept mounted in the DOM, just
// hidden, whichever tab is active, confirmed by inspecting the real page)
// and the Capture tab's transcript turns, asks the local server for a
// checklist of things worth double-checking, and renders it as a floating
// widget. Also captures each edit pass (snapshot when edit mode starts ->
// snapshot when "Save" is clicked) and ships it to the local server for the
// critical/non-critical diff judgment.
//
// Everything here is gated on the on/off toggle in chrome.storage.sync --
// see popup.js. See "oogle api/ambient-ai-capture-plan.html" for the wider
// design rationale.
(function () {
  "use strict";

  const SOAP_SECTIONS = ["Subjective", "Objective", "Assessment", "Plan"];
  const TYPE_LABELS = {
    medicine: "Drug name",
    number: "Number",
    negation: "Negation",
    inference: "Inference",
  };

  let captureEnabled = true;
  let shadow = null;
  let hostEl = null;
  let panelOpen = false;
  let checklistItems = [];
  // Keyed by content (type|section|highlight), not by the server's
  // index-based item.id -- ids shift every time the checklist regenerates,
  // but the same real finding (e.g. the same drug name) should stay
  // checked across a refresh if it didn't actually change. See
  // itemKey() below.
  let checkedKeys = new Set();
  let editBaseline = null; // { sections, capturedAt } snapshot from when edit mode was entered
  let wasEditing = false;

  // ---------- small utils ----------

  function debounce(fn, ms) {
    let t = null;
    return (...args) => {
      clearTimeout(t);
      t = setTimeout(() => fn(...args), ms);
    };
  }

  function capitalize(s) {
    return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
  }

  function normalizeWs(s) {
    return (s || "").replace(/\s+/g, " ").trim();
  }

  function getSessionId() {
    const m = location.pathname.match(/\/consultations\/([^/]+)/);
    return m ? m[1] : location.pathname;
  }

  function getVersionLabel() {
    const btn = document.querySelector('button[title="This version is saved to the archive"]');
    if (!btn) return "";
    const m = btn.textContent.match(/v\d+/i);
    return m ? m[0] : "";
  }

  // ---------- reading the note ----------

  // Collects each <p>/<li>'s own text as one line, recursing through
  // anything else (the header-less content wrapper div, the <ul>). Deliberately
  // .textContent-based, not .innerText -- .innerText only inserts proper
  // block-level line breaks for an element that's actually attached AND
  // rendered (display != none), so it silently glues every bullet into one
  // run-on line for (a) a detached clone (no layout at all) and (b) a
  // section sitting inside the currently-inactive tab's hidden panel
  // (display:none, so no rendered layout either) -- both real cases here,
  // since the Clinical Note tab's content stays mounted-but-hidden when
  // another tab is active. textContent has no such dependency.
  function collectViewLines(root, out) {
    for (const child of root.children) {
      if (child.tagName === "P" || child.tagName === "LI") {
        const t = child.textContent.trim();
        if (t) out.push(t);
      } else {
        collectViewLines(child, out);
      }
    }
  }

  function readNoteSections() {
    const sections = {};

    // Edit mode: textareas persist in the DOM (hidden, not unmounted) even
    // when the Clinical Note tab isn't the active tab.
    let found = false;
    for (const name of SOAP_SECTIONS) {
      const ta = document.getElementById(`soap-${name}`);
      if (ta) {
        sections[name] = ta.value || "";
        found = true;
      }
    }
    if (found) return sections;

    // View mode fallback: <section><h2>Name</h2>...content...</section>.
    // h2 is skipped naturally -- collectViewLines only picks up P/LI tags.
    document.querySelectorAll("section").forEach((sec) => {
      const h2 = sec.querySelector("h2");
      if (!h2) return;
      const name = SOAP_SECTIONS.find((n) => h2.textContent.trim() === n);
      if (!name) return;
      const lines = [];
      collectViewLines(sec, lines);
      const text = lines.join("\n").trim();
      sections[name] = text.toLowerCase() === "empty" ? "" : text;
    });
    return sections;
  }

  function isEditMode() {
    return !!document.getElementById("soap-Subjective");
  }

  function findViewSection(sectionName) {
    const target = capitalize(sectionName);
    return Array.from(document.querySelectorAll("section")).find((sec) => {
      const h2 = sec.querySelector("h2");
      return h2 && h2.textContent.trim() === target;
    });
  }

  // ---------- reading the transcript (Capture tab) ----------

  function readTranscript() {
    const audio = document.querySelector("audio");
    if (!audio) return "";
    const card = audio.closest(".rounded-2xl");
    if (!card) return "";
    const turnsContainer = card.querySelector(".space-y-5 .space-y-4");
    if (!turnsContainer) return "";
    const lines = [];
    turnsContainer.querySelectorAll(":scope > div").forEach((turn) => {
      const speakerEl = turn.querySelector("span");
      const bubbleEl = turn.querySelector("div");
      const speaker = speakerEl ? speakerEl.textContent.trim() : "Speaker";
      const text = bubbleEl ? bubbleEl.textContent.trim() : "";
      if (text) lines.push(`${speaker}: ${text}`);
    });
    return lines.join("\n");
  }

  // ---------- toolbar hooks ----------

  function findToolbarButton(matcher) {
    return Array.from(document.querySelectorAll('button[data-slot="button"]')).find(matcher);
  }

  function ensureClinicalNoteTabActive() {
    const btn = Array.from(document.querySelectorAll("button")).find(
      (b) => normalizeWs(b.textContent) === "Clinical Note" && b.className.includes("border-b-2")
    );
    if (btn && !btn.className.includes("border-primary")) btn.click();
  }

  // ---------- jump-to / highlight ----------

  function highlightInSection(sectionEl, needle) {
    if (!needle) return false;
    const walker = document.createTreeWalker(sectionEl, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      const hay = node.nodeValue;
      const idx = hay.toLowerCase().indexOf(needle.toLowerCase());
      if (idx !== -1) {
        const range = document.createRange();
        range.setStart(node, idx);
        range.setEnd(node, Math.min(idx + needle.length, hay.length));
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        const el = node.parentElement || sectionEl;
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        return true;
      }
    }
    return false;
  }

  function jumpToItem(item) {
    ensureClinicalNoteTabActive();
    setTimeout(() => {
      const ta = document.getElementById(`soap-${capitalize(item.section)}`);
      if (ta) {
        const idx = ta.value.toLowerCase().indexOf((item.highlight || "").toLowerCase());
        ta.scrollIntoView({ behavior: "smooth", block: "center" });
        if (idx !== -1) {
          ta.focus();
          ta.setSelectionRange(idx, idx + item.highlight.length);
        }
        return;
      }
      const sectionEl = findViewSection(item.section);
      if (sectionEl) highlightInSection(sectionEl, item.highlight);
    }, 150); // lets the tab-switch re-render settle before we search
  }

  // ---------- extension-context guard ----------
  //
  // Reloading the extension from chrome://extensions (normal during dev)
  // replaces the background service worker but leaves any already-open
  // DiSCaScribe tab's content script running -- the next chrome.* call it
  // makes throws "Extension context invalidated." Rather than let that
  // surface as an uncaught error, detect it once and quietly stop trying
  // until the tab itself is reloaded (which re-injects a fresh, valid
  // content script).

  function isContextValid() {
    try {
      return !!(chrome && chrome.runtime && chrome.runtime.id);
    } catch (e) {
      return false;
    }
  }

  function handleContextInvalidated() {
    if (observer) observer.disconnect();
    setWidgetVisible(false);
  }

  function safeSendMessage(message, callback) {
    if (!isContextValid()) {
      handleContextInvalidated();
      return;
    }
    try {
      chrome.runtime.sendMessage(message, (response) => {
        if (chrome.runtime.lastError) {
          handleContextInvalidated();
          return;
        }
        if (callback) callback(response);
      });
    } catch (e) {
      handleContextInvalidated();
    }
  }

  // ---------- checklist fetch + widget ----------

  function itemKey(item) {
    return `${item.type}|${item.section}|${item.highlight || item.text}`;
  }

  function refreshChecklist() {
    if (!captureEnabled || !shadow) return;
    const sections = readNoteSections();
    const hasContent = Object.values(sections).some((v) => v && v.trim());
    if (!hasContent) {
      checklistItems = [];
      renderNoSoap();
      return;
    }
    safeSendMessage({ type: "CHECK_NOTE", payload: { sections } }, (response) => {
      if (!response || !response.ok) {
        renderPanelError();
        return;
      }
      checklistItems = response.items;
      // checkedKeys is intentionally NOT cleared here -- an item that
      // didn't actually change keeps its green check across this refresh
      // (matched by itemKey, not by the regenerated item.id). Only the
      // explicit Reset button or a finalized submit clears it.
      renderPanel();
    });
  }
  const debouncedRefreshChecklist = debounce(refreshChecklist, 1200);

  function updateBubbleCount() {
    const countEl = shadow.getElementById("bubble-count");
    const unchecked = checklistItems.filter((it) => !checkedKeys.has(itemKey(it))).length;
    countEl.textContent = String(unchecked);
    countEl.hidden = unchecked === 0;
  }

  function renderNoSoap() {
    if (!shadow) return;
    shadow.getElementById("panel-body").innerHTML = `<div class="empty">No SOAP note detected.</div>`;
    shadow.getElementById("bubble-count").hidden = true;
  }

  function renderPanelError() {
    if (!shadow) return;
    const body = shadow.getElementById("panel-body");
    body.innerHTML = `<div class="empty">Can't reach the local server (localhost:5000). Is it running?</div>`;
  }

  function renderPanel() {
    if (!shadow) return;
    const body = shadow.getElementById("panel-body");
    updateBubbleCount();

    if (!checklistItems.length) {
      body.innerHTML = `<div class="empty">Nothing flagged.</div>`;
      return;
    }

    const bySection = {};
    for (const item of checklistItems) {
      (bySection[item.section] = bySection[item.section] || []).push(item);
    }

    let html = "";
    for (const section of Object.keys(bySection)) {
      html += `<div class="section-label">${capitalize(section)}</div><ul class="item-list">`;
      for (const item of bySection[section]) {
        const checked = checkedKeys.has(itemKey(item)) ? "checked" : "";
        html += `<li class="item ${checked}" data-id="${item.id}">
          <span class="dot" style="background:${item.color}"></span>
          <span class="item-text"><span class="item-type">${TYPE_LABELS[item.type] || item.type}:</span> ${escapeHtml(item.highlight || item.text)}</span>
        </li>`;
      }
      html += `</ul>`;
    }
    body.innerHTML = html;

    body.querySelectorAll(".item").forEach((li) => {
      li.addEventListener("click", () => {
        const id = li.dataset.id;
        const item = checklistItems.find((it) => it.id === id);
        if (!item) return;
        jumpToItem(item);
        checkedKeys.add(itemKey(item));
        li.classList.add("checked");
        updateBubbleCount();
      });
    });
  }

  function escapeHtml(s) {
    return (s || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  }

  const WIDGET_CSS = `
    :host { all: initial; }
    * { box-sizing: border-box; font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }
    .bubble {
      position: fixed; right: 22px; bottom: 22px; z-index: 2147483000;
      width: 52px; height: 52px; border-radius: 50%;
      background: #6d28d9; color: white; display: flex; align-items: center; justify-content: center;
      box-shadow: 0 4px 14px rgba(0,0,0,.25); cursor: pointer; font-size: 20px;
    }
    .bubble:hover { background: #5b21b6; }
    .bubble-count {
      position: absolute; top: -4px; right: -4px; background: #c1440e; color: white;
      font-size: 11px; font-weight: 700; min-width: 18px; height: 18px; border-radius: 9px;
      display: flex; align-items: center; justify-content: center; padding: 0 4px;
    }
    .panel {
      position: fixed; right: 22px; bottom: 84px; z-index: 2147483000;
      width: 320px; max-height: 60vh; background: #ffffff; border-radius: 14px;
      box-shadow: 0 8px 30px rgba(0,0,0,.22); display: flex; flex-direction: column; overflow: hidden;
      border: 1px solid #e2ddd1;
    }
    .panel[hidden] { display: none; }
    .panel-header {
      display: flex; align-items: center; gap: 8px; padding: 12px 14px; border-bottom: 1px solid #ede9de;
      background: #faf8f4;
    }
    .panel-title { font-size: 13px; font-weight: 700; color: #20211d; flex: 1; }
    .toggle { position: relative; display: inline-block; width: 32px; height: 18px; flex-shrink: 0; }
    .toggle input { opacity: 0; width: 0; height: 0; }
    .toggle-track { position: absolute; inset: 0; background: #d8d3c6; border-radius: 20px; cursor: pointer; transition: background .15s; }
    .toggle-track::before { content: ""; position: absolute; height: 14px; width: 14px; left: 2px; top: 2px; background: white; border-radius: 50%; transition: transform .15s; }
    .toggle input:checked + .toggle-track { background: #0f5c56; }
    .toggle input:checked + .toggle-track::before { transform: translateX(14px); }
    .close-btn { border: none; background: none; font-size: 18px; line-height: 1; cursor: pointer; color: #8a897f; padding: 2px 4px; }
    .close-btn:hover { color: #20211d; }
    .reset-btn { border: none; background: none; font-size: 15px; line-height: 1; cursor: pointer; color: #8a897f; padding: 2px 6px; }
    .reset-btn:hover { color: #20211d; }
    .panel-body { overflow-y: auto; padding: 8px 10px 12px; }
    .empty { color: #8a897f; font-size: 12.5px; padding: 16px 6px; text-align: center; }
    .section-label { font-size: 10.5px; text-transform: uppercase; letter-spacing: .04em; color: #8a897f; font-weight: 700; margin: 10px 6px 4px; }
    .item-list { list-style: none; margin: 0; padding: 0; }
    .item {
      display: flex; align-items: flex-start; gap: 8px; padding: 8px 8px; border-radius: 8px; cursor: pointer;
      font-size: 12.5px; line-height: 1.4; color: #20211d;
    }
    .item:hover { background: #f1ede6; }
    .item.checked { background: #e2f3e5; }
    .item.checked:hover { background: #d3ecd8; }
    .dot { width: 9px; height: 9px; border-radius: 50%; margin-top: 4px; flex-shrink: 0; }
    .item-type { font-weight: 700; margin-right: 4px; }
  `;

  function createWidget() {
    hostEl = document.createElement("div");
    hostEl.id = "discascribe-checklist-host";
    document.body.appendChild(hostEl);
    shadow = hostEl.attachShadow({ mode: "open" });
    shadow.innerHTML = `
      <style>${WIDGET_CSS}</style>
      <div class="bubble" id="bubble" title="Note checklist">
        <span>&#10003;</span>
        <span class="bubble-count" id="bubble-count" hidden>0</span>
      </div>
      <div class="panel" id="panel" hidden>
        <div class="panel-header">
          <span class="panel-title">Note checklist</span>
          <button class="reset-btn" id="reset-btn" title="Reload note & clear checks">&#8635;</button>
          <label class="toggle" title="Turn capture on/off">
            <input type="checkbox" id="toggle-input" checked>
            <span class="toggle-track"></span>
          </label>
          <button class="close-btn" id="close-btn">&times;</button>
        </div>
        <div class="panel-body" id="panel-body"><div class="empty">Loading&hellip;</div></div>
      </div>
    `;

    shadow.getElementById("bubble").addEventListener("click", () => {
      panelOpen = !panelOpen;
      shadow.getElementById("panel").hidden = !panelOpen;
      if (panelOpen) refreshChecklist();
    });
    shadow.getElementById("close-btn").addEventListener("click", () => {
      panelOpen = false;
      shadow.getElementById("panel").hidden = true;
    });
    shadow.getElementById("reset-btn").addEventListener("click", () => {
      checkedKeys.clear();
      refreshChecklist();
    });
    const toggleInput = shadow.getElementById("toggle-input");
    toggleInput.checked = captureEnabled;
    toggleInput.addEventListener("change", () => {
      if (!isContextValid()) {
        handleContextInvalidated();
        return;
      }
      try {
        chrome.storage.sync.set({ captureEnabled: toggleInput.checked });
      } catch (e) {
        handleContextInvalidated();
      }
    });
  }

  function setWidgetVisible(visible) {
    if (hostEl) hostEl.style.display = visible ? "" : "none";
  }

  // ---------- edit-pass capture ----------

  function pollEditModeTransition() {
    const editing = isEditMode();
    if (editing && !wasEditing) {
      editBaseline = { sections: readNoteSections(), capturedAt: Date.now() };
    }
    wasEditing = editing;
  }

  function handleFinalize() {
    if (!captureEnabled) return;
    const sectionsAfter = readNoteSections();
    const sectionsBefore = editBaseline ? editBaseline.sections : sectionsAfter;
    const payload = {
      sessionId: getSessionId(),
      url: location.href,
      version: getVersionLabel(),
      transcript: readTranscript(),
      sectionsBefore,
      sectionsAfter,
      capturedAt: editBaseline ? editBaseline.capturedAt : Date.now(),
      finalizedAt: Date.now(),
    };
    safeSendMessage({ type: "NOTE_FINALIZED", payload }, () => {
      // Once a version is submitted, its checklist is done with -- clear
      // every green check and re-pull the checklist fresh against
      // whatever the page shows post-save, rather than carrying stale
      // checks (or a stale item list) into the next edit pass.
      checkedKeys.clear();
      checklistItems = [];
      setTimeout(refreshChecklist, 400); // let the page's own post-save re-render settle first
    });
    editBaseline = null; // next "Edit" entry recaptures a fresh baseline
  }

  // Capture phase on document so this runs before the page's own click
  // handler can save-and-swap the DOM out from under us.
  document.addEventListener(
    "click",
    (e) => {
      if (!captureEnabled) return;
      const btn = e.target.closest('button[data-slot="button"]');
      if (btn && btn.title === "Save as the next note version") {
        handleFinalize();
      }
    },
    true
  );

  // ---------- boot ----------

  const debouncedOnMutation = debounce(() => {
    pollEditModeTransition();
    refreshChecklist();
  }, 800);

  let observer = null;
  let started = false;

  function start() {
    if (!hostEl) createWidget();
    setWidgetVisible(true);
    if (started) return; // avoid stacking duplicate listeners across repeated off/on toggles
    started = true;
    observer = new MutationObserver(debouncedOnMutation);
    observer.observe(document.body, { childList: true, subtree: true });
    document.addEventListener(
      "input",
      (e) => {
        if (!captureEnabled) return;
        if (e.target && e.target.id && e.target.id.startsWith("soap-")) {
          debouncedRefreshChecklist();
        }
      },
      true
    );
    pollEditModeTransition();
    setTimeout(refreshChecklist, 1000);
  }

  function stop() {
    setWidgetVisible(false);
  }

  chrome.storage.sync.get({ captureEnabled: true }, (result) => {
    captureEnabled = result.captureEnabled;
    if (captureEnabled) start();
  });

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "sync" || !("captureEnabled" in changes)) return;
    captureEnabled = changes.captureEnabled.newValue;
    if (shadow) shadow.getElementById("toggle-input").checked = captureEnabled;
    if (captureEnabled) start();
    else stop();
  });
})();
