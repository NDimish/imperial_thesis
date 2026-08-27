// Relays capture requests from content.js to the local server. Routed
// through the background service worker rather than fetched directly from
// content.js because a content script's fetch is subject to the target
// page's own Content-Security-Policy, which will typically block a request
// to an origin (localhost:5000) the page itself never whitelisted -- the
// service worker isn't a page and isn't bound by that CSP. See
// "oogle api/ambient-ai-capture-plan.html" section 3 for the full reasoning.
const API_BASE = "http://localhost:5000";
// Dev-only static key matching the server's default (server.py's
// CAPTURE_API_KEY env var) -- fine for a localhost research prototype with
// one machine reporting in; swap for a per-install token before this ever
// touches real patient data (see the plan doc's §3 callout).
const API_KEY = "dev-only-key-replace-me";

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "CHECK_NOTE") {
    fetch(`${API_BASE}/api/check`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
      body: JSON.stringify({ sections: message.payload.sections }),
    })
      .then((r) => r.json())
      .then((data) => sendResponse({ ok: true, items: data.items || [] }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true; // keep the message channel open for the async sendResponse
  }

  if (message.type === "NOTE_FINALIZED") {
    fetch(`${API_BASE}/events`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
      body: JSON.stringify(message.payload),
    })
      .then((r) => r.json())
      .then((data) => sendResponse({ ok: true, ...data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true;
  }

  return false;
});
