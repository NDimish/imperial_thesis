const toggle = document.getElementById("enabled-toggle");
const status = document.getElementById("status");

function render(enabled) {
  toggle.checked = enabled;
  status.textContent = enabled ? "Checklist + capture active" : "Off";
}

chrome.storage.sync.get({ captureEnabled: true }, (result) => {
  render(result.captureEnabled);
});

toggle.addEventListener("change", () => {
  chrome.storage.sync.set({ captureEnabled: toggle.checked });
  render(toggle.checked);
});
