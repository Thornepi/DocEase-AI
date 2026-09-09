let t10 = null;
let t30 = null;

function showLoading(message) {
  const overlay = document.getElementById("loadingOverlay");
  const msg = document.getElementById("loadingMessage");
  if (!overlay) return;

  if (msg && message) msg.textContent = message;
  overlay.classList.remove("hidden");
}

function hideLoading() {
  const overlay = document.getElementById("loadingOverlay");
  if (!overlay) return;
  overlay.classList.add("hidden");
}

function clearTimers() {
  if (t10) clearTimeout(t10);
  if (t30) clearTimeout(t30);
  t10 = null;
  t30 = null;
}

function setupUploadLoader() {
  const form = document.getElementById("uploadForm");
  const btn = document.getElementById("uploadBtn");
  const fileInput = document.getElementById("fileInput");

  if (!form) return;

  form.addEventListener("submit", () => {
    if (btn) btn.disabled = true;

    let baseMsg = "Processing your document… This may take a few seconds.";
    const f = fileInput && fileInput.files ? fileInput.files[0] : null;

    if (f) {
      const name = (f.name || "").toLowerCase();
      const mb = (f.size / (1024 * 1024)).toFixed(1);

      if (name.endsWith(".pdf")) baseMsg = `Processing PDF… (file size: ${mb} MB)`;
      else baseMsg = `Processing image… (file size: ${mb} MB)`;
    }

    showLoading(baseMsg);

    clearTimers();

    t10 = setTimeout(() => {
      showLoading("Still working… Scanned/compressed documents can take longer. Please wait.");
    }, 10000);

    t30 = setTimeout(() => {
      showLoading("This is taking longer than usual (likely low-quality scan/OCR or high demand). Please don’t refresh.");
    }, 30000);
  });

  window.addEventListener("pageshow", () => {
    clearTimers();
    if (btn) btn.disabled = false;
    hideLoading();
  });
}

function setupSensitiveToggles() {
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".toggle-sensitive");
    if (!btn) return;

    const row = btn.closest(".info-row");
    if (!row) return;

    const span = row.querySelector(".sensitive-value");
    if (!span) return;

    const masked = JSON.parse(span.dataset.masked || '""');
    const full = JSON.parse(span.dataset.full || '""');

    const isShowing = span.textContent === full;

    if (isShowing) {
      span.textContent = masked;
      btn.textContent = "Show";
    } else {
      span.textContent = full;
      btn.textContent = "Hide";
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setupUploadLoader();
  setupSensitiveToggles();
});