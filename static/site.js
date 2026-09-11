// Shared by every page: the light / system / dark switch in the header.
// The inline snippet in each <head> applies the saved theme before first paint;
// this only wires up the buttons and marks the active one.
function applyTheme(mode) {
  if (mode === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = mode;
  for (const b of document.querySelectorAll("[data-theme-choice]")) {
    b.classList.toggle("is-active", b.dataset.themeChoice === mode);
  }
  try { localStorage.setItem("theme", mode); } catch {}
}

function initTheme() {
  let saved = "system";
  try { saved = localStorage.getItem("theme") || "system"; } catch {}
  applyTheme(saved);
  for (const b of document.querySelectorAll("[data-theme-choice]")) {
    b.addEventListener("click", () => applyTheme(b.dataset.themeChoice));
  }
}

document.addEventListener("DOMContentLoaded", initTheme);
