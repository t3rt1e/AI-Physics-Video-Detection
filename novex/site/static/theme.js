/* Runs before paint. A saved choice wins; otherwise follow the OS. */
(() => {
  const system = matchMedia("(prefers-color-scheme: dark)");
  let preference = null;
  try { preference = localStorage.getItem("novex-theme"); } catch (_) { /* private browsing */ }
  const apply = (theme) => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    const dark = theme === "dark", button = document.getElementById("theme-toggle");
    if (button) {
      button.setAttribute("aria-label", `Switch to ${dark ? "light" : "dark"} mode`);
      button.title = button.getAttribute("aria-label");
    }
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", dark ? "#0d0f12" : "#f4f3ef");
    window.dispatchEvent(new Event("themechange"));
  };
  apply(preference === "light" || preference === "dark" ? preference : system.matches ? "dark" : "light");
  system.addEventListener("change", () => { if (!preference) apply(system.matches ? "dark" : "light"); });
  document.addEventListener("DOMContentLoaded", () => {
    apply(document.documentElement.dataset.theme);
    document.getElementById("theme-toggle").addEventListener("click", () => {
      preference = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      try { localStorage.setItem("novex-theme", preference); } catch (_) { /* still works for this visit */ }
      apply(preference);
    });
  });
})();
