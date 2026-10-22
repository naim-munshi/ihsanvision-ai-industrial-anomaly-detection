/* Shared account controls; session credentials remain in an HttpOnly cookie. */
(() => {
  const rawFetch = window.fetch.bind(window);
  let csrf = "";
  let ready;
  const sessionReady = new Promise(resolve => { ready = resolve; });
  window.ihsanSession = false;
  window.fetch = async (input, options = {}) => {
    const url = new URL(typeof input === "string" || input instanceof URL ? input : input.url, location.href);
    if (url.origin !== location.origin) return rawFetch(input, options);
    const method = (options.method || input.method || "GET").toUpperCase();
    if (!["GET", "HEAD", "OPTIONS"].includes(method)) await sessionReady;
    const headers = new Headers(options.headers || input.headers);
    if (csrf && !["GET", "HEAD", "OPTIONS"].includes(method)) headers.set("X-IhsanVision-CSRF", csrf);
    const response = await rawFetch(input, {...options, headers});
    if (response.status === 401 && window.ihsanSession) {
      window.ihsanSession = false;
      csrf = "";
      location.assign("/login");
    }
    return response;
  };
  document.addEventListener("DOMContentLoaded", async () => {
    try {
      const response = await rawFetch("/auth/me", {cache: "no-store"});
      if (!response.ok) return;
      const state = await response.json();
      if (!state.enabled) return;
      const bar = document.createElement("div");
      bar.style.cssText = "padding:12px 16px;margin:12px 0;border:1px solid #456985;border-radius:10px;color:#ecf3ff;background:#14243a;display:flex;gap:16px;align-items:center;flex-wrap:wrap";
      const label = document.createElement("span");
      bar.appendChild(label);
      const link = document.createElement("a");
      link.style.color = "#67d9c2";
      if (state.user) {
        window.ihsanSession = true;
        window.ihsanRole = state.user.role;
        csrf = state.user.csrf;
        label.textContent = "Signed in: " + state.user.username + " · " + state.user.role;
        link.href = "/login";
        link.textContent = "Sign out";
        link.addEventListener("click", async event => {
          event.preventDefault();
          const result = await window.fetch("/auth/logout", {method: "POST"});
          if (result.ok) location.replace("/login");
          else label.textContent = "Sign out failed. Please retry.";
        });
        const key = document.getElementById("api-key");
        if (key) {
          key.value = "";
          key.disabled = true;
          key.placeholder = "Authenticated with your account";
        }
        // History Lock also revokes the browser session, across all dashboard pages.
        const lock = document.getElementById("lock");
        if (lock) lock.addEventListener("click", () => link.click());
      } else {
        label.textContent = "Sign in to inspect and review with your operator account.";
        link.href = "/login";
        link.textContent = "Sign in";
      }
      bar.appendChild(link);
      document.querySelector("main").prepend(bar);
    } catch { /* Existing API-key workflow still handles connection errors. */ }
    finally { ready(); }
  });
})();
