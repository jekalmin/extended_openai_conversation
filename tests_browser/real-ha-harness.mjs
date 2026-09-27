const params = new URLSearchParams(location.search);
const route = params.get("route") || "assistant/basics";
const backendUrl = params.get("backend");
if (!backendUrl) throw new Error("Real HA browser fixture requires a backend URL");

history.replaceState({}, "", `/extended-openai/${route}`);

// Preserve the wire trace across page refreshes in a single acceptance journey.
// Every Playwright test gets a fresh browser context, so journeys stay isolated.
let calls;
try {
  calls = JSON.parse(sessionStorage.getItem("eocRealHaCalls") || "[]");
  if (!Array.isArray(calls)) throw new TypeError("Invalid browser call trace");
} catch (_err) {
  calls = [];
  sessionStorage.removeItem("eocRealHaCalls");
}
const hass = {
  config: {time_zone: "Europe/Dublin"},
  callWS: async (message) => {
    const trace = structuredClone(message);
    // Keep transfer payload shape without filling sessionStorage with base64 chunks.
    if (trace.type?.endsWith("/backup_transfer") && trace.data) {
      trace.data = Object.fromEntries(Object.keys(trace.data).map((key) => [key, true]));
    }
    calls.push(trace);
    sessionStorage.setItem("eocRealHaCalls", JSON.stringify(calls));
    const response = await fetch(backendUrl, {
      method: "POST",
      body: JSON.stringify(message),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload?.message || payload?.error?.message || `Home Assistant WebSocket bridge returned ${response.status}`);
    return payload;
  },
};

window.browserHarness = {calls, hass, windowErrors: [], rejections: []};
window.addEventListener("error", (event) => window.browserHarness.windowErrors.push(String(event.error || event.message)));
window.addEventListener("unhandledrejection", (event) => window.browserHarness.rejections.push(String(event.reason)));

if (params.get("bundle") === "1") {
  const frontendRoot = "/custom_components/extended_openai_conversation_responses/frontend/";
  const manifest = await (await fetch(`${frontendRoot}dist/manifest.json`)).json();
  const entry = Object.values(manifest).find((item) => item.isEntry && item.name === "management");
  await import(`${frontendRoot}dist/${entry.file}`);
} else {
  await import("/custom_components/extended_openai_conversation_responses/frontend/management-panel.js");
}
const panel = document.createElement("extended-openai-management-panel");
document.body.append(panel);
panel.hass = hass;
window.browserHarness.panel = panel;
