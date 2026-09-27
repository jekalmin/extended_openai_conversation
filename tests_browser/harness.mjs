import {createBackupTransferBackend} from "/tests_browser/backup-transfer-harness.mjs";
import {createStateBackend} from "/tests_browser/harness-state.mjs";

const params = new URLSearchParams(location.search);
const route = params.get("route") || "guide";
const isAdmin = params.get("admin") !== "0";
const predefine = params.get("predefine") === "1";
const agentCount = Math.max(1, Math.min(100, Number.parseInt(params.get("agents") || "1", 10) || 1));
const backend = createStateBackend({partialOverview: params.get("partial") === "1", failConfigurationOnce: params.get("fail_config_once") === "1"});
const backupTransfer = createBackupTransferBackend(backend);
const managementType = "extended_openai_conversation_responses/management";
const backupTransferType = "extended_openai_conversation_responses/management/backup_transfer";
const broadcastType = "extended_openai_conversation_responses/broadcast";
const calls = [];
history.replaceState({}, "", `/extended-openai/${route}`);

const hass = {
  config: {time_zone: "Europe/Dublin"},
  callWS: async (message) => {
    calls.push(structuredClone(message));
    if (message.type === managementType && message.action === "agents" && !message.section) {
      const primary = backend.agent();
      return {is_admin: isAdmin, agents: Array.from({length:agentCount}, (_, index) => index === 0 ? primary : {
        ...primary, entry_id:`scale-entry-${index}`, subentry_id:`scale-agent-${index}`, title:`Scale agent ${index}`,
      }), scopes: backend.scopes()};
    }
    if (message.type === backupTransferType) return backupTransfer(message);
    if (message.type === broadcastType && message.action === "snapshot") return {enabled: false, can_manage: isAdmin, catalog: {satellites: [], areas: []}, history: []};
    if (message.type !== managementType) return {};
    return backend.call(message);
  },
};

window.browserHarness = {calls, hass, getState: backend.state, resetState: backend.reset, windowErrors: [], rejections: []};
window.addEventListener("error", (event) => window.browserHarness.windowErrors.push(String(event.error || event.message)));
window.addEventListener("unhandledrejection", (event) => window.browserHarness.rejections.push(String(event.reason)));

let panel;
if (predefine) {
  panel = document.createElement("extended-openai-management-panel");
  panel.hass = hass;
  panel.route = {path: location.pathname};
  document.body.append(panel);
}
const frontendRoot = "/custom_components/extended_openai_conversation_responses/frontend/";
if (params.get("bundle") === "1") {
  const manifest = await (await fetch(`${frontendRoot}dist/manifest.json`)).json();
  const entry = Object.values(manifest).find((item) => item.isEntry && item.name === "management");
  await import(`${frontendRoot}dist/${entry.file}`);
} else {
  await import(`${frontendRoot}management-panel.js`);
}
if (!panel) {
  panel = document.createElement("extended-openai-management-panel");
  document.body.append(panel);
  panel.hass = hass;
}
window.browserHarness.panel = panel;
// Home Assistant supplies a new route property when its browser history changes.
window.addEventListener("popstate", () => { panel.route = {path: location.pathname}; });
