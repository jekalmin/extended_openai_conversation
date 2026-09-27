import {createBackupTransferBackend} from "/tests_browser/backup-transfer-harness.mjs";
import {createStateBackend} from "/tests_browser/harness-state.mjs";

const params = new URLSearchParams(location.search);
const route = params.get("route") || "guide";
const isAdmin = params.get("admin") !== "0";
const predefine = params.get("predefine") === "1";
const agentCount = Math.max(1, Math.min(100, Number.parseInt(params.get("agents") || "1", 10) || 1));
const backend = createStateBackend({partialOverview: params.get("partial") === "1", failConfigurationOnce: params.get("fail_config_once") === "1", seedConversations: params.get("seed_conversations") === "1"});
const backupTransfer = createBackupTransferBackend(backend);
const managementType = "extended_openai_conversation_responses/management";
const backupTransferType = "extended_openai_conversation_responses/management/backup_transfer";
const broadcastType = "extended_openai_conversation_responses/broadcast";
const requestDebugType = "extended_openai_conversation_responses/request_debug";
const modelCatalogType = "extended_openai_conversation_responses/model_catalog";
const credentialType = "extended_openai_conversation_responses/management/update_api_key";
const calls = [];
const broadcast = {enabled: false, history: []};
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
    if (message.type === requestDebugType) return backend.debugCall(message);
    if (message.type === broadcastType) {
      if (message.action === "snapshot") return {enabled: broadcast.enabled, can_manage: isAdmin, catalog: {satellites: [], areas: []}, history: structuredClone(broadcast.history)};
      if (message.action === "set_enabled") {
        if (!isAdmin || typeof message.enabled !== "boolean") throw new Error("Invalid EOAI Broadcast setting request");
        broadcast.enabled = message.enabled;
        return {enabled: broadcast.enabled};
      }
      if (message.action === "send") {
        if (!isAdmin || !String(message.message || "").trim() || (!message.whole_home && !message.entity_ids?.length)) throw new Error("Invalid EOAI Broadcast send request");
        const item = {message: message.message, status: "queued", entity_ids: message.entity_ids || []};
        broadcast.history.unshift(item);
        return structuredClone(item);
      }
      throw new Error(`Unsupported EOAI Broadcast fixture request: ${JSON.stringify(message)}`);
    }
    if (message.type === modelCatalogType) {
      if (!["lookup", "check", "update", "apply", "reset"].includes(message.action)) throw new Error(`Unsupported EOAI model catalogue fixture request: ${JSON.stringify(message)}`);
      if (message.model !== undefined && typeof message.model !== "string") throw new Error("Model catalogue model must be a string");
      return {catalog_version: 1, update_available: false, last_error: null, model_capabilities: {}, model_metadata: {}, catalog_models: [], reasoning_effort_options: ["low", "medium", "high"]};
    }
    if (message.type === credentialType) {
      if (!isAdmin || typeof message.entry_id !== "string" || !String(message.api_key || "").trim()) throw new Error("Invalid EOAI credential fixture request");
      return {status: "updated", entry_id: message.entry_id};
    }
    if (message.type !== managementType) {
      if (message.type?.startsWith("extended_openai_conversation_responses/")) throw new Error(`Unsupported EOAI fixture WebSocket type: ${JSON.stringify(message)}`);
      return {};
    }
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
