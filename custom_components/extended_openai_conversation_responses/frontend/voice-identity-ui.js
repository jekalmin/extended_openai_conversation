const SHARED_SCOPE = "shared:household";
const UNRETAINED_SCOPE = "unretained";

const POLICY_LABELS = Object.freeze({
  unretained: "Do not retain personal data",
  shared: "Use shared household data",
  default_user: "Use the default user",
  device_mapping: "Use a device assignment",
});

const OWNER_TYPES = Object.freeze({
  unretained: "No retained personal data",
  shared: "Shared household",
  user: "Home Assistant user",
});


const e = (panel,value) => panel._e(String(value ?? ""));
const rawUserId = (value) => String(value || "").replace(/^user:/,"");
const panelHass = (panel) => panel?.hass || panel?._hass || null;

export function voiceUsers(panel) {
  // Kept as a compatibility/fallback catalogue for summaries and older callers.
  // The editable controls themselves use Home Assistant's native user picker.
  const scopes = panel?._baseScopes?.length ? panel._baseScopes : panel?._data?.scopes || [];
  const seen = new Set();
  const result = [];
  for (const scope of scopes) {
    if (scope?.scope_type !== "user") continue;
    const id = rawUserId(scope.scope_id);
    if (!id || seen.has(id)) continue;
    seen.add(id);
    result.push({id,name:String(scope.display_name || id)});
  }
  return result;
}

export function voiceUserLabel(panel,userId) {
  const id = rawUserId(userId);
  return voiceUsers(panel).find((user) => user.id === id)?.name || null;
}

export function deviceIdForSatellite(entries,entityId) {
  if (!entityId) return "";
  const entry = (Array.isArray(entries) ? entries : []).find((item) => item?.entity_id === entityId);
  return String(entry?.device_id || "");
}

export function satelliteForDeviceId(entries,deviceId) {
  if (!deviceId) return "";
  const entry = (Array.isArray(entries) ? entries : []).find((item) =>
    item?.device_id === deviceId && String(item?.entity_id || "").startsWith("assist_satellite."));
  return String(entry?.entity_id || "");
}

function policyLabel(value,fallback=false) {
  if (fallback && value === "device_mapping") return "Device mapping again (no retained data)";
  return POLICY_LABELS[value] || String(value || "").replaceAll("_"," ");
}

function policySelect(panel,key,label,value,help,fallback=false) {
  const options = panel?._result?.options?.[key] || [];
  const values = options.map((item) => typeof item === "string" ? item : item.value);
  if (value && !values.includes(value)) values.push(value);
  return `<div class="setting" data-field="${key}" data-setting><label for="config-${key}">${e(panel,label)}</label><select id="config-${key}" data-config="${key}">${values.map((item) => `<option value="${e(panel,item)}" ${item === value ? "selected" : ""}>${e(panel,policyLabel(item,fallback))}</option>`).join("")}</select><small>${e(panel,help)}</small><span class="field-error" data-error="${key}"></span></div>`;
}

function normalizedOwner(owner) {
  const value = String(owner || "");
  if (value === "shared" || value === SHARED_SCOPE) return {type:"shared",stored:SHARED_SCOPE};
  if (!value || value === UNRETAINED_SCOPE) return {type:"unretained",stored:UNRETAINED_SCOPE};
  return {type:"user",id:rawUserId(value),stored:value};
}

function mappingEntries(config) {
  const value = config?.voice_device_mappings;
  return value && typeof value === "object" && !Array.isArray(value) ? Object.entries(value) : [];
}

function nativeUserPicker(panel,id,value,cssClass="") {
  return `<ha-user-picker id="${e(panel,id)}" class="voice-native-picker ${e(panel,cssClass)}" data-user-picker data-initial-value="${e(panel,rawUserId(value))}"></ha-user-picker>`;
}

function mappingRow(panel,deviceId="",owner=UNRETAINED_SCOPE) {
  const normalized = normalizedOwner(owner);
  return `<article class="voice-mapping-row" data-voice-mapping-row>
    <div><label>Assist satellite</label><ha-entity-picker class="voice-native-picker voice-satellite-picker" data-initial-device-id="${e(panel,deviceId)}"></ha-entity-picker><input type="hidden" class="voice-device-id" value="${e(panel,deviceId)}"><small class="voice-picker-warning voice-satellite-warning" hidden></small></div>
    <div><label>Use retained data for<select class="voice-owner-type"><option value="unretained" ${normalized.type === "unretained" ? "selected" : ""}>${OWNER_TYPES.unretained}</option><option value="shared" ${normalized.type === "shared" ? "selected" : ""}>${OWNER_TYPES.shared}</option><option value="user" ${normalized.type === "user" ? "selected" : ""}>${OWNER_TYPES.user}</option></select></label><div class="voice-owner-user" ${normalized.type === "user" ? "" : "hidden"}>${nativeUserPicker(panel,"",normalized.id,"voice-owner-user-picker")}</div><input type="hidden" class="voice-mapping-owner" value="${e(panel,normalized.stored)}"><small class="voice-picker-warning voice-owner-warning" hidden></small></div>
    <button type="button" class="danger remove-voice-mapping">Remove</button>
  </article>`;
}

// Loaded only after the policy core has painted and device assignment is active.
export function renderVoiceMappings(panel) {
  const entries = mappingEntries(panel?._draft || panel?._result?.config || {});
  return `<section id="voice-mappings" class="voice-mappings-card" data-voice-mappings-card data-setting><div class="section-heading"><div><h3>Voice device assignments</h3><p>Assign an Assist satellite to one Home Assistant user, the shared household, or no retained personal data.</p></div><button type="button" class="secondary" id="add-voice-mapping">+ Add assignment</button></div><div id="voice-mapping-list" class="voice-mapping-list">${entries.length ? entries.map(([deviceId,owner]) => mappingRow(panel,deviceId,owner)).join("") : '<div class="voice-mapping-empty">No device assignments saved. If device mapping is selected above, the unmapped-device fallback will be used.</div>'}</div><span class="field-error" data-error="voice_device_mappings"></span></section>`;
}

export function voiceIdentitySummary(config={},users=[]) {
  const userName = (value) => users.find((user) => user.id === rawUserId(value))?.name || "the selected Home Assistant user";
  const target = (policy) => {
    if (policy === "shared") return "shared household data";
    if (policy === "default_user") return config.voice_default_user_id ? `the default user (${userName(config.voice_default_user_id)})` : "the default user, but none is selected — so no personal data is retained";
    if (policy === "device_mapping") return "a device assignment";
    return "no retained personal data";
  };
  const main = String(config.voice_scope_policy || "unretained");
  if (main !== "device_mapping") return `Unidentified voice requests use ${target(main)}.`;
  const count = mappingEntries(config).length;
  const fallback = String(config.voice_unmapped_policy || "unretained");
  return `Unidentified voice requests use ${count} saved device assignment${count === 1 ? "" : "s"}. Devices without an assignment use ${target(fallback === "device_mapping" ? "unretained" : fallback)}.`;
}

export function renderVoiceIdentity(panel) {
  const config = panel?._draft || panel?._result?.config || {};
  const entries = mappingEntries(config);
  const selectedUser = String(config.voice_default_user_id || "");
  return `<div class="config-section-heading"><p class="eyebrow">Voice & identity</p><p>Choose whose memories and conversation history may be used when Home Assistant does not identify the speaker.</p></div>
    <div class="voice-identity-flow"><div class="voice-flow-step"><strong>1 · Signed-in identity wins</strong><small>If Home Assistant supplies an authenticated user, that user's personal scope is used regardless of the settings below.</small></div><div class="voice-flow-step"><strong>2 · Otherwise use the voice policy</strong><small>Unidentified requests can use no retained personal data, shared household data, a default user, or a device assignment.</small></div><div class="voice-flow-step"><strong>3 · No identity guessing</strong><small>Device assignments use Home Assistant's source device ID only; room, presence, Bluetooth, and camera data are not used to guess a speaker.</small></div></div>
    <div class="notice on"><strong>Current unidentified voice behavior</strong><p id="voice-current-summary">${e(panel,voiceIdentitySummary(config,voiceUsers(panel)))}</p></div>
    <div class="voice-policy-grid"><section class="voice-policy-card"><h3>Unidentified voice requests</h3><p>Applies only when Home Assistant has not already attached a user to the request.</p>${policySelect(panel,"voice_scope_policy","Use retained data from",String(config.voice_scope_policy || "unretained"),"Choose the data owner for unidentified voice requests.")}</section><section class="voice-policy-card" data-voice-fallback-card><h3>Unmapped-device fallback</h3><p>Used only when device assignment is selected and the source device has no saved assignment.</p>${policySelect(panel,"voice_unmapped_policy","If the device is not assigned",String(config.voice_unmapped_policy || "unretained"),"Choose the safe fallback for an unidentified device.",true)}</section></div>
    <section class="voice-default-card" data-voice-default-card><h3>Default voice user</h3><p>Used only when an unidentified-voice policy explicitly chooses the default user.</p><div class="setting" data-field="voice_default_user_id" data-setting><label>Home Assistant user</label>${nativeUserPicker(panel,"config-voice_default_user_picker",selectedUser)}<input type="hidden" id="config-voice_default_user_id" data-config="voice_default_user_id" value="${e(panel,selectedUser)}"><small>Choose from Home Assistant's user list. The user ID is stored internally.</small><span class="field-error" data-error="voice_default_user_id"></span></div></section>
    <section id="voice-mappings" class="voice-mappings-card" data-voice-mappings-card data-setting><div class="section-heading"><div><h3>Voice device assignments</h3><p>Assign an Assist satellite to one Home Assistant user, the shared household, or no retained personal data.</p></div><button type="button" class="secondary" id="add-voice-mapping">+ Add assignment</button></div><div id="voice-mapping-list" class="voice-mapping-list">${entries.length ? entries.map(([deviceId,owner]) => mappingRow(panel,deviceId,owner)).join("") : '<div class="voice-mapping-empty">No device assignments saved. If device mapping is selected above, the unmapped-device fallback will be used.</div>'}</div><span class="field-error" data-error="voice_device_mappings"></span></section>`;
}

function setWarning(element,message="") {
  if (!element) return;
  element.textContent = message;
  element.hidden = !message;
}

async function entityRegistry(panel) {
  if (Array.isArray(panel.__voiceEntityRegistry)) return panel.__voiceEntityRegistry;
  if (!panel.__voiceEntityRegistryPromise) {
    const hass = panelHass(panel);
    panel.__voiceEntityRegistryPromise = hass?.callWS
      ? hass.callWS({type:"config/entity_registry/list"}).then((entries) => {
          panel.__voiceEntityRegistry = Array.isArray(entries) ? entries : [];
          return panel.__voiceEntityRegistry;
        }).catch(() => [])
      : Promise.resolve([]);
  }
  return panel.__voiceEntityRegistryPromise;
}

function configureUserPicker(panel,picker,value) {
  if (!picker) return;
  picker.hass = panelHass(panel);
  picker.value = rawUserId(value);
}

function configureEntityPicker(panel,picker,value="") {
  if (!picker) return;
  picker.hass = panelHass(panel);
  picker.value = value;
  picker.includeDomains = ["assist_satellite"];
  picker.allowCustomEntity = false;
}

function rowsToMapping(root) {
  const mapping = {};
  let duplicate = null;
  root.querySelectorAll("[data-voice-mapping-row]").forEach((row) => {
    const deviceId = row.querySelector(".voice-device-id")?.value.trim() || "";
    if (!deviceId) return;
    if (Object.prototype.hasOwnProperty.call(mapping,deviceId)) duplicate ||= deviceId;
    mapping[deviceId] = row.querySelector(".voice-mapping-owner")?.value || UNRETAINED_SCOPE;
  });
  return {mapping,duplicate};
}

function currentConfig(panel) {
  const root = panel.shadowRoot;
  const config = {...(panel._draft || panel._result?.config || {})};
  config.voice_scope_policy = root.querySelector('[data-config="voice_scope_policy"]')?.value || "unretained";
  config.voice_unmapped_policy = root.querySelector('[data-config="voice_unmapped_policy"]')?.value || "unretained";
  config.voice_default_user_id = root.querySelector('[data-config="voice_default_user_id"]')?.value || "";
  config.voice_device_mappings = rowsToMapping(root).mapping;
  return config;
}

function updateDependencies(panel) {
  const root = panel.shadowRoot;
  const config = currentConfig(panel);
  const mappingActive = config.voice_scope_policy === "device_mapping";
  const defaultActive = config.voice_scope_policy === "default_user" || (mappingActive && config.voice_unmapped_policy === "default_user");
  for (const [selector,active] of [["[data-voice-fallback-card]",mappingActive],["[data-voice-default-card]",defaultActive],["[data-voice-mappings-card]",mappingActive]]) {
    const card = root.querySelector(selector);
    card?.classList.toggle("is-disabled",!active);
    card?.querySelectorAll("input,select,button,ha-user-picker,ha-entity-picker").forEach((control) => { control.disabled = !active; });
  }
  if (mappingActive) root.querySelectorAll("[data-voice-mapping-row]").forEach((row) => updateOwnerPicker(row,true));
  const summary = root.querySelector("#voice-current-summary");
  if (summary) summary.textContent = voiceIdentitySummary(config,voiceUsers(panel));
}

function syncMappings(panel) {
  const root = panel.shadowRoot;
  const card = root.querySelector("#voice-mappings");
  if (!card) return;
  const {mapping,duplicate} = rowsToMapping(root);
  const error = root.querySelector('[data-error="voice_device_mappings"]');
  if (error) error.textContent = duplicate ? `This Assist satellite is assigned more than once.` : "";
  card.value = duplicate ? JSON.stringify(`duplicate device id: ${duplicate}`) : JSON.stringify(mapping,null,2);
  card.dispatchEvent(new Event("input",{bubbles:true}));
  updateDependencies(panel);
}

function updateOwnerPicker(row,mappingActive=true) {
  const type = row.querySelector(".voice-owner-type")?.value || "unretained";
  const holder = row.querySelector(".voice-owner-user");
  const picker = row.querySelector(".voice-owner-user-picker");
  if (holder) holder.hidden = type !== "user";
  if (picker) picker.disabled = !mappingActive || type !== "user";
}

function syncOwner(row) {
  const type = row.querySelector(".voice-owner-type")?.value || "unretained";
  const hidden = row.querySelector(".voice-mapping-owner");
  const picker = row.querySelector(".voice-owner-user-picker");
  const warning = row.querySelector(".voice-owner-warning");
  if (!hidden) return;
  if (type === "shared") {
    hidden.value = SHARED_SCOPE;
    setWarning(warning);
  } else if (type === "unretained") {
    hidden.value = UNRETAINED_SCOPE;
    setWarning(warning);
  } else {
    const userId = rawUserId(picker?.value || "");
    if (userId) {
      hidden.value = `user:${userId}`;
      setWarning(warning);
    } else {
      hidden.value = UNRETAINED_SCOPE;
      setWarning(warning,"Choose a Home Assistant user. Until then, no personal data is retained for this assignment.");
    }
  }
}

async function bindSatellitePicker(panel,row) {
  const picker = row.querySelector(".voice-satellite-picker");
  const hidden = row.querySelector(".voice-device-id");
  const warning = row.querySelector(".voice-satellite-warning");
  if (!picker || !hidden) return;
  configureEntityPicker(panel,picker);
  const storedDeviceId = hidden.value.trim();
  const agentId = panel._agentId;
  const entries = await entityRegistry(panel);
  if (!row.isConnected || panel._agentId !== agentId || panel._viewKey?.() !== "assistant/voice") return;
  const entityId = satelliteForDeviceId(entries,storedDeviceId);
  if (entityId) {
    picker.value = entityId;
    setWarning(warning);
  } else if (storedDeviceId) {
    setWarning(warning,`Saved device is unavailable in Home Assistant (${storedDeviceId}). It will remain saved until you replace or remove it.`);
  }
  picker.addEventListener("value-changed",async (event) => {
    const selectedEntity = String(event?.detail?.value || picker.value || "");
    picker.value = selectedEntity;
    if (!selectedEntity) {
      hidden.value = "";
      setWarning(warning);
      syncMappings(panel);
      return;
    }
    const registry = await entityRegistry(panel);
    if (!row.isConnected || panel._agentId !== agentId || panel._viewKey?.() !== "assistant/voice") return;
    const deviceId = deviceIdForSatellite(registry,selectedEntity);
    if (!deviceId) {
      setWarning(warning,"That Assist satellite is not linked to a Home Assistant device, so it cannot be used for a device assignment.");
      return;
    }
    hidden.value = deviceId;
    setWarning(warning);
    syncMappings(panel);
  });
}

function bindRows(panel) {
  const root = panel.shadowRoot;
  root.querySelectorAll("[data-voice-mapping-row]").forEach((row) => {
    if (row.dataset.voiceBound !== undefined) return;
    row.dataset.voiceBound = "";
    const normalized = normalizedOwner(row.querySelector(".voice-mapping-owner")?.value);
    const ownerPicker = row.querySelector(".voice-owner-user-picker");
    configureUserPicker(panel,ownerPicker,normalized.id || "");
    bindSatellitePicker(panel,row);
    row.querySelector(".voice-owner-type")?.addEventListener("change",() => {
      updateOwnerPicker(row,true);
      syncOwner(row);
      syncMappings(panel);
    });
    ownerPicker?.addEventListener("value-changed",(event) => {
      ownerPicker.value = rawUserId(event?.detail?.value || "");
      syncOwner(row);
      syncMappings(panel);
    });
    row.querySelector(".remove-voice-mapping")?.addEventListener("click",() => {
      row.remove();
      if (!root.querySelector("[data-voice-mapping-row]")) root.querySelector("#voice-mapping-list").innerHTML = '<div class="voice-mapping-empty">No device assignments saved. If device mapping is selected above, the unmapped-device fallback will be used.</div>';
      syncMappings(panel);
    });
    updateOwnerPicker(row,true);
  });
}

export function bindVoiceIdentity(panel) {
  const root = panel.shadowRoot;
  const mappings = root.querySelector("#voice-mappings");
  if (mappings) mappings.value = JSON.stringify(rowsToMapping(root).mapping,null,2);
  const defaultPicker = root.querySelector("#config-voice_default_user_picker");
  const defaultHidden = root.querySelector('[data-config="voice_default_user_id"]');
  configureUserPicker(panel,defaultPicker,defaultHidden?.value || "");
  defaultPicker?.addEventListener("value-changed",(event) => {
    const value = rawUserId(event?.detail?.value || "");
    defaultPicker.value = value;
    if (defaultHidden) {
      defaultHidden.value = value;
      defaultHidden.dispatchEvent(new Event("input",{bubbles:true}));
      defaultHidden.dispatchEvent(new Event("change",{bubbles:true}));
    }
    updateDependencies(panel);
  });
  root.querySelector('[data-config="voice_scope_policy"]')?.addEventListener("change",() => updateDependencies(panel));
  root.querySelector('[data-config="voice_unmapped_policy"]')?.addEventListener("change",() => updateDependencies(panel));
  root.querySelector("#add-voice-mapping")?.addEventListener("click",() => {
    const list = root.querySelector("#voice-mapping-list");
    list.querySelector(".voice-mapping-empty")?.remove();
    list.insertAdjacentHTML("beforeend",mappingRow(panel));
    bindRows(panel);
    updateDependencies(panel);
    list.querySelector("[data-voice-mapping-row]:last-child .voice-satellite-picker")?.focus?.();
  });
  bindRows(panel);
  updateDependencies(panel);
}

export function bindVoiceMappings(panel) {
  const root = panel.shadowRoot;
  const mappings = root.querySelector("#voice-mappings");
  if (!mappings) return;
  mappings.value = JSON.stringify(rowsToMapping(root).mapping, null, 2);
  root.querySelector("#add-voice-mapping")?.addEventListener("click", () => {
    const list = root.querySelector("#voice-mapping-list");
    list.querySelector(".voice-mapping-empty")?.remove();
    list.insertAdjacentHTML("beforeend", mappingRow(panel));
    bindRows(panel);
    updateDependencies(panel);
    list.querySelector("[data-voice-mapping-row]:last-child .voice-satellite-picker")?.focus?.();
  });
  bindRows(panel);
  updateDependencies(panel);
}

export {SHARED_SCOPE,UNRETAINED_SCOPE,POLICY_LABELS};
