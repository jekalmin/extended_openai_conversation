const escape = (panel, value) => panel._e(String(value ?? ""));
const rawUserId = (value) => String(value || "").replace(/^user:/, "");
const policyLabels = {
  unretained: "Do not retain personal data",
  shared: "Use shared household data",
  default_user: "Use the default user",
  device_mapping: "Use a device assignment",
};

function users(panel) {
  const scopes = panel?._baseScopes?.length ? panel._baseScopes : panel?._data?.scopes || [];
  return scopes.filter((scope) => scope?.scope_type === "user")
    .map((scope) => ({id: rawUserId(scope.scope_id), name: scope.display_name || rawUserId(scope.scope_id)}));
}

function summary(config, catalogue) {
  const target = (policy) => {
    if (policy === "shared") return "shared household data";
    if (policy === "default_user") {
      if (!config.voice_default_user_id) return "the default user, but none is selected — so no personal data is retained";
      const name = catalogue.find((user) => user.id === rawUserId(config.voice_default_user_id))?.name || "the selected Home Assistant user";
      return `the default user (${name})`;
    }
    if (policy === "device_mapping") return "a device assignment";
    return "no retained personal data";
  };
  const policy = config.voice_scope_policy || "unretained";
  if (policy !== "device_mapping") return `Unidentified voice requests use ${target(policy)}.`;
  const count = Object.keys(config.voice_device_mappings || {}).length;
  const fallback = config.voice_unmapped_policy || "unretained";
  return `Unidentified voice requests use ${count} saved device assignment${count === 1 ? "" : "s"}. Devices without an assignment use ${target(fallback === "device_mapping" ? "unretained" : fallback)}.`;
}

function policySelect(panel, key, label, value, help, fallback = false) {
  const values = (panel?._result?.options?.[key] || ["unretained", "shared", "default_user", ...(fallback ? [] : ["device_mapping"])]).map((item) => typeof item === "string" ? item : item.value);
  if (value && !values.includes(value)) values.push(value);
  return `<div class="setting" data-field="${key}" data-setting><label for="config-${key}">${escape(panel, label)}</label><select id="config-${key}" data-config="${key}">${values.map((item) => `<option value="${escape(panel, item)}" ${item === value ? "selected" : ""}>${escape(panel, fallback && item === "device_mapping" ? "Device mapping again (no retained data)" : policyLabels[item] || item.replaceAll("_", " "))}</option>`).join("")}</select><small>${escape(panel, help)}</small><span class="field-error" data-error="${key}"></span></div>`;
}

export function renderVoiceIdentityCore(panel) {
  const config = panel?._draft || panel?._result?.config || {};
  const selectedUser = String(config.voice_default_user_id || "");
  const mapping = config.voice_scope_policy === "device_mapping";
  const defaultActive = config.voice_scope_policy === "default_user" || (mapping && config.voice_unmapped_policy === "default_user");
  const fallbackCard = `<section class="voice-policy-card" data-voice-fallback-card><h3>Unmapped-device fallback</h3><p>Used only when device assignment is selected and the source device has no saved assignment.</p>${policySelect(panel, "voice_unmapped_policy", "If the device is not assigned", String(config.voice_unmapped_policy || "unretained"), "Choose the safe fallback for an unidentified device.", true)}</section>`;
  const defaultCard = `<section class="voice-default-card" data-voice-default-card><h3>Default voice user</h3><p>Used only when an unidentified-voice policy explicitly chooses the default user.</p><div class="setting" data-field="voice_default_user_id" data-setting><label>Home Assistant user</label><ha-user-picker id="config-voice_default_user_picker" class="voice-native-picker" data-user-picker></ha-user-picker><input type="hidden" id="config-voice_default_user_id" data-config="voice_default_user_id" value="${escape(panel, selectedUser)}"><small>Choose a Home Assistant user.</small><span class="field-error" data-error="voice_default_user_id"></span></div></section>`;
  return `<div class="config-section-heading"><p class="eyebrow">Voice & identity</p><p>Choose whose memories and conversation history may be used when Home Assistant does not identify the speaker.</p></div>
    <p class="help">Authenticated Home Assistant identity takes precedence over these settings.</p>
    <details class="eoc-details-base voice-identity-help"><summary>How voice identity is resolved</summary><div class="voice-identity-flow"><div class="voice-flow-step"><strong>1 · Signed-in identity wins</strong><small>If Home Assistant supplies an authenticated user, that user's personal scope is used regardless of the settings below.</small></div><div class="voice-flow-step"><strong>2 · Otherwise use the voice policy</strong><small>Unidentified requests can use no retained personal data, shared household data, a default user, or a device assignment.</small></div><div class="voice-flow-step"><strong>3 · No identity guessing</strong><small>Device assignments use Home Assistant's source device ID only; room, presence, Bluetooth, and camera data are not used to guess a speaker.</small></div></div></details>
    <div class="notice on"><strong>Current unidentified voice behavior</strong><p id="voice-current-summary">${escape(panel, summary(config, users(panel)))}</p></div>
    <div class="voice-policy-grid" data-voice-policy-grid><section class="voice-policy-card"><h3>Unidentified voice requests</h3>${policySelect(panel, "voice_scope_policy", "Use retained data from", String(config.voice_scope_policy || "unretained"), "Choose the data owner for unidentified voice requests.")}</section>${mapping ? fallbackCard : ""}</div>
    <div data-voice-default-host>${defaultActive ? defaultCard : ""}</div>
    <details class="eoc-details-base voice-inactive-settings" data-voice-inactive-settings ${mapping && defaultActive ? "hidden" : ""}><summary>Saved inactive voice settings</summary><p class="help">These saved values are retained, but are used only when the voice policy above selects them.</p><div data-voice-inactive-host>${mapping ? "" : fallbackCard}${defaultActive ? "" : defaultCard}</div></details>
    <div data-voice-mapping-feature ${mapping ? 'style="min-height:96px"' : ""}>${mapping ? '<section class="voice-mappings-card"><h3>Voice device assignments</h3><p class="help">Loading saved assignments…</p></section>' : ""}</div>`;
}

function updatePolicy(panel) {
  const root = panel.shadowRoot;
  const policy = root.querySelector('[data-config="voice_scope_policy"]')?.value || "unretained";
  const fallback = root.querySelector('[data-config="voice_unmapped_policy"]')?.value || "unretained";
  const defaultUser = root.querySelector('[data-config="voice_default_user_id"]')?.value || "";
  const mapping = policy === "device_mapping";
  const defaultActive = policy === "default_user" || (mapping && fallback === "default_user");
  for (const [selector, active] of [["[data-voice-fallback-card]", mapping], ["[data-voice-default-card]", defaultActive]]) {
    const card = root.querySelector(selector);
    card?.classList.toggle("is-disabled", !active);
    card?.querySelectorAll("input,select,button,ha-user-picker").forEach((control) => { control.disabled = !active; });
    const host = root.querySelector(!active ? "[data-voice-inactive-host]"
      : selector === "[data-voice-fallback-card]" ? "[data-voice-policy-grid]" : "[data-voice-default-host]");
    // Move the existing picker/control, rather than remounting or clearing it.
    if (card && host && card.parentElement !== host) host.append(card);
  }
  root.querySelector("[data-voice-inactive-settings]").hidden = mapping && defaultActive;
  const text = root.querySelector("#voice-current-summary");
  if (text) text.textContent = summary({...panel._draft, voice_scope_policy: policy, voice_unmapped_policy: fallback, voice_default_user_id: defaultUser}, users(panel));
  const mappingFeature = root.querySelector("[data-voice-mapping-feature]");
  if (mappingFeature) {
    mappingFeature.hidden = !mapping;
    mappingFeature.querySelectorAll("input,select,button,ha-user-picker,ha-entity-picker")
      .forEach((control) => { control.disabled = !mapping; });
  }
  if (mapping) void hydrateMapping(panel);
}

async function hydrateMapping(panel) {
  const target = panel.shadowRoot?.querySelector("[data-voice-mapping-feature]");
  if (!target || target.dataset.loading || target.dataset.ready) return;
  target.dataset.loading = "";
  if (!target.firstElementChild) {
    target.style.minHeight = "96px";
    target.innerHTML = '<section class="voice-mappings-card"><h3>Voice device assignments</h3><p class="help">Loading saved assignments…</p></section>';
  }
  const agentId = panel._agentId;
  try {
    const feature = await import("./voice-identity-ui.js");
    if (!target.isConnected || panel._agentId !== agentId || panel._viewKey?.() !== "assistant/voice"
        || panel.shadowRoot.querySelector('[data-config="voice_scope_policy"]')?.value !== "device_mapping") return;
    target.innerHTML = feature.renderVoiceMappings(panel);
    target.style.minHeight = "";
    target.dataset.ready = "";
    feature.bindVoiceMappings(panel);
  } catch (error) {
    if (target.isConnected && panel._agentId === agentId) target.querySelector(".help").textContent = `Unable to load device assignments: ${error.message || String(error)}`;
  } finally {
    if (!target.dataset.ready) delete target.dataset.loading;
  }
}

export function bindVoiceIdentityCore(panel) {
  const root = panel.shadowRoot;
  const picker = root.querySelector("#config-voice_default_user_picker");
  const hidden = root.querySelector('[data-config="voice_default_user_id"]');
  if (picker) {
    picker.hass = panel.hass || panel._hass;
    picker.value = rawUserId(hidden?.value);
    picker.addEventListener("value-changed", (event) => {
      const value = rawUserId(event.detail?.value || "");
      picker.value = value;
      hidden.value = value;
      hidden.dispatchEvent(new Event("input", {bubbles: true}));
      hidden.dispatchEvent(new Event("change", {bubbles: true}));
      updatePolicy(panel);
    });
  }
  root.querySelector('[data-config="voice_scope_policy"]')?.addEventListener("change", () => updatePolicy(panel));
  root.querySelector('[data-config="voice_unmapped_policy"]')?.addEventListener("change", () => updatePolicy(panel));
  updatePolicy(panel);
}
