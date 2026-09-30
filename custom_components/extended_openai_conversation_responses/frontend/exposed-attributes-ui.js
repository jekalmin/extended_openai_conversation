import {applyTargetedConfigDirty} from "./management-state-safety.js";
const CONFIG_KEY = "exposed_entity_attributes";

const clonePreferences = (value = {}) => Object.fromEntries(
  Object.entries(value && typeof value === "object" && !Array.isArray(value) ? value : {})
    .filter(([reference, attributes]) => typeof reference === "string" && Array.isArray(attributes))
    .map(([reference, attributes]) => [reference, [...new Set(attributes.filter((name) => typeof name === "string" && name))].sort()]),
);

export function updateExposedAttributePreference(preferences, reference, attribute, enabled) {
  const result = clonePreferences(preferences);
  if (!reference || !attribute) return result;
  const selected = new Set(result[reference] || []);
  if (enabled) selected.add(attribute);
  else selected.delete(attribute);
  if (selected.size) result[reference] = [...selected].sort();
  else delete result[reference];
  return result;
}

export function removeExposedAttributePreference(preferences, reference) {
  const result = clonePreferences(preferences);
  delete result[reference];
  return result;
}

const selectedFor = (panel, reference, fallback = []) => {
  const configured = panel?._draft?.[CONFIG_KEY];
  return Array.isArray(configured?.[reference]) ? configured[reference] : fallback;
};

const catalogFor = (panel) => panel?._result?.exposed_attribute_catalog
  || panel?._configData?.exposed_attribute_catalog
  || null;

function attributeChoice(panel, entity, name, selected, missing = false) {
  const disabled = !entity.durable_selection_available;
  return `<label class="exposed-attribute-choice ${disabled ? "is-disabled" : ""}">
    <input type="checkbox" data-exposed-attribute data-reference="${panel._e(entity.reference || "")}" data-attribute="${panel._e(name)}" ${selected ? "checked" : ""} ${disabled ? "disabled" : ""}>
    <span><code>${panel._e(name)}</code>${missing ? `<small>Selected, but not present in the entity's current state</small>` : ""}</span>
  </label>`;
}

function entityEditor(panel, entity) {
  if (!entity) return "";
  const selected = new Set(selectedFor(panel, entity.reference, entity.selected_attributes || []));
  const available = Array.isArray(entity.attributes) ? entity.attributes : [];
  const missing = [...selected].filter((name) => !available.includes(name));
  const names = [...new Set([...available, ...missing])].sort();
  const identityNote = entity.durable_selection_available
    ? "Selections follow this Home Assistant entity across entity-ID renames. Values are read live only while the entity remains exposed to Assist."
    : "Home Assistant does not provide a stable entity-registry identity for this entity, so durable attribute selection is disabled to prevent preferences transferring if its entity ID is later reused.";
  return `<section class="exposed-attribute-editor" data-exposed-editor>
    <div class="exposed-editor-heading">
      <div><strong>${panel._e(entity.name || entity.entity_id)}</strong><small><code>${panel._e(entity.entity_id)}</code></small></div>
      <button type="button" class="secondary" data-close-exposed-editor>Close</button>
    </div>
    <p class="help">${panel._e(identityNote)}</p>
    ${names.length
      ? `<div class="exposed-attribute-grid">${names.map((name) => attributeChoice(panel, entity, name, selected.has(name), missing.includes(name))).join("")}</div>`
      : `<p class="help">This entity currently reports no state attributes.</p>`}
  </section>`;
}

function configuredEntity(panel, entity) {
  const selected = selectedFor(panel, entity.reference, entity.selected_attributes || []);
  if (!selected.length) return "";
  return `<article class="exposed-configured-entity" data-configured-exposed-entity="${panel._e(entity.entity_id)}">
    <div>
      <strong>${panel._e(entity.name || entity.entity_id)}</strong>
      <small><code>${panel._e(entity.entity_id)}</code></small>
      <small>${panel._e(selected.join(", "))}</small>
    </div>
    <div class="exposed-configured-actions">
      <button type="button" class="secondary" data-edit-exposed-entity="${panel._e(entity.entity_id)}">Edit</button>
      <button type="button" class="secondary remove-exposed-preference" data-reference="${panel._e(entity.reference || "")}">Remove</button>
    </div>
  </article>`;
}

function savedUnexposed(panel, item) {
  const selected = selectedFor(panel, item.reference, item.selected_attributes || []);
  return `<article class="exposed-saved-preference" data-saved-exposed-reference="${panel._e(item.reference)}"><div><strong>${panel._e(item.name || item.entity_id || "Unavailable entity")}</strong><small>${item.entity_id ? `<code>${panel._e(item.entity_id)}</code> · ` : ""}${panel._e(selected.join(", "))}</small><small>${item.registry_entry_exists ? "Saved preference is inactive because this entity is not currently exposed." : "The original registry entity no longer exists. The preference remains inert and will not transfer to a new entity that reuses its old entity ID."}</small></div><button type="button" class="secondary remove-exposed-preference" data-reference="${panel._e(item.reference)}">Remove saved preference</button></article>`;
}

export function renderExposedAttributeSettings(panel) {
  const catalog = catalogFor(panel);
  const catalogAvailable = catalog && typeof catalog === "object";
  const entities = Array.isArray(catalog?.entities) ? catalog.entities : [];
  const saved = Array.isArray(catalog?.saved_unexposed) ? catalog.saved_unexposed : [];
  const configured = entities.filter((entity) => selectedFor(panel, entity.reference, entity.selected_attributes || []).length);
  const editorEntity = entities.find((entity) => entity.entity_id === panel?._exposedAttributeEntityId) || null;
  const automaticContextEnabled = panel?._draft?.exposed_entities_enabled === true;
  return `<div class="exposed-attribute-settings" data-setting data-search="exposed entity attributes additional state context brightness color temperature live values">
    <style>
      .exposed-attribute-settings{display:grid;gap:14px;padding:16px 0 4px;border-top:1px solid var(--divider-color)}
      .exposed-attribute-heading{display:flex;justify-content:space-between;gap:18px;align-items:start}.exposed-attribute-heading h3,.exposed-attribute-heading p{margin:0}.exposed-attribute-heading p{margin-top:5px;color:var(--secondary-text-color);line-height:1.45}
      .exposed-picker-wrap{display:grid;gap:6px}.exposed-picker-wrap>label{font-weight:600}.exposed-picker-wrap small{color:var(--secondary-text-color);line-height:1.4}.exposed-picker-fallback{width:100%;min-height:42px}
      .exposed-attribute-editor{display:grid;gap:10px;padding:14px;border:1px solid var(--divider-color);border-radius:10px;background:var(--card-background-color)}
      .exposed-editor-heading{display:flex;align-items:start;justify-content:space-between;gap:16px}.exposed-editor-heading>div{display:grid;gap:2px}.exposed-editor-heading small{color:var(--secondary-text-color)}
      .exposed-attribute-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px 14px}.exposed-attribute-choice{display:grid;grid-template-columns:auto minmax(0,1fr);align-items:start;gap:9px;padding:7px 0;color:var(--primary-text-color)}.exposed-attribute-choice input{width:18px;min-height:18px;margin-top:1px}.exposed-attribute-choice span{display:grid;gap:2px;min-width:0}.exposed-attribute-choice code{overflow-wrap:anywhere}.exposed-attribute-choice.is-disabled{opacity:.55}
      .exposed-configured{display:grid;gap:8px}.exposed-configured h4{margin:0}.exposed-configured-list{display:grid;border:1px solid var(--divider-color);border-radius:10px;overflow:hidden}.exposed-configured-entity{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:12px 14px;border-bottom:1px solid var(--divider-color)}.exposed-configured-entity:last-child{border-bottom:0}.exposed-configured-entity>div:first-child{display:grid;gap:2px;min-width:0}.exposed-configured-entity small{color:var(--secondary-text-color);overflow-wrap:anywhere}.exposed-configured-actions{display:flex;gap:8px;flex-shrink:0}
      .exposed-saved{display:grid;gap:8px;margin-top:4px}.exposed-saved h4{margin:0}.exposed-saved-preference{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:12px;border:1px solid var(--divider-color);border-radius:9px}.exposed-saved-preference>div{display:grid;gap:3px;min-width:0}.exposed-saved-preference small{overflow-wrap:anywhere}
      .exposed-catalog-error{padding:12px 14px;border:1px solid var(--error-color);border-radius:9px}.exposed-catalog-error strong,.exposed-catalog-error p{margin:0}.exposed-catalog-error p{margin-top:4px}
      .exposed-inactive-notice{padding:12px 14px;border:1px solid var(--divider-color);border-radius:9px;background:var(--secondary-background-color)}.exposed-inactive-notice strong,.exposed-inactive-notice p{margin:0}.exposed-inactive-notice p{margin-top:4px;color:var(--secondary-text-color)}
      @media(max-width:680px){.exposed-attribute-grid{grid-template-columns:1fr}.exposed-saved-preference,.exposed-attribute-heading,.exposed-configured-entity,.exposed-editor-heading{display:grid}.exposed-saved-preference button,.exposed-configured-actions{width:100%}.exposed-configured-actions button{flex:1}}
    </style>
    <div class="exposed-attribute-heading"><div><h3>Additional entity attributes</h3><p>Select extra state attributes to include with automatic exposed-device context. Values are read live for each request, and only entities exposed to Assist are eligible.</p></div></div>
    ${!automaticContextEnabled ? `<div class="exposed-inactive-notice"><strong>Inactive while automatic entity context is off</strong><p>Your selections are preserved and can still be edited here, but Extended OpenAI will not add these attributes while you manage entity context manually in your prompt.</p></div>` : ""}
    ${!catalogAvailable
      ? `<div class="exposed-catalog-error" role="alert"><strong>Unable to load exposed entity attributes.</strong><p>The integration did not return the current Assist-exposed entity catalogue.</p></div>`
      : entities.length
        ? `<div class="exposed-picker-wrap">
            <label for="exposed-entity-picker">Add or edit an entity</label>
            <ha-entity-picker id="exposed-entity-picker" show-entity-id></ha-entity-picker>
            <select id="exposed-entity-picker-fallback" class="exposed-picker-fallback" aria-label="Select an Assist-exposed entity" hidden>
              <option value="">Select an entity…</option>
              ${entities.map((entity) => `<option value="${panel._e(entity.entity_id)}">${panel._e(entity.name || entity.entity_id)} (${panel._e(entity.entity_id)})</option>`).join("")}
            </select>
            <small>Only entities currently exposed to Assist are available here.</small>
          </div>
          ${editorEntity ? entityEditor(panel, editorEntity) : ""}
          <section class="exposed-configured"><h4>Configured entities</h4>
            ${configured.length
              ? `<div class="exposed-configured-list">${configured.map((entity) => configuredEntity(panel, entity)).join("")}</div>`
              : `<p class="help">No additional attributes are configured yet.</p>`}
          </section>`
        : `<p class="help">No entities are currently exposed to Assist.</p>`}
    ${saved.length ? `<section class="exposed-saved"><h4>Saved preferences not currently exposed</h4><p class="help">These preferences are retained for their original registry entities, but they cannot make an entity visible to the model.</p>${saved.map((item) => savedUnexposed(panel, item)).join("")}</section>` : ""}
    <span class="field-error" data-error="${CONFIG_KEY}"></span>
  </div>`;
}

function markConfigDirty(panel) {
  applyTargetedConfigDirty(panel, [CONFIG_KEY]);
}

function rerenderKeepingEditor(panel, entityId = null) {
  panel._exposedAttributeEntityId = entityId;
  panel._render?.();
}

function bindEntityPicker(panel, entities) {
  const root = panel?.shadowRoot;
  const picker = root?.querySelector("#exposed-entity-picker");
  const fallback = root?.querySelector("#exposed-entity-picker-fallback");
  if (!picker || !fallback) return;

  const entityIds = entities.map((entity) => entity.entity_id);
  const selectEntity = (entityId) => {
    const value = entityIds.includes(entityId) ? entityId : null;
    if (value) rerenderKeepingEditor(panel, value);
  };

  const activateNative = () => {
    picker.hidden = false;
    fallback.hidden = true;
    picker.includeEntities = entityIds;
    if (panel?._hass) picker.hass = panel._hass;
    picker.value = panel?._exposedAttributeEntityId || undefined;
  };

  if (globalThis.customElements?.get?.("ha-entity-picker")) activateNative();
  else {
    picker.hidden = true;
    fallback.hidden = false;
    fallback.value = panel?._exposedAttributeEntityId || "";
    globalThis.customElements?.whenDefined?.("ha-entity-picker").then(() => {
      if (picker.isConnected) activateNative();
    }).catch(() => {});
  }

  picker.addEventListener("value-changed", (event) => selectEntity(event.detail?.value));
  fallback.addEventListener("change", (event) => selectEntity(event.target.value));
}

export function bindExposedAttributeSettings(panel) {
  const root = panel?.shadowRoot;
  if (!root) return;
  const exposedToggle = root.querySelector('[data-config="exposed_entities_enabled"]');
  if (exposedToggle && exposedToggle.dataset.eocExposedRefreshBound === undefined) {
    exposedToggle.dataset.eocExposedRefreshBound = "";
    exposedToggle.addEventListener("change", () => queueMicrotask(() => {
      if (panel?._viewKey?.() === "assistant/prompt-context") {
        rerenderKeepingEditor(panel, panel._exposedAttributeEntityId || null);
      }
    }));
  }
  const catalog = catalogFor(panel);
  const entities = Array.isArray(catalog?.entities) ? catalog.entities : [];
  bindEntityPicker(panel, entities);

  root.querySelector("[data-close-exposed-editor]")?.addEventListener("click", () => rerenderKeepingEditor(panel, null));
  root.querySelectorAll("[data-edit-exposed-entity]").forEach((button) => button.addEventListener("click", () => {
    rerenderKeepingEditor(panel, button.dataset.editExposedEntity);
  }));

  root.querySelectorAll("[data-exposed-attribute]").forEach((input) => input.addEventListener("change", () => {
    panel._draft ||= {};
    panel._draft[CONFIG_KEY] = updateExposedAttributePreference(
      panel._draft[CONFIG_KEY], input.dataset.reference, input.dataset.attribute, input.checked,
    );
    markConfigDirty(panel);
    rerenderKeepingEditor(panel, panel._exposedAttributeEntityId);
  }));

  root.querySelectorAll(".remove-exposed-preference").forEach((button) => button.addEventListener("click", () => {
    panel._draft ||= {};
    panel._draft[CONFIG_KEY] = removeExposedAttributePreference(panel._draft[CONFIG_KEY], button.dataset.reference);
    markConfigDirty(panel);
    if (button.dataset.reference && entities.find((entity) => entity.reference === button.dataset.reference)?.entity_id === panel._exposedAttributeEntityId) {
      panel._exposedAttributeEntityId = null;
    }
    panel._render?.();
  }));
}
