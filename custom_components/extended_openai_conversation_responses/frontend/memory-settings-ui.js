import {bindConfigurationInputs} from "./configuration-inputs.js";
import {friendlySettingLabel, friendlySettingValue, settingSearchAliases} from "./management-setting-metadata.js";
import {settingBadgesMarkup} from "./management-decision-guidance.js";
import {saveBarMarkup} from "./unsaved-state.js";
const settingSearch = (label, description, key) => `${label} ${description} ${key} ${settingSearchAliases(key)}`.toLowerCase();

function option(panel, item, selected, key) {
  const value = typeof item === "string" ? item : item.value;
  const label = typeof item === "string" ? String(item).replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase()) : item.label;
  return `<option value="${panel._e(value)}" ${value === selected ? "selected" : ""}>${panel._e(friendlySettingValue(key, value) || label)}</option>`;
}

function select(panel, key, label, value, description) {
  const choices = panel._result?.options?.[key] || [];
  return `<div class="setting" data-setting data-field="${key}" data-search="${panel._e(settingSearch(label, description, key))}"><span class="setting-label-row"><label for="config-${key}">${panel._e(friendlySettingLabel(key) || label)}</label>${settingBadgesMarkup(panel, key, value)}</span><select id="config-${key}" data-memory-config="${key}">${choices.map((item) => option(panel, item, value, key)).join("")}</select><small>${panel._e(description)}</small><span class="field-error" data-error="${key}"></span></div>`;
}

function numberField(panel, key, label, value, description, min = null, max = null) {
  return `<div class="setting" data-setting data-field="${key}" data-search="${panel._e(settingSearch(label, description, key))}"><span class="setting-label-row"><label for="config-${key}">${panel._e(friendlySettingLabel(key) || label)}</label>${settingBadgesMarkup(panel, key, value)}</span><input id="config-${key}" data-memory-config="${key}" data-type="number" type="number" value="${panel._e(value)}" ${min === null ? "" : `min="${min}"`} ${max === null ? "" : `max="${max}"`}><small>${panel._e(description)}</small><span class="field-error" data-error="${key}"></span></div>`;
}

function textField(panel, key, label, value, description, disabled = false) {
  return `<div class="setting" data-setting data-field="${key}" data-search="${panel._e(settingSearch(label, description, key))}"><span class="setting-label-row"><label for="config-${key}">${panel._e(friendlySettingLabel(key) || label)}</label>${settingBadgesMarkup(panel, key, value)}</span><input id="config-${key}" data-memory-config="${key}" type="text" value="${panel._e(value || "")}" ${disabled ? "disabled" : ""}><small>${panel._e(description)}</small><span class="field-error" data-error="${key}"></span></div>`;
}

function saveBar(panel) {
  if (!panel._configDirty) return "";
  return saveBarMarkup({configuration: true, pending: Boolean(panel._configurationSaving)});
}

export function renderMemorySettings(panel) {
  const config = panel._draft || panel._result?.config || {};
  const hybrid = config.memory_retrieval_mode === "hybrid";
  return `<section class="page-intro"><div class="section-heading"><h1>Memory settings</h1><button type="button" class="secondary inline-route" data-page="data-memory" data-subsection="memories">Manage stored memories</button></div><p>Choose what the assistant may remember, how relevant memories are found, and whether household memories are available. <button type="button" class="guide-topic-link guide-link" data-guide-topic="memory">Learn more</button></p></section>
    <div class="content-card config-surface">
      <section id="config-memory" class="config-section" data-config-section data-search="memory persistent temporary short term long term automatic retrieval embeddings shared household">
        <div class="config-stack">
          <div class="setting-group"><div class="subheading"><h3>Personal memory</h3><p>Control durable memories and automatically expiring short-term details.</p></div>
            ${select(panel, "memory_mode", "Long-term memory", config.memory_mode, "Choose whether durable memories are off, saved only when explicitly requested, or may also be created automatically.")}
            ${select(panel, "temporary_memory", "Short-term memory", config.temporary_memory, "Choose how readily useful temporary details are remembered until they expire automatically.")}
          </div>
          <div class="setting-group"><div class="subheading"><h3>Retrieval</h3><p>Control which stored memories are supplied automatically and how relevance is calculated.</p></div>
            ${numberField(panel, "memory_auto_retrieve_limit", "Automatically include memories", config.memory_auto_retrieve_limit, "Select up to this many relevant memories when a new conversation starts. Set to 0 to use long-term memory only on demand.", 0, 10)}
            ${select(panel, "memory_retrieval_mode", "Memory retrieval", config.memory_retrieval_mode, "Lexical retrieval is local and dependency-free. Hybrid retrieval combines lexical matching with semantic embeddings.")}
            <div class="dependent ${hybrid ? "" : "is-disabled"}" data-memory-hybrid>${textField(panel, "memory_embedding_model", "Embedding model", config.memory_embedding_model || "text-embedding-3-small", "Used only for Hybrid retrieval. The configured provider must support embeddings.", !hybrid)}</div>
          </div>
          <div class="setting-group"><div class="subheading"><h3>Shared household memory</h3><p>Keep household-wide memory behavior separate from private user memories.</p></div>
            ${select(panel, "shared_memory_mode", "Shared household memory", config.shared_memory_mode, "Choose whether shared memories are disabled, saved only when explicitly requested, or may also be created automatically.")}
          </div>
        </div>
      </section>
      ${saveBar(panel)}<span id="memory-settings-save-anchor" class="sr-only"></span>
    </div>`;
}

export function bindMemorySettings(panel) {
  bindConfigurationInputs(panel);
}
