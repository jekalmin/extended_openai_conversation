const ROUTE_STYLE = `.function-groups-help{margin-bottom:18px}.tool-search{display:block;margin-bottom:18px}.function-groups{display:grid;gap:18px}.function-group-card{border:1px solid var(--divider-color);border-radius:12px;padding:18px;background:var(--card-background-color)}.function-group-heading{display:flex;justify-content:space-between;align-items:start;gap:20px}.function-group-heading h3,.tool-title h4{margin:0}.function-group-heading p{margin:7px 0;color:var(--secondary-text-color);line-height:1.45}.function-group-heading code{font-size:12px;color:var(--secondary-text-color)}.availability-badge,.function-count{display:inline-flex;padding:4px 9px;border-radius:999px;background:color-mix(in srgb,var(--success-color,#0f9d58) 15%,transparent);color:var(--success-color,#0f9d58);font-size:11px;font-weight:600}.availability-badge.on-demand{background:color-mix(in srgb,var(--primary-color) 14%,transparent);color:var(--primary-color)}.function-count{background:var(--secondary-background-color);color:var(--secondary-text-color)}.function-group-card details{margin-top:14px;padding-top:10px}.function-group-card summary{min-height:40px}.function-group-card .tool-list{padding-top:10px}.group-dialog{width:min(820px,calc(100vw - 28px))}.group-dialog-body{max-height:calc(100vh - 170px)}.group-functions-fieldset{margin:0;padding-top:18px}.group-function-choices{display:grid;gap:8px;max-height:300px;overflow:auto;margin-top:12px}.group-function-choice{display:grid;grid-template-columns:auto 1fr;align-items:start;gap:11px;padding:11px;border:1px solid var(--divider-color);border-radius:9px;color:var(--primary-text-color);cursor:pointer}.group-function-choice.is-disabled{opacity:.58}.group-function-choice:hover{background:var(--secondary-background-color)}.group-function-choice input{width:18px;min-height:18px;margin-top:2px}.group-function-choice span{display:grid;gap:3px}.group-function-choice small{overflow-wrap:anywhere}@media(max-width:679px){.group-dialog{width:calc(100vw - 12px)}.function-group-heading{display:grid}.group-function-choices{max-height:34vh}}`;
function ensureRouteStyle(panel) {
  if (typeof document === "undefined") return;
  if (panel.shadowRoot.querySelector("style[data-eoc-feature-style=\"function-groups\"]")) return;
  const style = document.createElement("style");
  style.dataset.eocFeatureStyle = "function-groups";
  style.textContent = ROUTE_STYLE;
  panel.shadowRoot.append(style);
}

import {readConfigurationDraft as readConfig} from "./configuration-controls.js";
import {bindConfigurationInputs, updateConfigurationControl} from "./configuration-inputs.js";
import {adoptKeyedElements, elementFromMarkup, keyedElement, placeChildren, pruneKeys, setAttribute, setText} from "./keyed-collection.js";
import {modelFieldPresentation, modelFieldNotes} from "./agent-config-model-presentation.js";
import {friendlySettingLabel, friendlySettingValue, settingSearchAliases} from "./management-setting-metadata.js";
import {settingBadgesMarkup} from "./management-decision-guidance.js";
import {bindSingleRequestSave} from "./management-actions.js";
import {saveBarMarkup} from "./unsaved-state.js";
import {modelDataControls, bindModelDataControls} from "./model-catalog.js";
import { haToolName, isHALlmTool, renderHAToolCard, toolDescription } from "./ha-llm-tools-list.js";
import { bindHelp, helpButton, helpPopover } from "./agent-config-help.js";
import {getToolYamlEditor} from "./tool-yaml-editor-adapter.js";

const clone = (value) => JSON.parse(JSON.stringify(value));
const bool = (value) => value ? "checked" : "";
export {skillNamesFromText} from "./configuration-controls.js";
export function invalidateImportPreview(panel, applyButton, summary) {
  panel._importDocument = null;
  if (applyButton) applyButton.disabled = true;
  if (summary) summary.textContent = "Validate the document to preview it.";
}
const option = (panel, value, selected, label = null, disabled = false) => `<option value="${panel._e(value)}" ${disabled ? "disabled" : ""} ${value === selected ? "selected" : ""}>${panel._e(label || panel._titleCase(value))}</option>`;
const CHOICE_LABELS = Object.freeze({
  conversation_continuity: Object.freeze({
    ha_default: "Use Home Assistant sessions",
    device: "Remember by voice device",
    user: "Remember by user across devices",
  }),
  voice_scope_policy: Object.freeze({
    unretained: "Do not retain personal data",
    shared: "Use shared household data",
    default_user: "Use the default user",
    device_mapping: "Use a device-to-user mapping",
  }),
  voice_unmapped_policy: Object.freeze({
    unretained: "Do not retain personal data",
    shared: "Use shared household data",
    default_user: "Use the default user",
    device_mapping: "Device mapping (no retained data)",
  }),
});
export const configurationChoiceLabel = (key, item) => friendlySettingValue(key, item.value) || CHOICE_LABELS[key]?.[item.value] || item.label;
const settingSearch = (label, description, key) => `${label} ${description} ${key} ${settingSearchAliases(key)}`.toLowerCase();
const labelRow = (panel, label, key, helpKey = null, strong = false, value = undefined, disabled = false) => {
  const text = panel._e(friendlySettingLabel(key) || label);
  return `<span class="setting-label-row"><label for="config-${key}">${strong ? `<strong>${text}</strong>` : text}</label>${helpKey ? helpButton(panel, helpKey) : ""}${settingBadgesMarkup(panel, key, value, disabled)}</span>`;
};
const field = (panel, key, label, value, type = "text", description = "", disabled = false, helpKey = null, forceVisible = false) => {
  if (key === "memory_auto_retrieve_limit") {
    label = "Automatically include memories";
    description = "Select up to this many relevant memories when a new conversation starts. The same memories remain available for that conversation. Set to 0 to use memory only on demand.";
  }
  let presentation = modelFieldPresentation(panel, key, value);
  if (presentation.visible === false && !forceVisible) return "";
  if (presentation.visible === false) presentation = {};
  disabled = forceVisible || (presentation.disabled ?? disabled);
  let result = `<div class="setting ${presentation.disabled || forceVisible ? "is-disabled eoc-unavailable-model-setting" : ""}" data-field="${key}" data-setting data-search="${panel._e(settingSearch(label, description, key, helpKey))}">${labelRow(panel, label, key, helpKey, false, value, disabled)}<input id="config-${key}" data-config="${key}" data-type="${type}" type="${type === "number" ? "number" : "text"}" value="${panel._e(value)}" ${presentation.max ? `max="${presentation.max}"` : ""} ${presentation.models ? 'list="extended-openai-model-catalog"' : ""} ${disabled ? "disabled" : ""}>${description ? `<small>${description}</small>` : ""}<span class="field-error" data-error="${key}"></span>${modelFieldNotes(panel, presentation)}</div>`;
  if (key === "memory_auto_retrieve_limit") {
    const config = panel._draft || panel._result?.config || {};
    const choices = panel._result?.options?.memory_retrieval_mode || [];
    result += select(panel, "memory_retrieval_mode", "Memory retrieval", config.memory_retrieval_mode, choices, "Semantic matching can find memories with related meaning, not just similar words. It falls back to local matching if unavailable.");
    result += field(panel, "memory_embedding_model", "Embedding model", config.memory_embedding_model || "text-embedding-3-small", "text", "Used only for hybrid semantic retrieval; the provider must support embeddings.");
  }
  return result;
};
const select = (panel, key, label, value, options, description = "", disabled = false, helpKey = null, forceVisible = false) => {
  let presentation = modelFieldPresentation(panel, key, value);
  if (presentation.visible === false && !forceVisible) return "";
  if (presentation.visible === false) presentation = {};
  disabled = forceVisible || disabled;
  value = presentation.value ?? value;
  options = presentation.options ?? options;
  return `<div class="setting ${forceVisible ? "is-disabled eoc-unavailable-model-setting" : ""}" data-field="${key}" data-setting data-search="${panel._e(settingSearch(label, description, key, helpKey))}">${labelRow(panel, label, key, helpKey, false, value, disabled)}<select id="config-${key}" data-config="${key}" ${disabled ? "disabled" : ""}>${options.map((item) => {
    const choice = typeof item === "string" ? item : item.value;
    return option(panel, choice, value, friendlySettingValue(key, choice) || (typeof item === "string" ? null : item.label), presentation.disabledOption?.(choice));
  }).join("")}</select>${description ? `<small>${description}</small>` : ""}<span class="field-error" data-error="${key}"></span>${modelFieldNotes(panel, presentation)}</div>`;
};
const toggle = (panel, key, label, value, description = "", disabled = false, helpKey = null) => `<div class="config-toggle setting" data-field="${key}" data-setting data-search="${panel._e(settingSearch(label, description, key, helpKey))}"><span class="setting-copy">${labelRow(panel, label, key, helpKey, true, value, disabled)}${description ? `<small>${description}</small>` : ""}</span><label class="switch-control" for="config-${key}"><input id="config-${key}" data-config="${key}" data-type="boolean" type="checkbox" role="switch" ${bool(value)} ${disabled ? "disabled" : ""}><span class="switch-track" aria-hidden="true"></span></label></div>`;

export async function copyTextToClipboard(text, clipboardNavigator = globalThis.navigator, clipboardDocument = globalThis.document) {
  const writeText = clipboardNavigator?.clipboard?.writeText;
  if (typeof writeText === "function") {
    try {
      await writeText.call(clipboardNavigator.clipboard, text);
      return;
    } catch (err) {
      if (!clipboardDocument) throw err;
    }
  }
  if (!clipboardDocument?.body || typeof clipboardDocument.createElement !== "function" || typeof clipboardDocument.execCommand !== "function") {
    throw new Error("Clipboard access is unavailable in this browser context");
  }
  const textarea = clipboardDocument.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  clipboardDocument.body.appendChild(textarea);
  try {
    textarea.select();
    if (!clipboardDocument.execCommand("copy")) throw new Error("Browser copy command failed");
  } finally {
    textarea.remove();
  }
}

export const functionGroupIdFromName = (name) => String(name || "").trim().toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").replace(/^[^a-z]+/, "").slice(0, 64);
export const isFunctionGroupEnabled = (group = {}) => group?.enabled !== false;
export const isFunctionToolEnabled = (tool) => tool?.enabled !== false;

export function functionToolCountLabel(tools = []) {
  const enabled = tools.filter(isFunctionToolEnabled).length;
  const disabled = tools.length - enabled;
  if (!disabled) return `${enabled} ${enabled === 1 ? "function" : "functions"}`;
  return `${enabled} enabled · ${disabled} disabled`;
}

export function backupSummaryLines(summary = {}) {
  return [
    "Agent configuration",
    `${Number(summary.request_rules || 0)} Request Rules`,
    `${Number(summary.persistent_memories || 0)} persistent memories`,
    `${Number(summary.temporary_memories || 0)} active temporary memories`,
    `${Number(summary.knowledge_sources || 0)} Knowledge sources`,
    `${Number(summary.archive_sessions || 0)} archived conversations (${Number(summary.archive_turns || 0)} turns)`,
    `Usage history (${Number(summary.usage_runs || 0)} runs, ${Number(summary.usage_requests || 0)} requests)`,
    `Guest Mode schedule ${summary.guest_mode_scheduled ? "included" : "inactive"}`,
  ];
}

export const canReplaceToolYamlWithoutConfirmation = (current, replaceable) => !String(current || "").trim() || current === replaceable;

const searchTokens = (value) => String(value || "")
  .normalize("NFKD")
  .replace(/[\u0300-\u036f]/g, "")
  .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
  .toLowerCase()
  .replace(/[^a-z0-9]+/g, " ")
  .trim()
  .split(/\s+/)
  .filter(Boolean)
  .map((token) => {
    if (token.length > 4 && token.endsWith("ies")) return `${token.slice(0, -3)}y`;
    if (token.length > 4 && /(ches|shes|sses|xes|zes)$/.test(token)) return token.slice(0, -2);
    if (token.length > 3 && token.endsWith("s") && !token.endsWith("ss")) return token.slice(0, -1);
    return token;
  });

function matchesFunctionSearchTokens(queryTokens, textTokens) {
  if (!queryTokens.length) return true;
  return queryTokens.every((queryToken) => textTokens.some((textToken) =>
    textToken === queryToken || (
      Math.min(textToken.length, queryToken.length) >= 3
      && (textToken.startsWith(queryToken) || queryToken.startsWith(textToken))
    )
  ));
}

export function matchesFunctionSearch(query, searchableText) {
  return matchesFunctionSearchTokens(searchTokens(query), searchTokens(searchableText));
}

export function deleteFunctionGroup(config, groupId) {
  const result = clone(config);
  result.function_groups = (result.function_groups || []).filter((group) => group.id !== groupId);
  return result;
}

function indexFunctionToolGroups(config) {
  const tools = config.functions || [];
  const groups = config.function_groups || [];
  const indexedGroups = groups.map((group) => ({...group, tools: []}));
  const groupIndexesByTool = new Map();
  const membership = new Map();

  groups.forEach((group, groupIndex) => {
    for (const name of group.functions || []) {
      const indexes = groupIndexesByTool.get(name);
      if (indexes) indexes.push(groupIndex);
      else groupIndexesByTool.set(name, [groupIndex]);
      membership.set(name, group);
    }
  });

  const alwaysAvailable = [];
  for (const tool of tools) {
    const name = tool.spec?.name;
    const groupIndexes = groupIndexesByTool.get(name);
    if (!groupIndexes) {
      alwaysAvailable.push(tool);
      continue;
    }
    for (const groupIndex of groupIndexes) indexedGroups[groupIndex].tools.push(tool);
  }

  return {alwaysAvailable, groups: indexedGroups, membership};
}

export function categorizeFunctionTools(config) {
  const {alwaysAvailable, groups} = indexFunctionToolGroups(config);
  return {alwaysAvailable, groups};
}

function section(panel, id, title, description, keywords, body, includeHeading = true) {
  if (panel._configSectionFilter && !panel._configSectionFilter.has(id)) return "";
  const content = typeof body === "function" ? body() : body;
  return `<section id="config-${id}" class="config-section" data-config-section data-search="${panel._e(`${title} ${description} ${keywords}`.toLowerCase())}">${includeHeading ? `<div class="config-section-heading"><h2 class="eyebrow">${title}</h2><p>${description}</p></div>` : ""}${content}</section>`;
}

function renderLocalHandling(panel, config) {
  const state = panel._result?.local_handling || {};
  const intents = state.intents || [];
  const excluded = new Set(config.local_intent_exclusions || []);
  const conflicts = state.pipeline_conflicts || [];
  const conflictNames = conflicts.map((item) => item.name).filter(Boolean);
  const conflictText = conflictNames.length === 1
    ? `The Assist pipeline “${conflictNames[0]}” still has Home Assistant's Prefer local handling turned on.`
    : `${conflictNames.length} Assist pipelines using this agent still have Home Assistant's Prefer local handling turned on: ${conflictNames.join(", ")}.`;
  return `<div class="notice local-handling-explainer"><strong>How this differs from Home Assistant's “Prefer local handling”</strong><p>Home Assistant's own option runs before a request reaches Extended OpenAI. That is simple and fast, but it means Extended OpenAI cannot apply Request Rules or choose that particular command for a Function Tool instead.</p><p><strong>Extended OpenAI local handling</strong> runs after Request Rules. It can still use Home Assistant's fast built-in commands, while letting you send selected command types on to your Function Tools or AI model.</p><p><strong>Example:</strong> “Turn on the kitchen light” can stay local, while “turn off the kitchen light in 20 minutes” can be sent to a deferred-action Function Tool.</p></div><div class="config-stack">
    ${toggle(panel,"local_intents_enabled","Use Extended OpenAI local handling",config.local_intents_enabled,"After Request Rules, try Home Assistant's built-in commands first. Requests that do not match locally, or that you exclude below, continue to your Function Tools or AI model.",state.supported === false)}
    ${state.supported === false ? `<div class="notice"><strong>Local handling is not available</strong><p>This Home Assistant version does not provide the local command interface this feature needs. Extended OpenAI will continue to use the AI path normally.</p></div>` : ""}
    ${conflicts.length ? `<div class="notice"><strong>Home Assistant is already handling some commands first</strong><p>${panel._e(conflictText)} Those commands may be completed before they reach this agent, so the choices below cannot apply to them. Turn off <strong>Prefer local handling</strong> for ${conflicts.length === 1 ? "that pipeline" : "those pipelines"} if you want Extended OpenAI to control this order.</p></div>` : ""}
    <div class="dependent ${config.local_intents_enabled ? "" : "is-disabled"}" data-dependent="local_intents_enabled">
        <div class="setting-group" data-setting data-search="local handling exceptions always send command types ai home assistant intents">
        <div class="subheading"><h3>Send these command types to AI</h3><p>Choose any commands that should skip local handling and continue to your Function Tools or AI model.</p></div>
        ${intents.length ? `<label class="tool-search"><span class="sr-only">Find a command type</span><input id="local-intent-search" type="search" placeholder="Find a command type..." aria-label="Find a command type" ${config.local_intents_enabled ? "" : "disabled"}></label><div id="local-intent-list" class="group-function-choices"><label class="group-function-choice" data-local-intent-choice data-choice-search="delayed device commands scheduled deferred actions turn off later"><input type="checkbox" data-config="local_intent_delayed_commands_to_ai" data-type="boolean" ${config.local_intent_delayed_commands_to_ai ? "checked" : ""} ${config.local_intents_enabled ? "" : "disabled"}><span><strong>Delayed device commands</strong><small>For example, “turn off the lights in 20 minutes”. Normal timers such as “set a 20 minute timer” can still stay local.</small></span></label>${intents.map((item) => `<label class="group-function-choice ${item.available === false ? "is-disabled" : ""}" data-local-intent-choice data-choice-search="${panel._e(`${item.label} ${item.intent}`.toLowerCase())}"><input type="checkbox" data-local-intent-exclusion value="${panel._e(item.intent)}" ${excluded.has(item.intent) ? "checked" : ""} ${config.local_intents_enabled ? "" : "disabled"}><span><strong>${panel._e(item.label)}</strong><small><code>${panel._e(item.intent)}</code>${item.available === false ? " · Not currently available in Home Assistant" : ""}</small></span></label>`).join("")}</div>` : `<p class="help">No Home Assistant command types are currently registered.</p>`}
        <span class="field-error" data-error="local_intent_exclusions"></span>
      </div>
      <p class="help">Request Rules always get the first chance. While Guest Mode is active, requests keep using the existing Guest Mode safeguards instead of this local shortcut.</p>
    </div>
  </div>`;
}

export function saveBar(panel) {
  if (!panel._configDirty) return "";
  return saveBarMarkup({configuration: true, pending: Boolean(panel._configurationSaving)});
}

export function renderConfigurationActions(panel, sections = panel._configSections || []) {
  const cleanOnly = panel._configDirty ? "disabled title=\"Save or revert the shared draft first\"" : "";
  const includesBackup = sections.includes("backup");
  return `<details class="agent-actions-menu eoc-details-base"><summary aria-haspopup="menu">Assistant actions</summary><div role="menu"><button type="button" class="secondary" role="menuitem" id="duplicate-agent" ${cleanOnly}>Duplicate agent</button>${includesBackup ? "" : `<button type="button" class="secondary" role="menuitem" id="import-agent">Import configuration</button><button type="button" class="secondary" role="menuitem" id="export-agent" ${cleanOnly}>Export configuration</button>`}</div></details>${panel._configDirty ? `<p class="action-help">Duplicate and Export configuration use the saved configuration. Save or revert your settings changes to enable them.</p>` : ""}`;
}

export function renderConfiguration(panel, {voiceIdentity = null, renderExposedAttributes = null, renderBackup = null} = {}) {
  const view = panel._viewKey?.();
  const webSkillsOnly = view === "capabilities/web-skills";
  const config = panel._draft || panel._result?.config || {};
  const capabilities = panel._result?.model_capabilities || {};
  const defaults = panel._result?.defaults || {};
  const regexRules = config.speech_regex_replacements || [];
  const archive = config.archive_enabled;
  const speech = config.speech_processing_enabled;
  const choices = (key) => (panel._result?.options?.[key] || []).map((item) => ({...item, label: configurationChoiceLabel(key, item)}));
  const cleanOnly = panel._configDirty ? "disabled title=\"Save or revert the shared draft first\"" : "";
  const continuity = config.conversation_continuity !== "ha_default";
  const timeoutChoices = choices("conversation_timeout_minutes");
  const timeoutPreset = timeoutChoices.some((item) => Number(item.value) === Number(config.conversation_timeout_minutes)) ? String(config.conversation_timeout_minutes) : "custom";
  const timeoutControl = `<div class="setting" data-field="conversation_timeout_minutes" data-setting data-search="${panel._e(settingSearch("Conversation timeout", "Starts a fresh conversation after this much inactivity.", "conversation_timeout_minutes", "conversation_timeout"))}">${labelRow(panel,"Conversation timeout","conversation_timeout_minutes","conversation_timeout")}<select id="conversation-timeout-preset" ${continuity ? "" : "disabled"}>${timeoutChoices.map((item) => option(panel,String(item.value),timeoutPreset,item.label)).join("")}${option(panel,"custom",timeoutPreset,"Custom")}</select><input id="config-conversation_timeout_minutes" data-config="conversation_timeout_minutes" data-type="number" type="number" min="1" max="1440" value="${panel._e(config.conversation_timeout_minutes)}" ${continuity && timeoutPreset === "custom" ? "" : "hidden"} ${continuity ? "" : "disabled"}><small>Starts a fresh conversation after this much inactivity.</small><span class="field-error" data-error="conversation_timeout_minutes"></span></div>`;
  const jumps = [["general","General"],["conversation","Conversation"],["local","Local handling"],["prompt","Prompt"],["capabilities","Capabilities"],["archive","Archive"],["voice","Voice"],["speech","Speech"],["context","Context"],["model","Model"],["retention","Retention"],["backup","Backup & Restore"]];
  panel._configSectionFilter = new Set(panel._configSections || jumps.map(([id]) => id));
  if (panel._configSectionFilter.has("conversation") && view !== "assistant/conversation") panel._configSectionFilter.add("local");
  const capabilityField = (capability, renderer) => capabilities[capability] === true ? renderer(false) : capabilities[capability] === false ? renderer(true) : "";
  return `<div class="content-card config-surface">
      ${section(panel,"general","General","Choose the model and basic response behaviour.","model api tokens function calls continue",() => `<div class="form-grid general-grid">${field(panel,"__title","Agent name",panel._draftTitle ?? panel._result?.title ?? "")}${field(panel,"chat_model","Chat model",config.chat_model)}${select(panel,"api_mode","Provider API format",config.api_mode,choices("api_mode"),"Choose how requests are formatted for the provider. Auto is recommended unless your provider requires a specific API.",false,"api_mode")}${field(panel,"max_tokens","Maximum response length (tokens)",config.max_tokens,"number","Sets the most tokens the model may use in one response.")}${field(panel,"max_function_calls_per_conversation","Tool-call limit per conversation",config.max_function_calls_per_conversation,"number","Stops additional tool calls when this limit is reached.")}${toggle(panel,"function_tool_error_recovery","Correct safe tool-call errors automatically",config.function_tool_error_recovery,"Lets the assistant correct and retry mistakes only when no action could already have started. Other failed actions are never retried automatically.")}${select(panel,"continue_conversation","Listen for a follow-up",config.continue_conversation,choices("continue_conversation"),"Choose whether Home Assistant keeps listening for an immediate reply after the assistant responds.",false,"continue_conversation")}</div>`)}
      ${section(panel,"conversation","Conversation","Control how conversations continue between separate Assist requests.","conversation continue resume cross-device timeout",() => `<div class="form-grid">${select(panel,"conversation_continuity","Remember recent conversation",config.conversation_continuity,choices("conversation_continuity"),"",false,"conversation_continuity")}<div class="dependent ${continuity ? "" : "is-disabled"}" data-dependent="conversation_continuity">${timeoutControl}</div></div>`)}
      ${section(panel,"local","Local handling","Let Extended OpenAI try Home Assistant's built-in commands after Request Rules, before using AI.","local home assistant intents exceptions request rules delayed commands ai",() => renderLocalHandling(panel,config))}
      ${section(panel,"prompt","Prompt & context","Instructions and live Home Assistant context used by this agent.","template instructions preview effective request date time exposed devices",() => `<div class="config-stack context-toggles">${toggle(panel,"current_datetime_enabled","Include current date & time",config.current_datetime_enabled,"")}${toggle(panel,"exposed_entities_enabled","Include exposed devices",config.exposed_entities_enabled,"")}${renderExposedAttributes ? renderExposedAttributes(panel) : ""}</div><details class="advanced-context-formatting eoc-details-base" data-setting data-search="advanced context formatting custom templates date devices"><summary>Advanced context formatting</summary><p class="help">Customize how date, time and device information is added to the prompt. Leave a field blank to use the default.</p><div class="config-stack"><label class="setting"><span class="setting-label-row"><span>Current date/time format</span><button type="button" class="secondary compact-button reset-context-template" data-template-key="current_datetime_template">Reset to default</button></span><textarea data-config="current_datetime_template" class="short-textarea" spellcheck="false" placeholder="Default integration-managed format">${panel._e(config.current_datetime_template || "")}</textarea><span class="field-error" data-error="current_datetime_template"></span></label><label class="setting"><span class="setting-label-row"><span>Device context format</span><button type="button" class="secondary compact-button reset-context-template" data-template-key="exposed_entities_template">Reset to default</button></span><textarea data-config="exposed_entities_template" class="short-textarea" spellcheck="false" placeholder="Default integration-managed format">${panel._e(config.exposed_entities_template || "")}</textarea><span class="field-error" data-error="exposed_entities_template"></span></label></div></details><label class="setting prompt-setting" data-setting data-search="prompt system instructions template"><span class="sr-only">System prompt</span><textarea id="prompt-editor" data-config="prompt" class="prompt-editor" spellcheck="false">${panel._e(config.prompt || "")}</textarea></label><div class="editor-meta"><span id="prompt-count">${(config.prompt || "").length.toLocaleString()} characters</span><span class="editor-meta-actions"><button type="button" class="secondary compact-button" id="preview-request">Preview effective request</button><button type="button" class="secondary compact-button" id="reset-prompt">Reset to default</button></span></div><span class="field-error" data-error="prompt"></span>`)}
      ${section(panel,"capabilities",webSkillsOnly ? "Web search & Skills" : "Capabilities",webSkillsOnly ? "Choose optional online information and installed instruction sets the assistant may load when needed." : "Choose which information sources and memory features the assistant can use.","web search skills knowledge memory temporary ephemeral short-term eager",() => `<div class="config-stack">${toggle(panel,"web_search","Web search",config.web_search,"")}<div class="dependent ${config.web_search ? "" : "is-disabled"}" data-dependent="web_search">${select(panel,"web_search_context","Web search detail",config.web_search_context,choices("web_search_context"),"Choose how much information is returned with search results.",!config.web_search,"web_search_context")}</div>${webSkillsOnly ? "" : `${toggle(panel,"knowledge_enabled","Knowledge Library",config.knowledge_enabled,"Allow the assistant to search your Knowledge Library when useful.",false,"knowledge_library")}${select(panel,"memory_mode","Long-term memory",config.memory_mode,choices("memory_mode"),"Choose whether the assistant can save durable facts for future conversations.",false,"memory_mode")}${select(panel,"temporary_memory","Short-term memory",config.temporary_memory,choices("temporary_memory"),"Choose how readily the assistant remembers useful temporary details until they expire.",false,"temporary_memory")}`}<label class="setting" data-field="skills" data-setting data-search="${panel._e(settingSearch("Skills", "Installed instruction sets the assistant may load when needed.", "skills"))}">${labelRow(panel,"Skills","skills")}<textarea id="config-skills" data-config="skills" class="short-textarea" spellcheck="false">${panel._e((config.skills || []).join("\n"))}</textarea><small>Enter one installed skill name per line. Commas are preserved as part of a skill name.</small><span class="field-error" data-error="skills"></span></label></div>`)}
      ${section(panel,"archive","Conversation archive","Manage saved conversation history and search.","retention shared session timeout model search",() => `<div class="config-stack">${toggle(panel,"archive_enabled","Save conversation history",archive,"")}<div class="dependent ${archive ? "" : "is-disabled"}" data-dependent="archive_enabled"><div class="form-grid">${select(panel,"archive_retention_days","Keep archived conversations for",config.archive_retention_days,choices("archive_retention_days"),"Choose how long saved conversations remain available before they are deleted automatically.",!archive)}${field(panel,"archive_session_timeout_minutes","New conversation after inactivity (minutes)",config.archive_session_timeout_minutes,"number","After this much inactivity, the next message starts a new saved conversation.",!archive,"archive_session_timeout")}</div>${toggle(panel,"archive_model_search_enabled","Let the assistant search the archive",config.archive_model_search_enabled,"Lets the assistant search saved conversations when an earlier discussion may help answer you.",!archive,"archive_model_search")}${toggle(panel,"shared_archive_enabled","Save shared-household conversations",config.shared_archive_enabled,"Also saves eligible conversations assigned to the shared household; private user archives stay private.",!archive,"shared_archive")}</div></div>`)}
      ${section(panel,"voice","Voice & identity","Decide whose memories and conversation history should be used for requests from each voice device.","unidentified unmapped default owner shared memory satellite device mappings",() => voiceIdentity ? voiceIdentity(panel) : `<div class="form-grid">${select(panel,"voice_scope_policy","When the speaker is not identified",config.voice_scope_policy,choices("voice_scope_policy"),"",false,"voice_scope_policy")}${select(panel,"voice_unmapped_policy","When a device has no mapping",config.voice_unmapped_policy,choices("voice_unmapped_policy"),"",false,"voice_unmapped_policy")}${field(panel,"voice_default_user_id","Default Home Assistant user ID",config.voice_default_user_id || "","text","Enter the user ID whose memories and history are used when a policy selects the default user.")}${select(panel,"shared_memory_mode","Shared household memory",config.shared_memory_mode,choices("shared_memory_mode"),"Choose whether the assistant can save long-term memories shared by everyone in the household.",false,"shared_memory_mode")}</div><div class="setting mappings-setting" data-setting data-search="${panel._e(settingSearch("Voice device assignments", "Assigns voice satellites and devices to a specific user or shared household data.", "voice_device_mappings", "voice_device_mappings"))}"><span class="setting-label-row"><label for="voice-mappings">Voice device assignments (JSON)</label>${helpButton(panel,"voice_device_mappings")}</span><textarea id="voice-mappings" class="yaml-editor mappings-editor" spellcheck="false">${panel._e(JSON.stringify(config.voice_device_mappings || {}, null, 2))}</textarea><small>Assign each satellite or device ID to a Home Assistant user, shared household data, or no retained data.</small><span class="field-error" data-error="voice_device_mappings"></span></div>`, !voiceIdentity)}
      ${section(panel,"speech","Spoken response","Adjust text before it is spoken without changing the saved assistant response.","tts markdown url regex replacements preview",() => `<div class="config-stack">${toggle(panel,"speech_processing_enabled","Clean responses for speech",speech,"")}<div class="dependent ${speech ? "" : "is-disabled"}" data-dependent="speech_processing_enabled">${toggle(panel,"speech_strip_markdown","Remove Markdown formatting",config.speech_strip_markdown,"Stops formatting and Markdown links from being read aloud.",!speech)}${toggle(panel,"speech_strip_urls","Remove bare URLs",config.speech_strip_urls,"Prevents raw web addresses from being spoken, including during progressive speech.",!speech)}<div class="setting-group" data-setting data-search="${panel._e(settingSearch("Custom replacements", "Rules run on the completed response and affect spoken output only.", "speech_regex_replacements", "custom_replacements"))}"><div class="subheading"><span class="setting-label-row"><h3>Custom replacements</h3>${helpButton(panel,"custom_replacements")}</span><p>Replace words or patterns in spoken responses. Streaming speech is disabled while custom replacements are active.</p></div><div class="rule-headings" aria-hidden="true"><span>Pattern</span><span>Replacement</span><span>Actions</span></div><div id="regex-rules" class="rule-list">${regexRules.map((rule,index)=>regexRow(panel,rule,index,!speech)).join("")}</div><button type="button" class="secondary add-regex" id="add-regex" ${speech ? "" : "disabled"}>+ Add rule</button></div><div class="preview-grid setting-group" data-setting data-search="spoken text preview sample assistant"><label>Sample assistant text<textarea id="speech-sample" class="short-textarea" placeholder="Paste a response containing links or abbreviations" ${speech ? "" : "disabled"}></textarea></label><label>Preview spoken text<textarea id="speech-output" class="short-textarea" readonly></textarea></label></div><button type="button" class="secondary preview-button" id="preview-speech" ${speech ? "" : "disabled"}>Preview spoken text</button></div></div>`)}
      ${section(panel,"context","Conversation history limits","Choose how older conversation history is reduced when it becomes too large.","threshold truncate summarize recent clear",() => `<div class="form-grid">${field(panel,"context_threshold","Trim history after (input tokens)",config.context_threshold,"number","When history reaches this size, use the trimming method below.",false,"context_threshold")}${select(panel,"context_truncate_strategy","Conversation history trimming",config.context_truncate_strategy,choices("context_truncate_strategy"),"Choose whether to keep recent messages, clear the history, or summarize older messages when the limit is reached.",false,"context_truncation")}</div>`)}
      ${section(panel,"model","Model parameters","Fine-tune supported model behaviour. Most users can keep the defaults.","temperature top p reasoning effort service tier tool id memory retrieve",() => `<div class="form-grid">${capabilityField("supports_temperature", (disabled) => field(panel,"temperature","Response creativity (temperature)",config.temperature,"number","Higher values make responses more varied; lower values make them more predictable.",disabled,null,disabled))}${capabilityField("supports_top_p", (disabled) => field(panel,"top_p","Response diversity (Top P)",config.top_p,"number","Adjusts how widely the model samples possible words. Usually leave this at its default.",disabled,null,disabled))}${capabilityField("supports_reasoning_effort", (disabled) => select(panel,"reasoning_effort","Reasoning effort",config.reasoning_effort || "low",choices("reasoning_effort"),"Choose how much work the model spends on difficult tasks; higher settings may be slower and cost more.",disabled,"reasoning_effort",disabled))}${capabilityField("supports_service_tier", (disabled) => select(panel,"service_tier","Processing tier",config.service_tier || "flex",choices("service_tier"),"Choose the provider's service tier, which can affect request priority, availability, or cost.",disabled,"service_tier",disabled))}${toggle(panel,"shorten_tool_call_id","Use shorter tool-call IDs",config.shorten_tool_call_id,"Enable only when your provider rejects the default tool-call identifiers.")}${view === "assistant/model-responses" ? "" : field(panel,"memory_auto_retrieve_limit","Memories included per request",config.memory_auto_retrieve_limit,"number","Sets the maximum number of relevant long-term memories added to each request; it does not limit stored memories.")}</div><button type="button" class="secondary compact-button section-reset" id="${view === "assistant/model-responses" ? "reset-model-parameters" : "reset-advanced"}">Reset model parameters</button>${modelDataControls(panel)}`)}
      ${section(panel,"retention","Usage history retention","Choose how long detailed usage records are kept. Overall totals are kept separately.","usage request run details totals",() => `<div class="form-grid">${select(panel,"usage_request_retention_days","Keep request details for",config.usage_request_retention_days,choices("usage_request_retention_days"),"")}${select(panel,"usage_run_retention_days","Keep run details for",config.usage_run_retention_days,choices("usage_run_retention_days"),"")}</div>`)}
      ${section(panel,"backup","Export, Backup, Import & Restore","Move reusable setup or durable agent data safely between agents and installations.","disaster recovery migration memories knowledge usage private",() => renderBackup ? renderBackup(Boolean(panel._configDirty)) : "")}
      ${saveBar(panel)}<span id="save-bar-anchor" class="sr-only" data-defaults="${panel._e(JSON.stringify(defaults))}"></span>
    </div>`;
}

function regexRow(panel, rule, index, disabled = false) {
  return `<article class="rule-row" data-regex-index="${index}"><label><span class="mobile-label">Pattern</span><input class="regex-pattern" value="${panel._e(rule.pattern || "")}" spellcheck="false" ${disabled ? "disabled" : ""}><span class="field-error" data-error="speech_regex_replacements[${index}].pattern"></span></label><label><span class="mobile-label">Replacement</span><input class="regex-replacement" value="${panel._e(rule.replacement || "")}" spellcheck="false" ${disabled ? "disabled" : ""}><span class="field-error" data-error="speech_regex_replacements[${index}].replacement"></span></label><div class="rule-actions"><button type="button" class="secondary move-regex" data-direction="-1" aria-label="Move rule up" ${disabled || index === 0 ? "disabled" : ""}>&uarr;</button><button type="button" class="secondary move-regex" data-direction="1" aria-label="Move rule down" ${disabled ? "disabled" : ""}>&darr;</button><button type="button" class="danger delete-regex" ${disabled ? "disabled" : ""}>Delete</button></div></article>`;
}

function dirty(panel) {
  readConfig(panel);
  panel._setConfigDirty(true);
  if (!panel.shadowRoot.querySelector(".save-bar")) {
    panel.shadowRoot.querySelector("#save-bar-anchor")?.insertAdjacentHTML("beforebegin", saveBar(panel));
    bindSaveBar(panel);
  }
}

function showErrors(panel, errors = {}) {
  panel.shadowRoot.querySelectorAll(".field-error").forEach((item) => item.textContent = "");
  Object.entries(errors).forEach(([key, message]) => {
    const target = panel.shadowRoot.querySelector(`[data-error="${CSS.escape(key)}"]`) || panel.shadowRoot.querySelector(`[data-error="${CSS.escape(key.split("[")[0])}"]`);
    if (target) target.textContent = message;
  });
}

function renderRegexRules(panel, focusIndex = null) {
  const list = panel.shadowRoot.querySelector("#regex-rules");
  if (!list) return;
  const disabled = !panel._draft.speech_processing_enabled;
  list.innerHTML = panel._draft.speech_regex_replacements.map((rule,index) => regexRow(panel,rule,index,disabled)).join("");
  bindRegexRules(panel);
  if (focusIndex !== null) {
    const input = list.querySelector(`[data-regex-index="${focusIndex}"] .regex-pattern`);
    input?.focus();
    input?.scrollIntoView({block:"nearest"});
  }
}

function bindRegexRules(panel) {
  const root = panel.shadowRoot;
  root.querySelectorAll(".delete-regex").forEach((button) => button.addEventListener("click", () => {
    const index = Number(button.closest(".rule-row").dataset.regexIndex);
    readConfig(panel);
    panel._draft.speech_regex_replacements.splice(index, 1);
    panel._setConfigDirty(true);
    renderRegexRules(panel, Math.min(index, panel._draft.speech_regex_replacements.length - 1));
  }));
  root.querySelectorAll(".move-regex").forEach((button) => button.addEventListener("click", () => {
    const index = Number(button.closest(".rule-row").dataset.regexIndex);
    const target = index + Number(button.dataset.direction);
    readConfig(panel);
    if (target < 0 || target >= panel._draft.speech_regex_replacements.length) return;
    [panel._draft.speech_regex_replacements[index], panel._draft.speech_regex_replacements[target]] = [panel._draft.speech_regex_replacements[target], panel._draft.speech_regex_replacements[index]];
    panel._setConfigDirty(true);
    renderRegexRules(panel, target);
  }));
}

function bindSaveBar(panel) {
  const root=panel.shadowRoot;
  bindSingleRequestSave(panel);

}

export function bindConfiguration(panel) {
  const root = panel.shadowRoot;
  bindModelDataControls(panel, (message) => { readConfig(panel); panel._render(); panel._toast(message); });
  bindHelp(panel);
  bindSaveBar(panel);
  bindConfigurationInputs(panel);
  root.querySelector("#local-intent-search")?.addEventListener("input", (event) => {
    const queryTokens = searchTokens(event.target.value);
    root.querySelectorAll("[data-local-intent-choice]").forEach((choice) => {
      choice.hidden = !matchesFunctionSearchTokens(
        queryTokens,
        cachedFunctionSearchTokens(choice, choice.dataset.choiceSearch),
      );
    });
  });
  bindRegexRules(panel);
  const actionsMenu = root.querySelector(".agent-actions-menu");
  actionsMenu?.addEventListener("keydown", (event) => { if (event.key === "Escape") { actionsMenu.open = false; actionsMenu.querySelector("summary")?.focus(); } });
  actionsMenu?.querySelectorAll("button").forEach((button) => button.addEventListener("click", () => { actionsMenu.open = false; }));
  root.querySelectorAll("[data-jump]").forEach((link) => link.addEventListener("click", (event) => { event.preventDefault(); root.querySelector(`#${link.dataset.jump}`)?.scrollIntoView({behavior:"smooth",block:"start"}); }));
  root.querySelector("#conversation-timeout-preset")?.addEventListener("change", (event) => {
    const input = root.querySelector('[data-config="conversation_timeout_minutes"]');
    const custom = event.target.value === "custom";
    input.hidden = !custom;
    if (!custom) input.value = event.target.value;
    updateConfigurationControl(panel, input);
    if (custom) input.focus();
  });
  root.querySelector("#reset-prompt")?.addEventListener("click", () => { const editor=root.querySelector("#prompt-editor"); editor.value=panel._result.defaults.prompt; for(const key of ["current_datetime_enabled","exposed_entities_enabled"]){const input=root.querySelector(`[data-config="${key}"]`);if(input)input.checked=true;} for(const key of ["current_datetime_template","exposed_entities_template"]){const input=root.querySelector(`[data-config="${key}"]`);if(input)input.value="";} editor.focus(); dirty(panel); root.querySelector("#prompt-count").textContent=`${editor.value.length.toLocaleString()} characters`; });
  root.querySelectorAll(".reset-context-template").forEach((button)=>button.addEventListener("click",()=>{const input=root.querySelector(`[data-config="${button.dataset.templateKey}"]`);if(input){input.value="";dirty(panel);input.focus();}}));
  root.querySelector("#preview-request")?.addEventListener("click", async () => {
    const dialog=root.querySelector("#prompt-preview-dialog");
    const status=root.querySelector("#prompt-preview-status");
    const sections=root.querySelector("#request-preview-sections");
    const notes=root.querySelector("#prompt-preview-notes");
    const copyButton=root.querySelector("#copy-prompt-preview");
    root.querySelector("#request-footprint").textContent="Calculating..."; root.querySelector("#function-group-savings").textContent=""; sections.innerHTML=""; notes.innerHTML=""; status.textContent="Assembling current Home Assistant request..."; status.className="validation"; copyButton.disabled=true; dialog.showModal();
    try {
      const result=await panel._call("configuration","request_preview",{config:readConfig(panel)});
      panel._effectiveRequestPreview=result;
      root.querySelector("#request-footprint").textContent=`Previewed Request Content: ${Number(result.total_character_count||0).toLocaleString()} characters`;
      const savings=result.function_group_savings||{};
      root.querySelector("#function-group-savings").textContent=`Saved by Function Groups: ${Number(savings.characters||0).toLocaleString()} characters (${Number(savings.percent||0)}%)`;
      sections.innerHTML=(result.sections||[]).map((section,index)=>`<details class="request-preview-section eoc-details-base" ${index===0?"open":""}><summary><span>${panel._e(section.label)}</span><span class="request-section-meta">${Number(section.character_count||0).toLocaleString()} characters <button type="button" class="secondary compact-button copy-request-section" data-section-index="${index}">Copy section</button></span></summary><textarea class="yaml-editor request-preview-output" readonly spellcheck="false">${panel._e(section.content||"")}</textarea></details>`).join("");
      root.querySelectorAll(".copy-request-section").forEach((button)=>button.addEventListener("click",async(event)=>{event.preventDefault();event.stopPropagation();const section=result.sections[Number(button.dataset.sectionIndex)];try{await copyTextToClipboard(section.content||"");panel._toast(`${section.label} copied`);}catch(err){panel._toast(`Unable to copy section: ${err.message||String(err)}`,true);}}));
      notes.innerHTML=(result.notes||[]).map((note)=>`<li>${panel._e(note)}</li>`).join("");
      status.textContent="Resolved using current Home Assistant state and production request assembly";
      status.className="validation valid";
      copyButton.disabled=false;
    } catch(err) {
      status.textContent=err.message||String(err);
      status.className="validation invalid";
    }
  });
  root.querySelector("#prompt-preview-close")?.addEventListener("click",()=>root.querySelector("#prompt-preview-dialog").close());
  root.querySelector("#copy-prompt-preview")?.addEventListener("click",async()=>{try{const text=(panel._effectiveRequestPreview?.sections||[]).map((section)=>`## ${section.label}\n${section.content}`).join("\n\n");await copyTextToClipboard(text);panel._toast("Effective request copied");}catch(err){panel._toast(`Unable to copy request: ${err.message||String(err)}`,true);}});
  root.querySelector("#reset-advanced")?.addEventListener("click", () => { ["temperature","top_p","reasoning_effort","service_tier","shorten_tool_call_id","memory_auto_retrieve_limit","memory_retrieval_mode","memory_embedding_model"].forEach((key) => { panel._draft[key]=clone(panel._result.defaults[key]); const input=root.querySelector(`[data-config="${key}"]`); if(input) input.dataset.type==="boolean" ? input.checked=panel._draft[key] : input.value=panel._draft[key]; }); panel._setConfigDirty(true); dirty(panel); });
  root.querySelector("#add-regex")?.addEventListener("click", () => { readConfig(panel); panel._draft.speech_regex_replacements.push({pattern:"",replacement:""}); panel._setConfigDirty(true); renderRegexRules(panel,panel._draft.speech_regex_replacements.length-1); });
  root.querySelector("#preview-speech")?.addEventListener("click", async () => { try { const response=await panel._call("configuration","speech_preview",{config:readConfig(panel),sample_text:root.querySelector("#speech-sample").value}); root.querySelector("#speech-output").value=response.speech_text; } catch(err){panel._toast(`Unable to preview speech: ${err.message||String(err)}`,true);} });
  root.querySelector("#duplicate-agent")?.addEventListener("click", async () => { if(panel._configDirty)return; try { const result=await panel._call("configuration","duplicate"); await panel._loadAgents(result.subentry_id); panel._toast(`Created ${result.title}`); } catch(err){panel._toast(`Unable to duplicate agent: ${err.message||String(err)}`,true);} });
  root.querySelector("#export-agent")?.addEventListener("click", async () => { if(panel._configDirty)return; if(!await panel._confirm("Export saved agent configuration?","Export applies best-effort secret redaction, but Function Tool definitions may contain embedded credentials. Review the downloaded file before sharing it.","Export"))return; const result=await panel._call("configuration","export"); const blob=new Blob([result.json],{type:"application/json"}); const url=URL.createObjectURL(blob); const link=document.createElement("a"); link.href=url; link.download=`${(panel._draftTitle||"agent").replace(/[^a-z0-9]+/gi,"-").toLowerCase()}.json`; link.click(); URL.revokeObjectURL(url); });
  const importDocument = root.querySelector("#import-document"), importApply = root.querySelector("#import-apply"), importSummary = root.querySelector("#import-summary");
  root.querySelector("#import-agent")?.addEventListener("click", () => { invalidateImportPreview(panel, importApply, importSummary); root.querySelector("#import-dialog")?.showModal(); });
  importDocument?.addEventListener("input", () => invalidateImportPreview(panel, importApply, importSummary));
  root.querySelector("#import-preview")?.addEventListener("click", async () => { try { const result=await panel._call("configuration","import_preview",{document:importDocument.value}); panel._importDocument=importDocument.value; importSummary.textContent=`${result.title} / ${result.summary.model} / ${result.summary.tools} tools / ${result.summary.function_groups} function groups / ${result.summary.speech_rules} speech rules`; importApply.disabled=false; } catch(err){panel._importDocument=null;importSummary.textContent=err.message||String(err);importApply.disabled=true;} });
  root.querySelector("#import-apply")?.addEventListener("click", async () => { const mode=root.querySelector('input[name="import-mode"]:checked').value, source=importDocument.value; if(!panel._importDocument||panel._importDocument!==source){invalidateImportPreview(panel,importApply,importSummary);return;} if(mode==="current"&&!await panel._confirm("Overwrite this agent?",`The saved configuration will be replaced.${panel._configDirty ? " Your unsaved shared draft will be discarded." : ""} Retained history and parent-entry credentials are not affected.`,"Overwrite"))return; try { await panel._call("configuration","import",{document:panel._importDocument,mode,confirm:mode==="current",...(mode==="current"?{revision:panel._configData?.revision}:{})}); root.querySelector("#import-dialog").close(); if(mode==="current")panel._clearConfigDraft(); await panel._loadAgents(panel._agentId); panel._toast(mode==="current"?"Configuration imported":"Agent created from import; your current draft is preserved"); } catch(err){panel._toast(`Unable to import: ${err.message||String(err)}`,true);} });
  root.querySelector("#import-cancel")?.addEventListener("click",()=>{root.querySelector("#import-dialog").close();actionsMenu?.querySelector("summary")?.focus();});
  root.querySelector("#import-dialog")?.addEventListener("cancel",()=>requestAnimationFrame(()=>actionsMenu?.querySelector("summary")?.focus()));
  if (panel._configRestoreFocus) { const selector=panel._configRestoreFocus; panel._configRestoreFocus=null; requestAnimationFrame(()=>root.querySelector(selector)?.focus({preventScroll:true})); }
}

const FUNCTION_GROUP_ASSIGNMENT_STYLE = `
  .function-group-assignment-control {
    display: inline-flex;
    align-items: center;
    gap: 0;
    width: fit-content;
    max-width: min(300px, 100%);
    min-height: 32px;
    margin-top: 9px;
    border: 1px solid var(--divider-color, rgba(127, 127, 127, .35));
    border-radius: 999px;
    background: var(--card-background-color, transparent);
    color: var(--primary-text-color);
    transition: border-color 120ms ease, background 120ms ease;
  }
  .function-group-assignment-control:hover,
  .function-group-assignment-control:focus-within {
    border-color: var(--primary-color);
    background: var(--secondary-background-color, transparent);
  }
  .function-group-assignment-control.is-disabled-group { opacity: .72; }
  .function-group-assignment-icon {
    --mdc-icon-size: 16px;
    display: inline-flex;
    align-items: center;
    flex: 0 0 auto;
    padding-inline-start: 10px;
    color: var(--secondary-text-color);
  }
  .function-group-assignment {
    width: auto;
    min-width: 0;
    max-width: 250px;
    min-height: 30px;
    border: 0;
    border-radius: 999px;
    outline: 0;
    background: transparent;
    color: inherit;
    font: inherit;
    font-size: 13px;
    font-weight: 500;
    padding: 4px 10px 4px 6px;
    cursor: pointer;
  }
  .function-group-assignment:disabled { cursor: progress; }
  .function-group-card[data-group-id],
  .function-group-card.always-card {
    background: var(--secondary-background-color, var(--card-background-color));
    background: color-mix(in srgb, var(--primary-color) 5%, var(--card-background-color));
    box-shadow: inset 3px 0 0 color-mix(in srgb, var(--primary-color) 28%, transparent);
  }
  .function-group-card.function-repair-attention {
    background: color-mix(in srgb, var(--warning-color, #ff9800) 8%, var(--card-background-color));
    box-shadow: inset 3px 0 0 color-mix(in srgb, var(--warning-color, #ff9800) 45%, transparent);
  }
  .function-group-card > details .tool-card {
    background: var(--card-background-color);
  }
  .function-group-card[data-group-id] + .function-group-card[data-group-id] {
    margin-top: 16px;
  }
  .group-enabled-control { flex: 0 0 auto; }
  @media (max-width: 700px) {
    .function-group-assignment-control { max-width: 100%; }
    .function-group-assignment { max-width: 210px; }
  }
`;

function assignmentOptions(panel, groups, currentId) {
  return [
    `<option value="" ${currentId ? "" : "selected"}>Available on every request</option>`,
    ...(groups || []).map((group) => {
      const disabled = group.enabled === false ? " (disabled)" : "";
      return `<option value="${panel._e(group.id)}" ${group.id === currentId ? "selected" : ""}>${panel._e(group.name)}${disabled}</option>`;
    }),
  ].join("");
}

function functionGroupAssignment(panel, tool, index) {
  const name = tool?.spec?.name;
  if (!name) return "";
  const groups = (panel._draft || panel._result?.config || {}).function_groups || [];
  const current = groups.find((group) => (group.functions || []).includes(name));
  const title = current?.enabled === false
    ? `Function group: ${current.name}. This group is currently disabled.`
    : `Function group: ${current?.name || "Available on every request"}`;
  return `<label class="function-group-assignment-control${current?.enabled === false ? " is-disabled-group" : ""}" title="${panel._e(title)}"><span class="sr-only">Function group for ${panel._e(name)}</span><ha-icon class="function-group-assignment-icon" icon="mdi:folder-outline" aria-hidden="true"></ha-icon><select class="function-group-assignment" data-index="${index}" aria-label="Function group for ${panel._e(name)}">${assignmentOptions(panel, groups, current?.id || "")}</select></label>`;
}

function functionToolCard(panel, tool, allTools) {
  const index = allTools.indexOf(tool);
  const assignment = functionGroupAssignment(panel, tool, index);
  if (isHALlmTool(tool)) return renderHAToolCard(panel, tool, index, assignment);
  const enabled = isFunctionToolEnabled(tool);
  return `<article class="list-card tool-card ${enabled ? "" : "is-disabled"}" data-tool-key="${panel._e(tool.spec?.name || "")}" data-tool-index="${index}" data-tool-search="${panel._e(`${tool.spec?.name || ""} ${tool.spec?.description || ""} ${tool.function?.type || ""} ${enabled ? "enabled" : "disabled"}`.toLowerCase())}"><div class="card-main"><div class="tool-title"><h4>${panel._e(tool.spec?.name||"Unnamed tool")}</h4><span class="type-badge">${panel._e(tool.function?.type||"Unknown type")}</span>${enabled ? "" : '<span class="disabled-badge">Disabled</span>'}</div><p class="description">${panel._e(tool.spec?.description||"No description")}</p>${assignment}</div><div class="actions tool-card-actions"><label class="tool-enabled-control"><span>Enabled</span><span class="switch-control"><input class="tool-enabled" data-index="${index}" type="checkbox" role="switch" aria-label="Enable ${panel._e(tool.spec?.name||"Function Tool")}" ${enabled ? "checked" : ""}><span class="switch-track" aria-hidden="true"></span></span></label><button type="button" class="secondary edit-tool" data-index="${index}">Edit</button><button type="button" class="secondary duplicate-tool" data-index="${index}">Duplicate</button><button type="button" class="danger delete-tool" data-index="${index}">Delete</button></div></article>`;
}

function functionGroupCard(panel, group, tools) {
  const enabled = isFunctionGroupEnabled(group);
  const mode = group.loading_mode === "on_demand" ? "Load when needed" : "Always available";
  const members = group.tools.map((tool) => functionToolCard(panel,tool,tools)).join("");
  return `<article class="function-group-card ${enabled ? "" : "is-disabled"}" data-group-id="${panel._e(group.id)}" data-group-search="${panel._e(`${group.name} ${group.description} ${group.id}`.toLowerCase())}"><div class="function-group-heading"><div><div class="tool-title"><h3>${panel._e(group.name)}</h3><span class="availability-badge ${group.loading_mode === "on_demand" ? "on-demand" : ""}">${mode}</span><span class="function-count">${functionToolCountLabel(group.tools)}</span>${enabled ? "" : `<span class="availability-badge group-disabled-badge">Disabled</span>`}</div><p>${panel._e(group.description)}</p></div><div class="actions"><label class="tool-enabled-control group-enabled-control" title="Disable the group without changing the enabled state of its member Function Tools"><span>Enabled</span><span class="switch-control"><input type="checkbox" role="switch" class="group-enabled" data-group-id="${panel._e(group.id)}" aria-label="Enable Function Group ${panel._e(group.name)}" ${enabled ? "checked" : ""}><span class="switch-track" aria-hidden="true"></span></span></label><button type="button" class="secondary edit-group" data-group-id="${panel._e(group.id)}" ${enabled ? "" : "disabled"} aria-label="Edit Function Group ${panel._e(group.name)}" title="${enabled ? `Edit Function Group ${panel._e(group.name)}` : "Enable this Function Group before editing it"}">Edit</button><button type="button" class="danger delete-group" data-group-id="${panel._e(group.id)}" aria-label="Delete Function Group ${panel._e(group.name)}" title="Delete Function Group ${panel._e(group.name)}">Delete</button></div></div><details class="eoc-details-base"><summary>Show member functions</summary><div class="list tool-list">${members || panel._empty("This group has no functions yet. Edit it to choose functions.")}</div></details></article>`;
}

export function renderTools(panel, {repairCards = ""} = {}) {
  const config = panel._draft || panel._result?.config || {};
  const tools = config.functions || [];
  const enabledCount = tools.filter(isFunctionToolEnabled).length;
  const categories = categorizeFunctionTools(config);
  const ungrouped = categories.alwaysAvailable.map((tool)=>functionToolCard(panel,tool,tools)).join("");
  return `<style data-function-group-assignment>${FUNCTION_GROUP_ASSIGNMENT_STYLE}</style><section class="content-card tools-surface"><div class="section-heading"><div><span class="setting-label-row"><h2>Function Tools & Groups</h2>${helpButton(panel,"function_tools")}</span><p>Function Tools give the assistant actions beyond normal Home Assistant access. Changes to functions and groups save immediately. <button type="button" class="guide-topic-link guide-link" data-guide-topic="functions">Learn more</button></p><small data-function-totals>${enabledCount} enabled · ${tools.length - enabledCount} disabled · ${categories.groups.length} groups</small></div><div class="actions"><button type="button" class="secondary" id="add-group">+ Create group</button><button type="button" class="secondary" id="add-ha-tools">+ Add LLM Tools</button><button type="button" id="add-tool">+ Add Function Tool</button></div></div><div class="notice function-groups-help"><strong>Loading groups only when needed</strong><p>The assistant initially sees each group's name and description, then loads its full tool instructions if the current task needs them. This reduces input-token usage but may add one model round-trip the first time a group is used.</p></div><label class="tool-search"><span class="sr-only">Search functions and groups</span><input id="tool-search" type="search" placeholder="Search functions and groups..." aria-label="Search functions and groups"></label><div class="function-groups">${repairCards}<article class="function-group-card always-card" data-group-search="always available ungrouped general"><div class="function-group-heading"><div><div class="tool-title"><h3>Available on every request</h3><span class="function-count">${functionToolCountLabel(categories.alwaysAvailable)}</span></div><p>These ungrouped functions send their full instructions with every request, so the assistant can use them immediately.</p></div></div><details class="eoc-details-base" open><summary>Show included functions</summary><div class="list tool-list">${ungrouped || panel._empty("No ungrouped functions.")}</div></details></article>${categories.groups.map((group)=>functionGroupCard(panel,group,tools)).join("")||panel._empty("No groups yet. Existing functions remain available on every request until you create one.")}</div><div id="function-search-empty" class="empty-state" role="status" hidden><h3>No functions or groups match your search</h3><button type="button" class="secondary" id="clear-tool-search">Clear search</button></div><div class="section-actions tools-actions"><button type="button" class="secondary" id="refresh-ha-tools" ${tools.some(isHALlmTool)?"":"hidden"}>Refresh HA tool availability</button><button type="button" class="secondary" id="validate-tools">Check tool configuration</button><span id="tool-status" class="validation" aria-live="polite"></span></div></section>`;
}

// Only DOM references and presentation signatures are retained here. Mutations
// still synchronize _draft/_configData from the backend before reconciling.
const toolCollections = new WeakMap();
const functionSearchTokenCache = new WeakMap();

function cachedFunctionSearchTokens(node, searchableText) {
  const value = String(searchableText || "");
  const cached = functionSearchTokenCache.get(node);
  if (cached?.value === value) return cached.tokens;
  const tokens = searchTokens(value);
  functionSearchTokenCache.set(node, {value, tokens});
  return tokens;
}

function applyFunctionSearch(panel) {
  const root = panel.shadowRoot;
  const query = root.querySelector("#tool-search")?.value || "";
  const hasQuery = Boolean(query.trim());
  const queryTokens = hasQuery ? searchTokens(query) : [];
  let visible = 0;
  root.querySelectorAll(".function-group-card").forEach(card => {
    const groupMatch = hasQuery && matchesFunctionSearchTokens(
      queryTokens,
      cachedFunctionSearchTokens(card, card.dataset.groupSearch),
    );
    let toolMatch = false;
    card.querySelectorAll(".tool-card").forEach(tool => {
      const matches = !hasQuery || groupMatch || matchesFunctionSearchTokens(
        queryTokens,
        cachedFunctionSearchTokens(tool, tool.dataset.toolSearch),
      );
      if (tool.hidden === matches) tool.hidden = !matches;
      toolMatch ||= matches;
    });
    const hidden = hasQuery && !groupMatch && !toolMatch;
    if (card.hidden !== hidden) card.hidden = hidden;
    if (!hidden) visible++;
    if (hasQuery && toolMatch) {
      const details = card.querySelector("details");
      if (details && !details.open) details.open = true;
    }
  });
  const empty = root.querySelector("#function-search-empty");
  if (empty) empty.hidden = !hasQuery || visible > 0;
  const noGroups = toolCollections.get(root.querySelector(".tools-surface"))?.noGroups;
  if (noGroups) noGroups.hidden = hasQuery;
}

function prepareToolsCollection(panel) {
  const host = panel.shadowRoot.querySelector(".tools-surface");
  if (!host || toolCollections.has(host)) return;
  toolCollections.set(host, {
    tools: adoptKeyedElements(host, "[data-tool-key]", "toolKey"),
    groups: adoptKeyedElements(host, ".function-group-card[data-group-id]", "groupId"),
    always: {node: host.querySelector(".always-card")},
    repair: host.querySelector(".function-repair-attention"),
    repairSignature: JSON.stringify(panel._result?.function_repair),
    noGroups: [...host.querySelector(".function-groups").children].find(node => node.classList.contains("empty")),
  });
  reconcileTools(panel);
}

export function reconcileTools(panel, {repairCards} = {}) {
  const host = panel.shadowRoot.querySelector(".tools-surface");
  const state = toolCollections.get(host);
  if (!state) return false;
  const config = panel._draft || panel._result?.config || {};
  const tools = config.functions || [];
  const groups = config.function_groups || [];
  const categories = indexFunctionToolGroups(config);
  const membership = categories.membership;
  const choices = JSON.stringify(groups.map(({id, name, enabled}) => ({id, name, enabled})));
  const toolNodes = new Map();
  tools.forEach((tool, index) => {
    const name = tool.spec?.name || "";
    const info = isHALlmTool(tool) && panel._haCatalogAgent === panel._agentId ? panel._haCatalog?.saved?.[name] : null;
    const record = keyedElement(state.tools, name, JSON.stringify([tool, info]), () => functionToolCard(panel, tool, tools));
    const card = record.node;
    record.indices ||= [card, ...card.querySelectorAll("[data-index]")];
    for (const node of record.indices) setAttribute(node, node === card ? "data-tool-index" : "data-index", index);
    const current = membership.get(name);
    const assignment = `${choices}|${current?.id || ""}`;
    if (record.assignment !== undefined && record.assignment !== assignment) {
      const old = card.querySelector(".function-group-assignment-control");
      old?.replaceWith(elementFromMarkup(functionGroupAssignment(panel, tool, index)));
      record.indices = null;
    }
    record.assignment = assignment;
    toolNodes.set(name, card);
  });

  function updateMembers(record, members, emptyText) {
    record.list ||= record.node.querySelector(".tool-list");
    record.count ||= record.node.querySelector(".function-count");
    setText(record.count, functionToolCountLabel(members));
    if (!members.length) record.empty ||= record.list.querySelector(".empty") || elementFromMarkup(panel._empty(emptyText));
    placeChildren(record.list, members.length ? members.map(tool => toolNodes.get(tool.spec.name)) : [record.empty]);
  }
  updateMembers(state.always, categories.alwaysAvailable, "No ungrouped functions.");
  const groupNodes = categories.groups.map(group => {
    let record = state.groups.get(group.id);
    if (!record) {
      record = {node: elementFromMarkup(functionGroupCard(panel, {...group, tools: []}, tools))};
      state.groups.set(group.id, record);
    }
    const {functions, tools: members, ...metadata} = group;
    const signature = JSON.stringify(metadata);
    if (record.signature !== undefined && record.signature !== signature) {
      const fresh = elementFromMarkup(functionGroupCard(panel, {...group, tools: []}, tools));
      record.node.querySelector(".function-group-heading").replaceWith(fresh.querySelector(".function-group-heading"));
      record.node.className = fresh.className;
      setAttribute(record.node, "data-group-search", fresh.dataset.groupSearch);
      record.count = null;
    }
    record.signature = signature;
    updateMembers(record, members, "This group has no functions yet. Edit it to choose functions.");
    return record.node;
  });
  if (repairCards !== undefined && state.repairMarkup !== repairCards) {
    // Repair cards have a separate owner; only rebuild that small section when
    // its rendered validation/quarantine information actually changes.
    if (state.repairSignature !== JSON.stringify(panel._result?.function_repair) || !state.repair) state.repair = repairCards ? elementFromMarkup(repairCards) : null;
    state.repairSignature = JSON.stringify(panel._result?.function_repair);
    state.repairMarkup = repairCards;
  }
  if (!groupNodes.length) state.noGroups ||= elementFromMarkup(panel._empty("No groups yet. Existing functions remain available on every request until you create one."));
  placeChildren(host.querySelector(".function-groups"), [
    ...(state.repair ? [state.repair] : []), state.always.node,
    ...(groupNodes.length ? groupNodes : [state.noGroups]),
  ]);
  pruneKeys(state.tools, new Set(tools.map(tool => tool.spec?.name || "")));
  pruneKeys(state.groups, new Set(groups.map(group => group.id)));
  const enabled = tools.filter(isFunctionToolEnabled).length;
  setText(host.querySelector("[data-function-totals]"), `${enabled} enabled · ${tools.length - enabled} disabled · ${groups.length} groups`);
  const refresh = host.querySelector("#refresh-ha-tools");
  const hideRefresh = !tools.some(isHALlmTool);
  if (refresh.hidden !== hideRefresh) refresh.hidden = hideRefresh;
  applyFunctionSearch(panel);
  return true;
}

function collectionToolIndex(panel, control) {
  const name = control.closest("[data-tool-key]")?.dataset.toolKey;
  return (panel._draft.functions || []).findIndex(tool => tool.spec?.name === name);
}

function queueFunctionGroupMutation(panel, operation) {
  const previous = panel._eocFunctionGroupUiTail || Promise.resolve();
  const pending = previous.catch(() => {}).then(async () => {
    const backendTail = panel._eocFunctionMutationTail;
    if (backendTail) await backendTail.catch(() => {});
    return operation();
  });
  const tracked = pending.finally(() => {
    if (panel._eocFunctionGroupUiTail === tracked) panel._eocFunctionGroupUiTail = null;
  });
  panel._eocFunctionGroupUiTail = tracked;
  return tracked;
}

function bindToolCollection(panel) {
  const root = panel.shadowRoot;
  const host = root.querySelector(".tools-surface");
  if (!host || host.__eocToolsBound) return;
  host.__eocToolsBound = true;
  host.addEventListener("input", event => {
    if (event.target.id === "tool-search") applyFunctionSearch(panel);
  });
  host.addEventListener("click", async event => {
    const button = event.target.closest?.("button");
    if (!button || button.disabled) return;
    if (button.id === "clear-tool-search") {
      const search = root.querySelector("#tool-search");
      search.value = "";
      applyFunctionSearch(panel);
      search.focus();
      return;
    }
    if (button.id === "add-tool") return openTool(panel);
    if (button.id === "add-group") return openFunctionGroup(panel);
    if (button.matches(".edit-group")) return openFunctionGroup(panel, button.dataset.groupId);
    if (button.matches(".delete-group")) {
      const group = (panel._draft.function_groups || []).find(item => item.id === button.dataset.groupId);
      panel._eocDeletingFunctionGroups ||= new Set();
      if (!group || panel._eocDeletingFunctionGroups.has(group.id)
        || !await panel._confirm("Delete function group?", `The group “${group.name}” will be removed. Its functions will not be deleted; they will move to Always available.`, "Delete group")) return;
      if (panel._eocDeletingFunctionGroups.has(group.id)) return;
      panel._eocDeletingFunctionGroups.add(group.id);
      const idleLabel = button.textContent;
      button.disabled = true;
      button.textContent = "Deleting…";
      button.setAttribute("aria-busy", "true");
      try {
        const result = await queueFunctionGroupMutation(panel, () => panel._call("tools", "delete_group", {group_id: group.id, confirm: true}));
        synchronizePersistedFunctions(panel, result);
        panel._toast("Function group deleted");
        panel._render();
      } catch (err) {
        button.disabled = false;
        button.textContent = idleLabel;
        button.removeAttribute("aria-busy");
        panel._toast(`Unable to delete group: ${err.message || String(err)}`, true);
      } finally {
        panel._eocDeletingFunctionGroups.delete(group.id);
      }
      return;
    }
    const index = collectionToolIndex(panel, button);
    const tool = panel._draft.functions[index];
    if (!tool) return;
    if (button.matches(".edit-tool")) return openTool(panel, index);
    if (button.matches(".duplicate-tool")) {
      const copy = clone(tool);
      const names = new Set(panel._draft.functions.map(item => item.spec?.name));
      const original = copy.spec.name;
      copy.spec.name = `${original}_copy`;
      let suffix = 2;
      while (names.has(copy.spec.name)) copy.spec.name = `${original}_copy_${suffix++}`;
      return openTool(panel, null, copy);
    }
    if (button.matches(".delete-tool")) {
      const functionName = tool.spec?.name;
      panel._eocDeletingFunctions ||= new Set();
      if (!functionName || panel._eocDeletingFunctions.has(functionName)) return;
      if (!await panel._confirm(isHALlmTool(tool) ? "Remove HA LLM Tool?" : "Delete function tool?", isHALlmTool(tool)
        ? `Remove “${haToolName(tool)}” from this agent and its groups? The underlying Home Assistant capability remains unchanged. Saved dependencies must be removed first.`
        : `The Function Tool “${tool.spec?.name || "Unnamed"}” will be deleted and removed from any Function Group. Deletion is refused while Request Rules or Guest Mode still reference it.`, isHALlmTool(tool) ? "Remove tool" : "Delete function")) return;
      if (panel._eocDeletingFunctions.has(functionName)) return;
      panel._eocDeletingFunctions.add(functionName);
      const idleLabel = button.textContent;
      button.disabled = true;
      button.textContent = "Deleting…";
      button.setAttribute("aria-busy", "true");
      try {
        const result = await panel._call("tools", "delete", {name: functionName, confirm: true});
        const syncStarted = globalThis.performance?.now?.() ?? Date.now();
        synchronizePersistedFunctions(panel, result);
        const syncCompleted = globalThis.performance?.now?.() ?? Date.now();
        panel._toast("Function deleted");
        panel._render();
        const renderCompleted = globalThis.performance?.now?.() ?? Date.now();
        panel._recordFunctionMutationUi?.(result, {
          syncMs: syncCompleted - syncStarted,
          renderAndReconcileMs: renderCompleted - syncCompleted,
        });
      } catch (err) {
        button.disabled = false;
        button.textContent = idleLabel;
        button.removeAttribute("aria-busy");
        panel._toast(`Unable to delete function: ${err.message || String(err)}`, true);
      } finally {
        panel._eocDeletingFunctions.delete(functionName);
      }
    }
  });
  host.addEventListener("change", async event => {
    const input = event.target;
    if (!input.matches?.(".tool-enabled,.group-enabled") || input.disabled) return;
    const isGroup = input.matches(".group-enabled");
    const item = isGroup
      ? (panel._draft.function_groups || []).find(group => group.id === input.dataset.groupId)
      : panel._draft.functions[collectionToolIndex(panel, input)];
    if (!item) return;
    const enabled = input.checked;
    input.disabled = true;
    try {
      const result = isGroup
        ? await queueFunctionGroupMutation(panel, () => {
            const latest = (panel._draft.function_groups || []).find(group => group.id === item.id);
            if (!latest) throw new Error("The Function Group no longer exists");
            return panel._call("tools", "save_group", {group: {...latest, enabled}, original_id: latest.id});
          })
        : await panel._call("tools", "set_enabled", {name: item.spec.name, enabled});
      synchronizePersistedFunctions(panel, result);
      const references = result.references || {};
      const affected = (references.request_rules || []).length + (references.guest_mode ? 1 : 0);
      panel._toast(isGroup
        ? enabled ? "Function group enabled" : "Function group disabled; member Function Tool settings were kept"
        : enabled ? "Function enabled" : affected ? `Function disabled; ${affected} saved reference${affected === 1 ? "" : "s"} remain configured but unavailable until it is re-enabled` : "Function disabled");
      panel._render();
    } catch (err) { input.checked = item.enabled !== false; panel._toast(`Unable to update ${isGroup ? "Function Group" : "function"}: ${err.message || String(err)}`, true); }
    finally { input.disabled = false; }
  });
}

export function synchronizePersistedFunctions(panel, result) {
  const functions = clone(result.functions || []);
  const function_groups = clone(result.function_groups || []);
  panel._configData = {
    ...panel._configData,
    config: {...panel._configData.config, functions, function_groups},
  };
  panel._result = panel._configData;
  panel._draft = {
    ...panel._draft,
    functions: clone(functions),
    function_groups: clone(function_groups),
  };
  panel._syncConfigDirty();
}

function setToolEditorLoading(root, loading) {
  const textarea = root.querySelector("#tool-yaml");
  textarea.readOnly = loading;
  // The native HA YAML editor is inserted beside the textarea in this label.
  // Inert gates both editors without depending on HA's internal editor API.
  if (textarea.parentElement) textarea.parentElement.inert = loading;
  for (const id of ["tool-save", "tool-validate", "built-in-function"]) {
    const control = root.querySelector(`#${id}`);
    if (control) control.disabled = loading;
  }
  root.querySelector("#tool-dialog").setAttribute("aria-busy", String(loading));
}

export async function openTool(panel, index = null, initialTool = null) {
  const tool = initialTool || (index === null ? null : panel._draft.functions[index]);
  panel._toolIndex = index;
  panel._toolOriginalName = index === null ? null : tool?.spec?.name || null;
  const root = panel.shadowRoot;
  const dialog = root.querySelector("#tool-dialog");
  let editor = getToolYamlEditor(panel);
  const status = root.querySelector("#tool-error");
  const agentId = panel._agentId;
  const loadToken = {};
  panel._toolEditorLoad = loadToken;
  const isCurrent = () => panel._toolEditorLoad === loadToken
    && panel._agentId === agentId && dialog.open
    && root.querySelector("#tool-dialog") === dialog;
  root.querySelector("#tool-dialog-title").textContent = index === null ? "Add Function Tool" : "Edit Function Tool";
  root.querySelector("#tool-dialog-meta").textContent = "Loading YAML...";
  setToolEditorLoading(root, true);
  editor.setYaml("");
  status.className = "validation";
  status.textContent = "Loading editor...";
  root.querySelector("#built-in-picker").hidden = index !== null || Boolean(initialTool);
  panel._toolRevision = panel._configData?.revision;
  panel._toolInitialYaml = null;
  dialog.showModal();
  try {
    const nativeYaml = await import("./agent-config-native-yaml.js");
    if (!isCurrent()) return;
    nativeYaml.bindNativeToolYaml(panel);
    editor = getToolYamlEditor(panel);
    let response;
    if (tool) {
      response = await panel._call("tools", "serialize", {tool});
    } else {
      const [starter, catalog] = await Promise.all([
        panel._call("tools", "starter"),
        panel._call("tools", "built_in_catalog", {tools: panel._draft.functions || []}),
      ]);
      if (!isCurrent()) return;
      response = starter;
      panel._builtInFunctions = catalog.functions || [];
      const selector = root.querySelector("#built-in-function");
      selector.innerHTML = '<option value="">Insert Built-in Function…</option>' + panel._builtInFunctions.map((preset) => `<option value="${panel._e(preset.implementation)}" ${preset.already_configured ? "disabled" : ""}>${panel._e(preset.label)}${preset.already_configured ? " — Already configured" : ""}</option>`).join("");
    }
    if (!isCurrent()) return;
    editor.setYaml(response.yaml);
    panel._toolInitialYaml = response.yaml;
    panel._toolReplaceableYaml = response.yaml;
    root.querySelector("#tool-dialog-meta").textContent = index === null ? "New tool / YAML" : `${tool.spec?.name || "Unnamed"} / ${tool.function?.type || "unknown"}`;
    status.textContent = "Edit the YAML, then save the function.";
    setToolEditorLoading(root, false);
    editor.focus();
  } catch (err) {
    if (!isCurrent()) return;
    status.className = "validation invalid";
    status.textContent = err.message || String(err);
    // Keep the unloaded editor/save disabled; Cancel remains available.
    dialog.setAttribute("aria-busy", "false");
  }
}

function toolErrorText(errors={}) { return Object.entries(errors).map(([key,value])=>`${key}: ${value}`).join(" "); }

async function validateDialogTool(panel) {
  const root=panel.shadowRoot;
  const status=root.querySelector("#tool-error");
  status.className="validation";
  status.textContent="Validating...";
  try {
    const result=await panel._call("tools","validate_yaml",{yaml:getToolYamlEditor(panel).getYaml()});
    if(!result.valid){status.className="validation invalid";status.textContent=`Function configuration is invalid: ${toolErrorText(result.errors)}`;return null;}
    status.className="validation valid";
    status.textContent=`Valid function tool / Name: ${result.name} / Type: ${result.type}`;
    root.querySelector("#tool-dialog-meta").textContent=`${result.name} / ${result.type}`;
    return result.config;
  } catch(err){status.className="validation invalid";status.textContent=err.message||String(err);return null;}
}

function renderGroupFunctionChoices(panel, selected = []) {
  const root=panel.shadowRoot;
  const selectedNames=new Set(selected);
  const tools=panel._draft.functions || [];
  const list=root.querySelector("#group-functions");
  list.innerHTML=tools.map((tool)=>{const name=tool.spec?.name||"";const enabled=isFunctionToolEnabled(tool);return `<label class="group-function-choice ${enabled?"":"is-disabled"}" data-choice-search="${panel._e(`${haToolName(tool)} ${tool.function?.source_id||""} ${toolDescription(panel,tool)} ${enabled?"enabled":"disabled"}`.toLowerCase())}"><input type="checkbox" value="${panel._e(name)}" ${selectedNames.has(name)?"checked":""}><span><strong>${panel._e(haToolName(tool))}${enabled?"":" · Disabled"}</strong><small>${panel._e(toolDescription(panel,tool))}</small></span></label>`;}).join("") || panel._empty("Add function tools before assigning them to a group.");
}

function openFunctionGroup(panel, groupId = null) {
  const root=panel.shadowRoot;
  const group=(panel._draft.function_groups||[]).find((item)=>item.id===groupId);
  panel._groupOriginalId=group?.id||null;
  panel._groupIdEdited=Boolean(group);
  root.querySelector("#group-dialog-title").textContent=group?"Edit function group":"Create function group";
  root.querySelector("#group-name").value=group?.name||"";
  root.querySelector("#group-id").value=group?.id||"";
  root.querySelector("#group-description").value=group?.description||"";
  root.querySelector("#group-loading-mode").value=group?.loading_mode||"on_demand";
  root.querySelector("#group-function-search").value="";
  root.querySelector("#group-error").textContent="";
  renderGroupFunctionChoices(panel,group?.functions||[]);
  panel._groupRevision = panel._configData?.revision;
  root.querySelector("#group-dialog").showModal();
  panel._captureDialogBaseline?.(root.querySelector("#group-dialog"));
  root.querySelector("#group-name").focus();
}

async function saveFunctionGroup(panel) {
  const root=panel.shadowRoot;
  const button = root.querySelector("#group-save");
  if (button.disabled) return;
  const name=root.querySelector("#group-name").value.trim();
  const id=root.querySelector("#group-id").value.trim();
  const description=root.querySelector("#group-description").value.trim();
  const loading_mode=root.querySelector("#group-loading-mode").value;
  const guest_allowed=(panel._draft.function_groups||[]).find((group)=>group.id===panel._groupOriginalId)?.guest_allowed===true;
  const functions=[...root.querySelectorAll("#group-functions input:checked")].map((input)=>input.value);
  const error=root.querySelector("#group-error");
  if(!name){error.textContent="Group name is required.";return;}
  if(!/^[a-z][a-z0-9_-]{0,63}$/.test(id)){error.textContent="Group ID must start with a lowercase letter and use only lowercase letters, numbers, underscores, or hyphens.";return;}
  if(!description){error.textContent="Add a concise description so the model knows when this group is relevant.";return;}
  error.textContent="Saving...";
  panel._setSaving(button, true);
  try {
    const result=await queueFunctionGroupMutation(panel,()=>panel._call("tools","save_group",{group:{id,name,description,loading_mode,functions,guest_allowed},...(panel._groupOriginalId?{original_id:panel._groupOriginalId}:{})}));
    synchronizePersistedFunctions(panel,result);
    root.querySelector("#group-dialog").close();
    panel._toast("Function group saved");
    panel._render();
  } catch(err) { error.textContent=err.message||String(err); }
  finally { panel._setSaving(button, false); }
}

async function assignToolToGroup(panel, select) {
  const name = select.closest?.("[data-tool-key]")?.dataset.toolKey;
  const targetId = select.value;
  if (!name) return;
  select.disabled = true;
  let previousGroupId = "";
  try {
    const result = await queueFunctionGroupMutation(panel, async () => {
      const config = panel._draft || panel._result?.config || {};
      const groups = config.function_groups || [];
      const current = groups.find(group => (group.functions || []).includes(name));
      previousGroupId = current?.id || "";
      if (previousGroupId === targetId) return null;
      const target = targetId ? groups.find(group => group.id === targetId) : current;
      if (!target) throw new Error("The selected Function Group no longer exists");
      const functions = targetId
        ? [...new Set([...(target.functions || []), name])]
        : (target.functions || []).filter(item => item !== name);
      return panel._call("tools", "save_group", {
        group: {...target, functions}, original_id: target.id,
      });
    });
    if (!result) return;
    synchronizePersistedFunctions(panel, result);
    const target = (panel._draft.function_groups || []).find(group => group.id === targetId);
    panel._toast(targetId ? `${name} moved to ${target?.name || targetId}` : `${name} is now available on every request`);
    panel._render();
  } catch (err) {
    select.value = previousGroupId;
    panel._toast(`Unable to change Function Group: ${err.message || String(err)}`, true);
  } finally { select.disabled = false; }
}

export function bindTools(panel) {
  ensureRouteStyle(panel);
  const root=panel.shadowRoot;
  const host = root.querySelector(".tools-surface");
  if (host && !host.__eocAssignmentBound) {
    host.__eocAssignmentBound = true;
    host.addEventListener("change", event => {
      if (event.target.matches?.(".function-group-assignment") && !event.target.disabled) void assignToolToGroup(panel, event.target);
    });
  }
  // Catalogue discovery and its dialog code are optional for the initial list.
  const attachHA = () => import("./ha-llm-tools.js").then(({bindHALlmTools}) => {
    if (panel.shadowRoot === root && root.querySelector(".tools-surface") === host) {
      bindHALlmTools(panel, synchronizePersistedFunctions);
    }
  });
  if ((panel._draft?.functions || []).some(isHALlmTool)) void attachHA().catch(() => {});
  for (const id of ["add-ha-tools", "refresh-ha-tools"]) {
    const button = root.querySelector(`#${id}`);
    if (!button) continue;
    const firstClick = async () => {
      button.removeEventListener("click", firstClick);
      button.disabled = true;
      try {
        await attachHA();
        if (button.isConnected) {
          button.disabled = false;
          button.click();
        }
      } catch (err) {
        button.disabled = false;
        button.addEventListener("click", firstClick);
        panel._toast(err.message || String(err), true);
      }
    };
    button.addEventListener("click", firstClick);
  }
  bindHelp(panel);
  prepareToolsCollection(panel);
  bindToolCollection(panel);
  root.querySelector("#validate-tools")?.addEventListener("click",async()=>{const status=root.querySelector("#tool-status");try{const result=await panel._call("tools","validate_current");status.className=`validation ${result.valid?"valid":"invalid"}`;status.textContent=result.valid?"All saved tools and groups are valid":toolErrorText(result.errors);}catch(err){status.className="validation invalid";status.textContent=err.message||String(err);}});
  getToolYamlEditor(panel);
  root.querySelector("#built-in-function")?.addEventListener("change",async(event)=>{const selector=event.target;const preset=(panel._builtInFunctions||[]).find((item)=>item.implementation===selector.value);if(!preset)return;const editor=getToolYamlEditor(panel);const replaceable=canReplaceToolYamlWithoutConfirmation(editor.getYaml(),panel._toolReplaceableYaml);if(!replaceable&&!await panel._confirm("Replace current YAML with this built-in function preset?","Your current Function Tool YAML will be replaced in the editor. Nothing is saved until you select Save.","Replace YAML")){selector.value="";return;}editor.setYaml(preset.yaml);panel._toolReplaceableYaml=preset.yaml;await validateDialogTool(panel);});
  root.querySelector("#tool-cancel")?.addEventListener("click",()=>root.querySelector("#tool-dialog").close());
  root.querySelector("#tool-dialog")?.addEventListener("cancel",()=>{panel._toolIndex=null;});
  root.querySelector("#tool-dialog")?.addEventListener("close", (event) => {
    // Ignore queued close events after navigation, DOM replacement or reopening.
    if (event.target !== root.querySelector("#tool-dialog") || event.target.open) return;
    panel._toolEditorLoad = null;
    setToolEditorLoading(root, false);
  });
  root.querySelector("#tool-validate")?.addEventListener("click",()=>validateDialogTool(panel));
  root.querySelector("#tool-save")?.addEventListener("click", async () => {
    const button = root.querySelector("#tool-save");
    if (button.disabled) return;
    panel._setSaving(button, true);
    try {
      const tool = await validateDialogTool(panel);
      if (!tool) return;
      const result = await panel._call("tools", "save", {tool, ...(panel._toolOriginalName ? {original_name: panel._toolOriginalName} : {})});
      synchronizePersistedFunctions(panel, result);
      root.querySelector("#tool-dialog").close();
      panel._toast("Changes saved");
      panel._render();
    } catch (err) {
      root.querySelector("#tool-error").className = "validation invalid";
      root.querySelector("#tool-error").textContent = err.message || String(err);
    } finally { panel._setSaving(button, false); }
  });
  root.querySelector("#group-name")?.addEventListener("input",(event)=>{if(!panel._groupIdEdited)root.querySelector("#group-id").value=functionGroupIdFromName(event.target.value);});
  root.querySelector("#group-id")?.addEventListener("input",()=>{panel._groupIdEdited=true;});
  root.querySelector("#group-function-search")?.addEventListener("input",(event)=>{const queryTokens=searchTokens(event.target.value);root.querySelectorAll(".group-function-choice").forEach((choice)=>{choice.hidden=!matchesFunctionSearchTokens(queryTokens,cachedFunctionSearchTokens(choice,choice.dataset.choiceSearch));});});
  root.querySelector("#group-cancel")?.addEventListener("click",()=>root.querySelector("#group-dialog").close());
  root.querySelector("#group-save")?.addEventListener("click",()=>saveFunctionGroup(panel));
}


export function configurationDialogs(panel) {
  const view = panel._viewKey?.();
  return [
    (!view || view === "assistant/prompt-context") ? `<dialog id="prompt-preview-dialog" class="editor-dialog wide prompt-preview-dialog" aria-labelledby="prompt-preview-title"><div class="dialog-header"><div><h2 id="prompt-preview-title">Preview effective request</h2><p class="dialog-meta">Locally assembled fresh-request content</p></div></div><div class="dialog-body prompt-preview-body"><div class="notice"><strong>Current request preview</strong><p>Shows locally assembled content that would accompany a brand-new message now. Current templates and Home Assistant context are resolved. User input and conversation history are excluded.</p><p>This may contain private entity state, memory data, Knowledge context, user instructions, and Function Tool schemas. Provider-internal framing and hidden server-side content cannot be inspected.</p></div><div class="request-preview-metrics"><strong id="request-footprint">Calculating...</strong><span id="function-group-savings"></span></div><div id="request-preview-sections" class="request-preview-sections"></div><ul id="prompt-preview-notes" class="prompt-preview-notes"></ul><div id="prompt-preview-status" class="validation" role="status" aria-live="polite"></div></div><div class="dialog-actions"><button type="button" class="secondary" id="prompt-preview-close">Close</button><button type="button" id="copy-prompt-preview" disabled>Copy all</button></div></dialog>` : "",
    `<dialog id="import-dialog" class="editor-dialog wide" aria-labelledby="import-dialog-title"><div class="dialog-header"><h2 id="import-dialog-title">Import agent configuration</h2></div><div class="dialog-body">${panel._configDirty ? `<div class="notice"><strong>Unsaved shared draft</strong><p>Creating a new agent preserves this draft. Overwriting the current agent discards it after confirmation.</p></div>` : ""}<label>Exported JSON or YAML<textarea id="import-document" class="yaml-editor" spellcheck="false"></textarea></label><div class="mode-row"><label><input type="radio" name="import-mode" value="new" checked> Create a new agent</label><label><input type="radio" name="import-mode" value="current"> Overwrite current agent</label></div><div id="import-summary" class="validation" aria-live="polite">Validate the document to preview it.</div></div><div class="dialog-actions"><button type="button" class="secondary" id="import-cancel">Cancel</button><button type="button" class="secondary" id="import-preview">Validate & preview</button><button type="button" id="import-apply" disabled>Import</button></div></dialog>`,
    (!view || view === "capabilities/functions") ? `<dialog id="tool-dialog" class="editor-dialog tool-dialog" aria-labelledby="tool-dialog-title"><div class="dialog-header"><div><span class="setting-label-row"><h2 id="tool-dialog-title">Function Tool</h2>${helpButton(panel,"function_tools")}</span><p id="tool-dialog-meta" class="dialog-meta">YAML</p></div></div><div class="dialog-body tool-dialog-body"><div id="built-in-picker" class="built-in-picker"><label for="built-in-function">Insert Built-in Function</label><select id="built-in-function"><option value="">Insert Built-in Function…</option></select><small>Built-in functions are implemented directly by Extended OpenAI Conversation. Selecting one inserts an editable Function Tool that exposes the capability to the model.</small></div><label class="tool-editor-label"><span class="sr-only">Function Tool YAML</span><textarea data-native-yaml-fallback id="tool-yaml" class="yaml-editor tool-yaml-editor" spellcheck="false" wrap="off" aria-describedby="tool-error"></textarea><ha-yaml-editor id="tool-yaml-native" class="tool-yaml-native-editor" hidden in-dialog aria-label="Function Tool YAML" aria-describedby="tool-error"></ha-yaml-editor></label><div id="tool-error" class="validation" role="status" aria-live="polite"></div></div><div class="dialog-actions"><button type="button" class="secondary" id="tool-cancel">Cancel</button><button type="button" class="secondary" id="tool-validate">Validate</button><button type="button" id="tool-save">Save</button></div></dialog>` : "",
    (!view || view === "capabilities/functions") ? `<dialog id="group-dialog" class="editor-dialog group-dialog" aria-labelledby="group-dialog-title"><div class="dialog-header"><div><h2 id="group-dialog-title">Function group</h2><p>Create a compact capability the model can load only when needed.</p></div></div><div class="dialog-body group-dialog-body"><div class="form-grid"><label>Group name<input id="group-name" maxlength="100" autocomplete="off" placeholder="Reminders"></label><label>Group ID<input id="group-id" maxlength="64" autocomplete="off" spellcheck="false" placeholder="reminders"><small>Stable lowercase ID used by the model. Advanced users may edit it.</small></label></div><label>Description<textarea id="group-description" class="short-textarea" maxlength="500" placeholder="Create and manage scheduled, recurring, and triggered reminders."></textarea><small>Keep this concise; it is included in the compact catalogue sent with normal requests.</small></label><label>Availability<select id="group-loading-mode"><option value="on_demand">Load when needed</option><option value="always">Always available</option></select><small>On-demand groups add one model round-trip the first time they are needed in an active conversation.</small></label><fieldset class="group-functions-fieldset"><legend>Functions</legend><input id="group-function-search" type="search" placeholder="Search functions..." aria-label="Search functions to assign"><small>Selecting a function moves it from any other group. Guest availability is managed centrally on the Guest page.</small><div id="group-functions" class="group-function-choices"></div></fieldset><div id="group-error" class="validation invalid" role="alert"></div></div><div class="dialog-actions"><button type="button" class="secondary" id="group-cancel">Cancel</button><button type="button" id="group-save">Save</button></div></dialog>` : "",
    helpPopover(),
  ].join("");
}
