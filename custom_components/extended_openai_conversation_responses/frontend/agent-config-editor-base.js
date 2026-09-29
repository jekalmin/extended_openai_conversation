import {readConfigurationDraft as readConfig} from "./configuration-controls.js";
import {bindConfigurationInputs, updateConfigurationControl} from "./configuration-inputs.js";
import {modelFieldPresentation, modelFieldNotes, webSearchControlState} from "./agent-config-model-presentation.js";
import {friendlySettingLabel, friendlySettingValue, settingSearchAliases} from "./management-setting-metadata.js";
import {settingBadgesMarkup} from "./management-decision-guidance.js";
import {bindSingleRequestSave} from "./management-actions.js";
import {saveBarMarkup} from "./unsaved-state.js";
import {modelDataControls, bindModelDataControls} from "./model-catalog.js";
import {bindHelp, helpButton, helpPopover} from "./agent-config-help.js";
export {modelDataControls} from "./model-catalog.js";
export {helpButton} from "./agent-config-help.js";

const clone = (value) => JSON.parse(JSON.stringify(value));
const bool = (value) => value ? "checked" : "";
export {skillNamesFromText} from "./configuration-controls.js";
export function invalidateImportPreview(panel, applyButton, summary) {
  panel._importDocument = null;
  if (applyButton) applyButton.disabled = true;
  if (summary) summary.textContent = "Validate the document to preview it.";
}
export const option = (panel, value, selected, label = null, disabled = false) => `<option value="${panel._e(value)}" ${disabled ? "disabled" : ""} ${value === selected ? "selected" : ""}>${panel._e(label || panel._titleCase(value))}</option>`;
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
export const settingSearch = (label, description, key) => `${label} ${description} ${key} ${settingSearchAliases(key)}`.toLowerCase();
export const labelRow = (panel, label, key, helpKey = null, strong = false, value = undefined, disabled = false) => {
  const text = panel._e(friendlySettingLabel(key) || label);
  return `<span class="setting-label-row"><label for="config-${key}">${strong ? `<strong>${text}</strong>` : text}</label>${helpKey ? helpButton(panel, helpKey) : ""}${settingBadgesMarkup(panel, key, value, disabled)}</span>`;
};
export const field = (panel, key, label, value, type = "text", description = "", disabled = false, helpKey = null, forceVisible = false) => {
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
export const select = (panel, key, label, value, options, description = "", disabled = false, helpKey = null, forceVisible = false) => {
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
export const toggle = (panel, key, label, value, description = "", disabled = false, helpKey = null) => {
  const state = key === "web_search" ? webSearchControlState(panel) : {disabled:false, note:""};
  const inactive = disabled || state.disabled;
  return `<div class="config-toggle setting" data-field="${key}" data-setting data-search="${panel._e(settingSearch(label, description, key, helpKey))}"><span class="setting-copy">${labelRow(panel, label, key, helpKey, true, value, inactive)}${description ? `<small>${description}</small>` : ""}${state.note ? `<small class="capability-note">${panel._e(state.note)}</small>` : ""}</span><label class="switch-control" for="config-${key}"><input id="config-${key}" data-config="${key}" data-type="boolean" type="checkbox" role="switch" ${bool(value)} ${inactive ? "disabled" : ""}><span class="switch-track" aria-hidden="true"></span></label></div>`;
};

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

export function section(panel, id, title, description, keywords, body, includeHeading = true) {
  if (panel._configSectionFilter && !panel._configSectionFilter.has(id)) return "";
  const content = typeof body === "function" ? body() : body;
  if (id === "capabilities" && panel._viewKey?.() === "capabilities/web-skills") includeHeading = false;
  return `<section id="config-${id}" class="config-section" data-config-section data-search="${panel._e(`${title} ${description} ${keywords}`.toLowerCase())}">${includeHeading ? `<div class="config-section-heading"><h2 class="eyebrow">${title}</h2><p>${description}</p></div>` : ""}${content}</section>`;
}

export function renderLocalHandling(panel, config) {
  const state = panel._result?.local_handling || {};
  const intents = state.intents || [];
  const excluded = new Set(config.local_intent_exclusions || []);
  const conflicts = state.pipeline_conflicts || [];
  const conflictNames = conflicts.map((item) => item.name).filter(Boolean);
  const conflictText = conflictNames.length === 1
    ? `The Assist pipeline “${conflictNames[0]}” still has Home Assistant's Prefer local handling turned on.`
    : `${conflictNames.length} Assist pipelines using this agent still have Home Assistant's Prefer local handling turned on: ${conflictNames.join(", ")}.`;
  return `<div class="config-stack local-handling-layout">
    ${toggle(panel,"local_intents_enabled","Use Extended OpenAI local handling",config.local_intents_enabled,"Try Home Assistant's built-in commands after Request Rules. Anything that does not match locally, or that you exclude below, continues to your Function Tools or AI model.",state.supported === false)}
    ${state.supported === false ? `<div class="notice local-handling-warning" role="status"><div class="local-handling-warning-title"><ha-icon icon="mdi:alert-outline" aria-hidden="true"></ha-icon><strong>Local handling is not available</strong></div><p>This Home Assistant version does not provide the local command interface this feature needs. Extended OpenAI will continue to use the AI path normally.</p></div>` : ""}
    ${conflicts.length ? `<div class="notice local-handling-warning" role="status"><div class="local-handling-warning-title"><ha-icon icon="mdi:alert-outline" aria-hidden="true"></ha-icon><strong>Home Assistant is already handling some commands first</strong></div><p>${panel._e(conflictText)} Those commands may be completed before they reach this agent, so the choices below cannot apply to them. Turn off <strong>Prefer local handling</strong> for ${conflicts.length === 1 ? "that pipeline" : "those pipelines"} if you want Extended OpenAI to control this order.</p></div>` : ""}
    <details class="local-handling-help">
      <summary>What's the difference from Home Assistant's “Prefer local handling”?</summary>
      <div class="local-handling-help-body"><p>Home Assistant's own option runs before a request reaches Extended OpenAI. That is simple and fast, but Extended OpenAI cannot then apply Request Rules or choose a Function Tool for that command.</p><p><strong>Extended OpenAI local handling</strong> runs after Request Rules, so built-in commands can still stay fast while selected command types continue to Function Tools or the AI model.</p><p><strong>Example:</strong> “Turn on the kitchen light” can stay local, while “turn off the kitchen light in 20 minutes” can continue to a deferred-action Function Tool.</p></div>
    </details>
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

// Production routes load one cohesive section family alongside the editor
// shell. Keep the draft, save bar and section visibility in this shared core.
export function prepareConfigurationSections(panel) {
  const view = panel._viewKey?.();
  panel._configSectionFilter = new Set(panel._configSections || []);
  if (panel._configSectionFilter.has("conversation") && view !== "assistant/conversation") {
    panel._configSectionFilter.add("local");
  }
}

export function renderConfigurationShell(panel, sections) {
  const defaults = panel._result?.defaults || {};
  const intro = panel._viewKey?.() === "capabilities/web-skills"
    ? '<section class="page-intro"><h1>Web search & Skills</h1><p>Choose optional online information and installed instruction sets the assistant may load when needed.</p></section>'
    : "";
  return `${intro}<div class="content-card config-surface">${sections}${saveBar(panel)}<span id="save-bar-anchor" class="sr-only" data-defaults="${panel._e(JSON.stringify(defaults))}"></span></div>`;
}

export function regexRow(panel, rule, index, disabled = false) {
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
    const query = event.target.value;
    root.querySelectorAll("[data-local-intent-choice]").forEach((choice) => {
      choice.hidden = !matchesFunctionSearch(query, choice.dataset.choiceSearch);
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
    const token=(panel._eocRequestPreviewToken||0)+1;
    panel._eocRequestPreviewToken=token;
    const agentId=panel._agentId, view=panel._viewKey?.();
    const current=()=>panel._eocRequestPreviewToken===token
      && panel._agentId===agentId && panel._viewKey?.()===view
      && root.querySelector("#prompt-preview-dialog")===dialog;
    root.querySelector("#request-footprint").textContent="Calculating..."; root.querySelector("#function-group-savings").textContent=""; sections.innerHTML=""; notes.innerHTML=""; status.textContent="Assembling current Home Assistant request..."; status.className="validation"; copyButton.disabled=true; dialog.showModal();
    try {
      const result=await panel._call("configuration","request_preview",{config:readConfig(panel)});
      if(!current())return;
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
      if(!current())return;
      status.textContent=err.message||String(err);
      status.className="validation invalid";
    }
  });
  root.querySelector("#prompt-preview-close")?.addEventListener("click",()=>root.querySelector("#prompt-preview-dialog").close());
  root.querySelector("#copy-prompt-preview")?.addEventListener("click",async()=>{try{const text=(panel._effectiveRequestPreview?.sections||[]).map((section)=>`## ${section.label}\n${section.content}`).join("\n\n");await copyTextToClipboard(text);panel._toast("Effective request copied");}catch(err){panel._toast(`Unable to copy request: ${err.message||String(err)}`,true);}});
  for(const selector of ["#reset-advanced","#reset-model-parameters"])root.querySelector(selector)?.addEventListener("click", () => { ["temperature","top_p","reasoning_effort","service_tier","shorten_tool_call_id","memory_auto_retrieve_limit","memory_retrieval_mode","memory_embedding_model"].forEach((key) => { panel._draft[key]=clone(panel._result.defaults[key]); const input=root.querySelector(`[data-config="${key}"]`); if(input) input.dataset.type==="boolean" ? input.checked=panel._draft[key] : input.value=panel._draft[key]; }); panel._setConfigDirty(true); dirty(panel); });
  root.querySelector("#add-regex")?.addEventListener("click", () => { readConfig(panel); panel._draft.speech_regex_replacements.push({pattern:"",replacement:""}); panel._setConfigDirty(true); renderRegexRules(panel,panel._draft.speech_regex_replacements.length-1); });
  root.querySelector("#preview-speech")?.addEventListener("click", async () => { const token=(panel._eocSpeechPreviewToken||0)+1;panel._eocSpeechPreviewToken=token;const agentId=panel._agentId,view=panel._viewKey?.(),output=root.querySelector("#speech-output");const current=()=>panel._eocSpeechPreviewToken===token&&panel._agentId===agentId&&panel._viewKey?.()===view&&root.querySelector("#speech-output")===output;try { const response=await panel._call("configuration","speech_preview",{config:readConfig(panel),sample_text:root.querySelector("#speech-sample").value});if(current())output.value=response.speech_text; } catch(err){if(current())panel._toast(`Unable to preview speech: ${err.message||String(err)}`,true);} });
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
