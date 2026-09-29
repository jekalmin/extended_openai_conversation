import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";

import {routeAssetPromise} from "../custom_components/extended_openai_conversation_responses/frontend/management-route.js";
await Promise.all(["assistant/basics", "assistant/voice", "capabilities/home-assistant", "data-memory/conversations", "usage-maintenance/retention"].map((view) => routeAssetPromise(view)));
import {NAVIGATION, routeFromPath, shouldShowGlobalSettingsSearch} from "../custom_components/extended_openai_conversation_responses/frontend/frontend-navigation.js";
import {searchSettings} from "../custom_components/extended_openai_conversation_responses/frontend/management-settings-index.js";
import {GUIDE_TOPICS, MEMORY_COMPARISON} from "../custom_components/extended_openai_conversation_responses/frontend/guide-content.js";
import {renderGuide} from "../custom_components/extended_openai_conversation_responses/frontend/guide-page.js";
import {knowledgeAvailabilityMarkup} from "../custom_components/extended_openai_conversation_responses/frontend/management-knowledge-feature.js";
import {MODEL_RESET_FIELDS} from "../custom_components/extended_openai_conversation_responses/frontend/management-memory-settings.js";
import {settingCurrentState} from "../custom_components/extended_openai_conversation_responses/frontend/management-navigation-search.js";
import {renderMemorySettings} from "../custom_components/extended_openai_conversation_responses/frontend/memory-settings-ui.js";
import {renderOverview} from "../custom_components/extended_openai_conversation_responses/frontend/overview-page.js";
import {renderConfiguration, renderConfigurationActions} from "../custom_components/extended_openai_conversation_responses/frontend/agent-config-editor.js";

const escape = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

assert.deepEqual(NAVIGATION.map((item) => item.id), [
  "overview", "guide", "assistant", "capabilities", "data-memory", "usage-maintenance",
]);
assert.deepEqual(NAVIGATION.find((item) => item.id === "assistant").sections.map((item) => item.id), [
  "basics", "model-responses", "conversation", "prompt-context", "voice", "speech",
]);
assert.deepEqual(NAVIGATION.find((item) => item.id === "capabilities").sections.map((item) => item.id), [
  "home-assistant", "web-skills", "request-rules", "functions", "quiet-hours", "guest-mode",
]);
assert.deepEqual(routeFromPath("/extended-openai/assistant/advanced"), {page:"capabilities", section:"web-skills", legacy:true});
assert.deepEqual(routeFromPath("/extended-openai/capabilities/web-skills"), {page:"capabilities", section:"web-skills", legacy:false});
const dataMemory = NAVIGATION.find((item) => item.id === "data-memory");
assert.equal(dataMemory.path, "/extended-openai/data-memory/memory-settings");
assert.deepEqual(dataMemory.sections.map((item) => item.id), [
  "memory-settings", "memories", "knowledge", "conversations",
]);
assert.ok(NAVIGATION.every((item) => item.label?.trim() && item.path));
assert.ok(NAVIGATION.flatMap((item) => item.sections).every((item) => item.label?.trim() && item.description?.trim()));

const legacy = {
  configuration: ["assistant", "basics"],
  tools: ["capabilities", "functions"],
  guest: ["capabilities", "guest-mode"],
  memories: ["data-memory", "memories"],
  knowledge: ["data-memory", "knowledge"],
  conversations: ["data-memory", "conversations"],
  usage: ["usage-maintenance", "usage"],
  diagnostics: ["usage-maintenance", "diagnostics"],
};
for (const [oldRoute, expected] of Object.entries(legacy)) {
  const route = routeFromPath(`/extended-openai/${oldRoute}`);
  assert.deepEqual([route.page, route.section], expected);
  assert.equal(route.legacy, true);
}
assert.deepEqual(routeFromPath("/extended-openai/assistant/conversation"), {page:"assistant", section:"conversation", legacy:false});
assert.deepEqual(routeFromPath("/extended-openai/data-memory"), {page:"data-memory", section:"memory-settings", legacy:false});

for (const query of ["archive", "memory", "timeout", "model", "Guest Mode", "voice", "backup", "embeddings", "context", "tools functions", "local handling", "web search", "skills", "knowledge", "tool calls", "markdown", "processing tier"]) {
  assert.ok(searchSettings(query).length, `global settings search should find ${query}`);
}
assert.equal(searchSettings("conversation timeout")[0].configKey, "conversation_timeout_minutes");
assert.equal(searchSettings("response creativity")[0].configKey, "temperature");
assert.equal(searchSettings("web search detail")[0].configKey, "web_search_context");
assert.equal(searchSettings("backup")[0].section, "backup-restore");
for (const [query, key] of [["long-term memory","memory_mode"],["short-term memory","temporary_memory"],["embedding model","memory_embedding_model"],["shared household memory","shared_memory_mode"]]) {
  const result = searchSettings(query)[0];
  assert.equal(result?.page, "data-memory", `${query} should resolve to Data & Memory`);
  assert.equal(result?.section, "memory-settings", `${query} should resolve to Memory settings`);
  assert.equal(result?.configKey, key);
  assert.equal(result?.target, `config-${key}`);
}
for (const [query, section, key] of [["local handling","home-assistant","local_intents_enabled"],["web search","web-skills","web_search"],["skills","web-skills","skills"]]) {
  const result = searchSettings(query)[0];
  assert.equal(result?.page, "capabilities", `${query} should resolve to Capabilities`);
  assert.equal(result?.section, section);
  assert.equal(result?.configKey, key);
}
assert.equal(searchSettings("knowledge library access")[0].section, "knowledge");
for (const [page, section] of [["overview", null], ["guide", null], ["assistant", "basics"], ["capabilities", "functions"], ["data-memory", "memories"], ["data-memory", "conversations"], ["usage-maintenance", "usage"], ["usage-maintenance", "retention"]]) {
  assert.equal(shouldShowGlobalSettingsSearch(page, section), true, `settings search should be available on ${page}/${section || ""}`);
}

const configData = {
  title: "Kitchen",
  model_capabilities: {supports_temperature:false, supports_reasoning_effort:true},
  options: {
    conversation_continuity: [{value:"device",label:"Remember by voice device"}],
    memory_mode: [{value:"manual",label:"Manual"}],
  },
  config: {
    web_search: true,
    conversation_timeout_minutes: 30,
    conversation_continuity: "device",
    voice_device_mappings: {satellite_one:"user:1", satellite_two:"shared"},
    speech_regex_replacements: [{pattern:"A",replacement:"B"}],
    skills: ["weather", "calendar"],
    memory_mode: "manual",
    current_datetime_template: "",
    prompt: "You are helpful.",
    temperature: 0.4,
  },
};
const currentPanel = {_agentId:"agent-1", _draft:null, _draftAgentId:null, _configData:null, _settingsSearchConfig:configData, _settingsSearchConfigAgentId:"agent-1"};
assert.deepEqual(settingCurrentState(searchSettings("web search")[0], currentPanel), {label:"Current",value:"On"});
assert.deepEqual(settingCurrentState(searchSettings("conversation timeout")[0], currentPanel), {label:"Current",value:"30 min"});
assert.deepEqual(settingCurrentState(searchSettings("voice device assignments")[0], currentPanel), {label:"Current",value:"2 assignments"});
assert.deepEqual(settingCurrentState(searchSettings("custom speech replacements")[0], currentPanel), {label:"Current",value:"1 rule"});
assert.deepEqual(settingCurrentState(searchSettings("skills")[0], currentPanel), {label:"Current",value:"2 skills"});
assert.deepEqual(settingCurrentState(searchSettings("long-term memory")[0], currentPanel), {label:"Current",value:"Only when I ask (Manual)"});
assert.deepEqual(settingCurrentState(searchSettings("current date/time format")[0], currentPanel), {label:"Current",value:"Default"});
assert.deepEqual(settingCurrentState(searchSettings("system prompt")[0], currentPanel), {label:"Current",value:"16 characters"});
assert.deepEqual(settingCurrentState(searchSettings("response creativity")[0], currentPanel), {label:"Current",value:"Not supported by current model"});
const draftPanel = {...currentPanel, _draft:{...configData.config, web_search:false}, _draftAgentId:"agent-1", _draftTitle:"Kitchen draft", _configData:configData};
assert.deepEqual(settingCurrentState(searchSettings("web search")[0], draftPanel), {label:"Current draft",value:"Off"});
const otherAgentPanel = {...currentPanel, _agentId:"agent-2"};
assert.equal(settingCurrentState(searchSettings("web search")[0], otherAgentPanel), null, "cached values must not bleed between agents");
const guestPanel = {_selectedAgent:()=>({guest_mode:{state:"scheduled"}})};
assert.deepEqual(settingCurrentState(searchSettings("guest mode")[0], guestPanel), {label:"Current",value:"Scheduled"});

const memoryPanel = {
  _configDirty: false,
  _e: escape,
  _result: {
    options: {
      memory_mode: [{value:"off",label:"Off"},{value:"manual",label:"Manual"},{value:"automatic",label:"Automatic"}],
      temporary_memory: [{value:"off",label:"Off"},{value:"balanced",label:"Balanced"},{value:"eager",label:"Eager"}],
      memory_retrieval_mode: [{value:"lexical",label:"Lexical"},{value:"hybrid",label:"Hybrid"}],
      shared_memory_mode: [{value:"disabled",label:"Disabled"},{value:"explicit",label:"Explicit"},{value:"automatic",label:"Automatic"}],
    },
    config: {
      memory_mode: "manual",
      temporary_memory: "balanced",
      memory_auto_retrieve_limit: 3,
      memory_retrieval_mode: "lexical",
      memory_embedding_model: "text-embedding-3-small",
      shared_memory_mode: "explicit",
    },
  },
};
const lexicalMemoryHtml = renderMemorySettings(memoryPanel);
for (const key of ["memory_mode", "temporary_memory", "memory_auto_retrieve_limit", "memory_retrieval_mode", "memory_embedding_model", "shared_memory_mode"]) {
  assert.match(lexicalMemoryHtml, new RegExp(`data-memory-config="${key}"`));
}
assert.match(lexicalMemoryHtml, /data-memory-config="memory_embedding_model"[^>]*disabled/);
assert.match(lexicalMemoryHtml, /data-page="data-memory"[^>]*data-subsection="memories"/);
memoryPanel._result.config.memory_retrieval_mode = "hybrid";
const hybridMemoryHtml = renderMemorySettings(memoryPanel);
assert.doesNotMatch(hybridMemoryHtml, /data-memory-config="memory_embedding_model"[^>]*disabled/);
assert.deepEqual(MODEL_RESET_FIELDS, ["temperature", "top_p", "reasoning_effort", "service_tier", "shorten_tool_call_id"]);
assert.ok(MODEL_RESET_FIELDS.every((key) => !key.includes("memory")));

const knowledgePanel = {
  _e: escape,
  _data:{is_admin:true},
  _selectedAgent:()=>({feature_status:{knowledge:{state:"enabled"}}}),
};
const knowledgeAvailability = knowledgeAvailabilityMarkup(knowledgePanel);
assert.match(knowledgeAvailability, /knowledge-enabled-toggle/);
assert.match(knowledgeAvailability, /checked/);
knowledgePanel._data.is_admin = false;
const readOnlyKnowledge = knowledgeAvailabilityMarkup(knowledgePanel);
assert.match(readOnlyKnowledge, /Assistant access/);
assert.match(readOnlyKnowledge, /id="knowledge-status"/);
assert.doesNotMatch(readOnlyKnowledge, /knowledge-enabled-toggle/);

assert.ok(GUIDE_TOPICS.length >= 12);
assert.ok(GUIDE_TOPICS.some((topic) => topic.id === "guest-mode"));
assert.ok(GUIDE_TOPICS.some((topic) => topic.id === "request-rules"));
assert.ok(GUIDE_TOPICS.every((topic) => topic.action?.page));
assert.equal(MEMORY_COMPARISON.length, 5);
assert.ok(MEMORY_COMPARISON.every((row) => row[0]?.trim()));

const guidePanel = {_guideQuery:"guest", _guideTopic:"guest-mode", _e:escape, _empty:(message) => message};
const guideHtml = renderGuide(guidePanel);
assert.match(guideHtml, /data-page="capabilities"[^>]*data-subsection="guest-mode"/);
assert.doesNotMatch(guideHtml, /data-topic="model-provider"/);

const overviewAgent = {title:"Kitchen",provider:"OpenAI",model:"gpt-test",function_count:3,function_group_count:2,memory_mode:"hybrid",memory_count:4,knowledge_source_count:2,archive_enabled:true,guest_mode:{state:"active",has_home_assistant_exclusions:false}};
const overviewPanel = {
  _result:{usage:{today:{total_tokens:12},month:{total_tokens:34}},conversations:{archive_retention_days:30}},
  _e:escape,
  _titleCase:(value) => value,
};
const overviewHtml = renderOverview(overviewPanel, overviewAgent);
assert.match(overviewHtml, /data-page="capabilities"/);
for (const page of ["assistant", "data-memory", "usage-maintenance"]) assert.match(overviewHtml, new RegExp(`data-page="${page}"`));

const partialOverviewPanel = {
  _result:{usage:{today:{total_tokens:1234},month:{total_tokens:5678}},conversations:{archive_retention_days:30},load_errors:[{key:"knowledge",label:"Knowledge",message:"offline"}]},
  _e:escape,
  _titleCase:(value) => value,
};
const partialOverviewHtml = renderOverview(partialOverviewPanel, {...overviewAgent, guest_mode:{state:"inactive",has_home_assistant_exclusions:true}});
assert.match(partialOverviewHtml, /Knowledge could not be loaded/);
assert.match(partialOverviewHtml, /1,234/);
assert.match(partialOverviewHtml, /5,678/);

// A hidden configuration subsection must keep its unsaved values.
const {readConfigurationDraft} = await import("../custom_components/extended_openai_conversation_responses/frontend/configuration-controls.js");
const hiddenConfig = {voice_device_mappings:{speaker:"user"}, speech_regex_replacements:[{pattern:"a",replacement:"b"}]};
assert.deepEqual(readConfigurationDraft({_draft:hiddenConfig, shadowRoot:{querySelector:() => null, querySelectorAll:() => []}}), hiddenConfig);

const actionOwner = {
  _e:escape,
  _titleCase:String,
  _empty:escape,
  _draft:{},
  _result:{config:{},options:{},defaults:{}},
  _configDirty:false,
  _configSections:["general"],
  _viewKey:() => "assistant/basics",
};
const cleanActions = renderConfigurationActions(actionOwner, ["general"]);
for (const id of ["duplicate-agent", "import-agent", "export-agent"]) {
  assert.match(cleanActions, new RegExp(`id="${id}"`));
}
actionOwner._configDirty = true;
const dirtyActions = renderConfigurationActions(actionOwner, ["general"]);
for (const id of ["duplicate-agent", "export-agent"]) {
  assert.match(dirtyActions, new RegExp(`id="${id}"[^>]*disabled`));
}
assert.match(dirtyActions, /id="import-agent"/);
assert.doesNotMatch(renderConfiguration(actionOwner), /id="duplicate-agent"/);

const configurationOwner = {_e:escape,_titleCase:String,_result:{config:{},options:{}},_configSections:["model"],_viewKey:() => "assistant/model-responses"};
assert.doesNotMatch(renderConfiguration(configurationOwner), /data-field="memory_auto_retrieve_limit"/);
assert.match(renderConfiguration(configurationOwner), /id="reset-model-parameters"/);
configurationOwner._viewKey = () => "assistant/conversation";
configurationOwner._configSections = ["conversation"];
assert.doesNotMatch(renderConfiguration(configurationOwner), /id="config-local"/);

// These warnings describe the Home Assistant exposure boundary and remain
// intentional copy contracts until a browser accessibility test covers them.
const panelSource = await readFile(new URL("../custom_components/extended_openai_conversation_responses/frontend/management-panel.js", import.meta.url), "utf8");
assert.match(panelSource, /Home Assistant's Assist exposure settings decide which entities may be used by the assistant/);
assert.match(panelSource, /Turning this off does not necessarily prevent the assistant from using exposed entities/);
