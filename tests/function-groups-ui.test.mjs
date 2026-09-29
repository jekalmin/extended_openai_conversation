import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import {renderTools} from "../custom_components/extended_openai_conversation_responses/frontend/agent-config-tools.js";
import {configurationChoiceLabel, matchesFunctionSearch, saveBar} from "../custom_components/extended_openai_conversation_responses/frontend/agent-config-editor-base.js";
import {canReplaceToolYamlWithoutConfirmation, categorizeFunctionTools, deleteFunctionGroup, functionGroupIdFromName, functionToolCountLabel, isFunctionGroupEnabled, isFunctionToolEnabled, synchronizePersistedFunctions} from "../custom_components/extended_openai_conversation_responses/frontend/agent-config-tools-base.js";

const tool = (name) => ({spec: {name, description: name}, function: {type: "native"}});
const config = {
  functions: [tool("general"), tool("remind"), tool("calendar")],
  function_groups: [
    {id: "reminders", name: "Reminders", description: "Manage reminders", loading_mode: "on_demand", functions: ["remind"], enabled: true},
    {id: "calendar", name: "Calendar", description: "Manage calendars", loading_mode: "always", functions: ["calendar"], enabled: true},
  ],
};

assert.equal(functionGroupIdFromName("  Conditional Notifications  "), "conditional-notifications");
assert.equal(functionGroupIdFromName("123 Reminders"), "reminders");
assert.equal(configurationChoiceLabel("conversation_continuity", {value: "ha_default", label: "Ha Default"}), "Use Home Assistant sessions");
assert.equal(configurationChoiceLabel("voice_scope_policy", {value: "shared", label: "Shared"}), "Use shared household data");
assert.equal(configurationChoiceLabel("service_tier", {value: "flex", label: "Flex"}), "Flex");
assert.equal(matchesFunctionSearch("Reminder", "Reminders Manage reminders"), true);
assert.equal(matchesFunctionSearch("cafe", "Quick Reads / Café 📚"), true);
assert.equal(matchesFunctionSearch("café", "Quick Reads / Cafe 📚"), true);
assert.equal(matchesFunctionSearch("Reminder", "remind family native"), true);
assert.equal(matchesFunctionSearch("manage calendar", "Calendar Manage calendars"), true);
assert.equal(matchesFunctionSearch("weather", "Reminders Manage reminders"), false);
assert.equal(isFunctionToolEnabled(tool("enabled")), true);
assert.equal(isFunctionToolEnabled({...tool("disabled"), enabled: false}), false);
assert.equal(isFunctionGroupEnabled({id: "legacy"}), true);
assert.equal(isFunctionGroupEnabled({id: "disabled", enabled: false}), false);
assert.equal(functionToolCountLabel([tool("one"), {...tool("two"), enabled: false}]), "1 enabled · 1 disabled");
assert.equal(canReplaceToolYamlWithoutConfirmation("", "starter"), true);
assert.equal(canReplaceToolYamlWithoutConfirmation("starter", "starter"), true);
assert.equal(canReplaceToolYamlWithoutConfirmation("edited", "starter"), false);
const managementStyles = await readFile(
  new URL(
    "../custom_components/extended_openai_conversation_responses/frontend/management.css",
    import.meta.url,
  ),
  "utf8",
);
assert.ok(
  managementStyles.includes("[hidden]{display:none!important}"),
  "dialog search results marked hidden must actually be removed from layout",
);

const categorized = categorizeFunctionTools(config);
assert.deepEqual(categorized.alwaysAvailable.map((item) => item.spec.name), ["general"]);
assert.deepEqual(categorized.groups.map((group) => group.tools.map((item) => item.spec.name)), [["remind"], ["calendar"]]);

const duplicateMembership = {
  functions: [tool("first"), tool("shared"), tool("last")],
  function_groups: [
    {id: "alpha", name: "Alpha", functions: ["shared", "first"]},
    {id: "beta", name: "Beta", functions: ["shared", "last"]},
  ],
};
const duplicateCategorized = categorizeFunctionTools(duplicateMembership);
assert.deepEqual(
  duplicateCategorized.groups.map((group) => group.tools.map((item) => item.spec.name)),
  [["first", "shared"], ["shared", "last"]],
  "group membership must preserve configured tool order and duplicate membership",
);
assert.deepEqual(
  duplicateCategorized.alwaysAvailable,
  [],
  "a tool assigned to any group must not also be categorized as ungrouped",
);

const deleted = deleteFunctionGroup(config, "reminders");
assert.equal(deleted.function_groups.length, 1);
assert.deepEqual(deleted.functions, config.functions, "deleting a group must retain its functions");
assert.deepEqual(
  categorizeFunctionTools(deleted).alwaysAvailable.map((item) => item.spec.name),
  ["general", "remind"],
);
assert.equal(config.function_groups.length, 2, "group helpers must not mutate the saved draft");

assert.equal(saveBar({_configDirty:false}), "");
assert.match(saveBar({_configDirty:true}), /<strong id="dirty-state" class="dirty-state">Unsaved changes<\/strong>/);
assert.doesNotMatch(saveBar({_configDirty:true}), /All changes saved/);

const panel = {
  _configData:{title:"Agent",config:{prompt:"A",functions:[tool("old")],function_groups:[]}},
  _result:null,
  _draft:{prompt:"B",functions:[tool("old")],function_groups:[]},
  _draftTitle:"Agent",
  _configDirty:false,
  _syncConfigDirty(){this._configDirty=this._draftTitle!==this._configData.title||JSON.stringify(this._draft)!==JSON.stringify(this._configData.config);},
};
synchronizePersistedFunctions(panel,{functions:[tool("new")],function_groups:[]});
assert.equal(panel._configData.config.prompt,"A", "direct Function saves must preserve the saved prompt");
assert.equal(panel._draft.prompt,"B", "direct Function saves must preserve the local prompt draft");
assert.equal(panel._configData.config.functions[0].spec.name,"new");
assert.equal(panel._draft.functions[0].spec.name,"new");
assert.equal(panel._configDirty,true,"an unrelated prompt draft must remain dirty");

panel._draft.prompt="A";
synchronizePersistedFunctions(panel,{functions:[tool("newer")],function_groups:[]});
assert.equal(panel._configDirty,false,"a direct Function save must not create false dirty state");

const many = {
  functions: Array.from({length: 100}, (_, index) => tool(`tool_${index}`)),
  function_groups: [],
};
assert.equal(categorizeFunctionTools(many).alwaysAvailable.length, 100);

const editorSource = (
  await Promise.all([
    "../custom_components/extended_openai_conversation_responses/frontend/agent-config-editor.js",
    "../custom_components/extended_openai_conversation_responses/frontend/agent-config-editor-base.js",
    "../custom_components/extended_openai_conversation_responses/frontend/agent-config-tools.js",
    "../custom_components/extended_openai_conversation_responses/frontend/agent-config-tools-base.js",
  ].map((path) => readFile(new URL(path, import.meta.url), "utf8")))
).join("\n");
assert.match(editorSource, /insertAdjacentHTML\("beforebegin", saveBar\(panel\)\)/, "the bar should appear as soon as a clean configuration becomes dirty");
const inputSource = await readFile(new URL("../custom_components/extended_openai_conversation_responses/frontend/configuration-inputs.js", import.meta.url), "utf8");
assert.match(inputSource, /#revert-config[\s\S]*?_setConfigDirty\(false\);[\s\S]*?panel\._eocMainMarkup = null;\s*panel\._render\(\)/, "reverting should clear dirty state and restore the form including the save bar");
assert.match(editorSource, /bindSingleRequestSave\(panel\)/, "configuration saves must use the canonical shell handler");
assert.match(
  editorSource,
  /const functionSearchTokenCache = new WeakMap\(\)/,
  "Function Tool search should cache static tokenization by persistent DOM node",
);
assert.match(
  editorSource,
  /const queryTokens = hasQuery \? searchTokens\(query\) : \[\]/,
  "Function Tool search should tokenize the changing query once per input event",
);
assert.match(
  editorSource,
  /#local-intent-search[\s\S]*?const queryTokens = searchTokens\(event\.target\.value\)[\s\S]*?cachedFunctionSearchTokens/,
  "Local Intent search should tokenize its query once and cache static choice tokens",
);
assert.match(
  editorSource,
  /#group-function-search[\s\S]*?queryTokens=searchTokens\(event\.target\.value\)[\s\S]*?cachedFunctionSearchTokens/,
  "Function Group assignment search should tokenize its query once and cache static choice tokens",
);
assert.match(
  editorSource,
  /cachedFunctionSearchTokens\(card, card\.dataset\.groupSearch\)/,
  "group search text should reuse cached static tokens",
);
assert.match(
  editorSource,
  /cachedFunctionSearchTokens\(tool, tool\.dataset\.toolSearch\)/,
  "tool search text should reuse cached static tokens",
);
assert.match(
  editorSource,
  /const categories = indexFunctionToolGroups\(config\);[\s\S]*?const membership = categories\.membership;/,
  "reconciliation should reuse the categorisation membership index",
);
assert.doesNotMatch(
  editorSource,
  /new Map\(groups\.flatMap\(group => \(group\.functions \|\| \[\]\)\.map\(name => \[name, group\]\)\)\)/,
  "reconciliation must not rebuild group membership after categorisation",
);
assert.doesNotMatch(editorSource, /Save tools and groups|Keep in draft|unsaved tool draft/i);
assert.match(editorSource, /id="tool-save">Save<\/button>/);
assert.match(editorSource, /id="group-save">Save<\/button>/);
assert.match(editorSource, /class=\"group-enabled\"/, "groups should expose an independent enabled switch");
const disabledMarkup = renderTools({_e:String, _empty:String, _draft:{...config, function_groups:config.function_groups.map((group) => ({...group, enabled:false}))}});
assert.match(disabledMarkup, /class="secondary edit-group"[^>]*disabled[^>]*Enable this Function Group before editing it/, "disabled groups must render a disabled edit button immediately");
assert.match(editorSource, /member Function Tool settings were kept/, "group state changes should explain that member states are retained");
for (const action of ["save","set_enabled","delete","save_group","delete_group"]) {
  assert.match(editorSource,new RegExp(`panel\\._call\\(\"tools\",\\s*\"${action}\"`));
}
