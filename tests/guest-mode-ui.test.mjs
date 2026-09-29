import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {freshGuestPolicyDraft, GUEST_EXCLUSION_KEYS} from "../custom_components/extended_openai_conversation_responses/frontend/guest-mode-ui.js";

const panel = await readFile(new URL("../custom_components/extended_openai_conversation_responses/frontend/management-guest-feature.js", import.meta.url), "utf8") + await readFile(
  new URL("../custom_components/extended_openai_conversation_responses/frontend/management-panel.js", import.meta.url),
  "utf8",
);
const managementStyles = await readFile(
  new URL("../custom_components/extended_openai_conversation_responses/frontend/management.css", import.meta.url),
  "utf8",
);
const editor = await readFile(
  new URL("../custom_components/extended_openai_conversation_responses/frontend/agent-config-editor.js", import.meta.url),
  "utf8",
);
const guide = await readFile(
  new URL("../custom_components/extended_openai_conversation_responses/frontend/guide-content.js", import.meta.url),
  "utf8",
);

assert.match(panel, /capabilities\/guest-mode/);
assert.match(panel, /guest_mode", "update"/);
assert.match(panel, /guest_mode", "disable"/);
assert.match(panel, /Voice and model safety/);
assert.match(panel, /Capability safety/);
assert.match(panel, /side effects cannot be safely limited/);
assert.match(panel, /This is intentional/);
assert.match(panel, /Allow the assistant to activate Guest Mode/);
assert.match(panel, /guest_excluded_entities/);
assert.match(panel, /guest_control_excluded_entities/);
assert.match(panel, /guest_knowledge_source_ids/);
assert.match(panel, /guest_allowed_function_names/);
assert.match(panel, /guest_shared_memory_policy/);
assert.match(panel, /ha-selector/);
assert.match(panel, /savePageChanges\(this\)/);
assert.match(panel, /"Labels"/);
assert.match(panel, /"Areas"/);
assert.match(panel, /"Domains"/);
assert.match(panel, /"Individual entities"/);
assert.ok(panel.indexOf('"Labels"') < panel.indexOf('"Areas"'));
assert.ok(panel.indexOf('"Areas"') < panel.indexOf('"Domains"'));
assert.ok(panel.indexOf('"Domains"') < panel.indexOf('"Individual entities"'));
assert.match(panel, /Make some visible entities read-only/);
assert.match(panel, /Previous Guest Mode settings found/);
assert.match(panel, /Review converted settings/);
assert.match(panel, /Start fresh/);
assert.match(panel, /old policy remains enforced until you save/i);
assert.match(panel, /Labels are usually the easiest way to manage Guest access/);
assert.match(panel, /Normally, anything a guest can see can also be controlled/);
assert.match(panel, /formatUsageNumber\(\(config\[key\] \|\| \[\]\)\.length\)} excluded/);
assert.match(panel, /class="[^"]*\bguest-manager\b[^"]*"/);
assert.match(panel, /class="manage-label">Manage/);
assert.match(panel, /config\.guest_separate_control_restrictions \? `<div class="guest-managers">/);
assert.match(panel, /All current Knowledge sources are available to guests. New sources added later will also be included/);
assert.match(panel, /All eligible enabled functions are available to guests. New eligible functions added later will also be included/);
assert.match(panel, /filter\(\(item\) => !item\.unsafe_in_guest_mode\)/);
assert.match(panel, /The assistant can never shorten or disable Guest Mode/);
assert.doesNotMatch(panel, /Trusted actions|trusted Home Assistant actions|trusted controls/i);
assert.doesNotMatch(panel, /futureKnowledge|futureFunctions/);
assert.match(panel, /Guest Mode activation/);
assert.match(panel, /The assistant can enable or extend Guest Mode, but cannot shorten or disable it\. Administrators and Home Assistant automations can change or end Guest Mode\./);
assert.match(guide, /The assistant can enable or extend Guest Mode, but cannot shorten or disable it\. Administrators and Home Assistant automations can change or end Guest Mode\./);
assert.doesNotMatch(guide, /trusted Home Assistant controls/i);
assert.match(panel, /<summary>Assistant activation permission<\/summary><p class="help">This permission is separate from the Guest Mode activation state\.<\/p><label class="toggle"><span>Allow the assistant to activate Guest Mode/);
assert.match(managementStyles, /top:50%/);
assert.match(managementStyles, /translateY\(-50%\)/);
assert.doesNotMatch(editor, /Guest Mode policy/);
assert.doesNotMatch(editor, /group-guest-allowed/);

globalThis.window = {location:{pathname:"/extended-openai/capabilities/guest-mode"}, addEventListener() {}, removeEventListener() {}};
globalThis.history = {pushState() {}};
globalThis.localStorage = {getItem() { return null; }, setItem() {}};
globalThis.HTMLElement = class { attachShadow() { this.shadowRoot = {hasChildNodes: () => false}; } };
globalThis.customElements = {define() {}, get() { return null; }, whenDefined() { return Promise.resolve(); }};
const {ExtendedOpenAIManagementPanel} = await import("../custom_components/extended_openai_conversation_responses/frontend/management-panel.js");

const guestPanel = new ExtendedOpenAIManagementPanel();
const {routeAssetPromise} = await import("../custom_components/extended_openai_conversation_responses/frontend/management-route.js");
await routeAssetPromise("capabilities/guest-mode", guestPanel);
guestPanel._data = {is_admin:true};
guestPanel._page = "capabilities";
guestPanel._subsection = "guest-mode";

guestPanel._result = {config:{}, policy:{}, status:{state:"inactive", currently_active:false, scheduled:false, active_from:null, active_until:null}};
const inactive = guestPanel._content({});
assert.match(inactive, /Guest Mode activation/);
assert.match(inactive, /Inactive · No interval configured/);
assert.match(inactive, /id="guest-start"/);
assert.match(inactive, /id="guest-end"/);
assert.match(inactive, /id="guest-now">Activate now/);
assert.match(inactive, /class="secondary" id="guest-update">Update interval/);
assert.doesNotMatch(inactive, /id="guest-disable"/);
assert.doesNotMatch(managementStyles, /\.guest-intro\+\.content-card\{display:none\}/);

guestPanel._result = {config:{}, policy:{}, status:{state:"scheduled", currently_active:false, scheduled:true, active_from:"2026-08-26T10:00:00Z", active_until:null}};
const scheduled = guestPanel._content({});
assert.match(scheduled, /Scheduled · Starts/);
assert.match(scheduled, /id="guest-update">Update interval/);
assert.match(scheduled, /id="guest-now">Activate now/);
assert.match(scheduled, /id="guest-disable">Cancel schedule/);

const legacyDraft = {guest_excluded_entities:["light.private"], guest_mode_enabled:false, unrelated:"kept"};
const freshDraft = freshGuestPolicyDraft(legacyDraft);
assert.equal(legacyDraft.guest_excluded_entities.length, 1, "start fresh must not mutate the enforced legacy snapshot");
assert.ok(GUEST_EXCLUSION_KEYS.every((key) => freshDraft[key].length === 0));
assert.equal(freshDraft.guest_mode_enabled, true);
assert.equal(freshDraft.guest_knowledge_policy, "off");
assert.equal(freshDraft.guest_function_policy, "off");
assert.equal(freshDraft.guest_shared_memory_policy, "off");
assert.equal(freshDraft.unrelated, "kept");
