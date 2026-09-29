import assert from "node:assert/strict";
import test from "node:test";
import {UnsavedState, draftScope, same, saveBarMarkup} from "../custom_components/extended_openai_conversation_responses/frontend/unsaved-state.js";
import {initializePageDraft, currentPageScope} from "../custom_components/extended_openai_conversation_responses/frontend/management-page-drafts.js";
import {confirmStateSafeNavigation} from "../custom_components/extended_openai_conversation_responses/frontend/management-state-safety.js";

test("drafts compare values, discard, save once, and preserve edits made during save", async () => {
  let value = {a: 1, b: 2}, release, calls = 0;
  const scope = draftScope({baseline: value, read: () => value, write: (next) => { value = next; },
    save: async (submitted) => { calls++; await new Promise((resolve) => { release = resolve; }); return submitted; }});
  value = {b: 2, a: 1};
  assert.equal(scope.dirty(), false, "key order does not make a draft dirty");
  value.a = 3; assert.equal(scope.dirty(), true);
  scope.discard(); assert.equal(value.a, 1); assert.equal(scope.dirty(), false);
  value.a = 4; const pending = scope.save();
  assert.equal(scope.pending, true); assert.equal(await scope.save(), false); assert.equal(calls, 1);
  value.a = 5; release(); await pending;
  assert.equal(scope.baseline.a, 4); assert.equal(value.a, 5); assert.equal(scope.dirty(), true);
  assert.match(saveBarMarkup({pending: true}), /disabled>Saving…/);
  assert.match(saveBarMarkup(), /Discard changes/);
});

for (const view of ["capabilities/guest-mode", "capabilities/quiet-hours"]) {
  test(`${view}: authoritative save and failure preserve draft`, async () => {
    let fail = true;
    const panel = {_agentId: "a", _viewKey: () => view, _result: {config: {enabled: false}, revision: "v1"},
      _call: async (_section, action, payload) => {
        if (view === "capabilities/guest-mode" && action === "details") return {policy: {readable_entity_count: 1, controllable_entity_count: 1, configured_tool_count: 0}};
        assert.equal(payload.revision, "v1"); if (fail) throw Error("offline"); return {config: {enabled: true}, revision: "v2"};
      }};
    initializePageDraft(panel); const scope = currentPageScope(panel);
    scope.read().enabled = true;
    await assert.rejects(scope.save(), /offline/);
    assert.equal(scope.dirty(), true); assert.equal(scope.pending, false); assert.equal(scope.read().enabled, true);
    fail = false; await scope.save(); assert.equal(scope.dirty(), false); assert.equal(scope.revision, "v2");
    scope.read().enabled = false; scope.discard(); assert.equal(scope.read().enabled, true);
  });
}

test("Request Rules settings save atomically and preserve the whole draft on failure", async () => {
  let fail = true;
  const calls = [];
  const panel = {
    _agentId: "a",
    _viewKey: () => "capabilities/request-rules",
    _result: {defaults: {fuzzy: false}, wording_groups: [], revision: "v1"},
    _call: async (section, action, payload) => {
      calls.push([section, action, payload.revision]);
      assert.equal(section, "request_rules");
      assert.equal(action, "settings");
      assert.deepEqual(payload.defaults, {fuzzy: true});
      assert.deepEqual(payload.wording_groups, [{canonical: "on", alternatives: ["enable"]}]);
      if (fail) throw Error("offline");
      return {defaults: payload.defaults, wording_groups: payload.wording_groups, revision: "v2"};
    },
  };
  initializePageDraft(panel);
  const scope = currentPageScope(panel);
  scope.read().defaults.fuzzy = true;
  scope.read().wording_groups.push({canonical: "on", alternatives: ["enable"]});

  await assert.rejects(scope.save(), /offline/);
  assert.deepEqual(calls, [["request_rules", "settings", "v1"]]);
  assert.equal(scope.dirty(), true);
  assert.equal(scope.baseline.defaults.fuzzy, false);
  assert.deepEqual(scope.baseline.wording_groups, []);
  assert.equal(scope.revision, "v1");

  fail = false;
  await scope.save();
  assert.equal(scope.dirty(), false);
  assert.equal(scope.revision, "v2");
  assert.deepEqual(calls.at(-1), ["request_rules", "settings", "v1"]);
});

test("central navigation shares config context, cancels safely, and discards without persistence", async () => {
  const state = new UnsavedState(); let value = 1, accepted = false, prompts = 0;
  const scope = draftScope({baseline: 0, read: () => value, write: (next) => { value = next; },
    owns: (destination) => destination?.startsWith("assistant/"), save: () => { throw Error("must not persist"); }});
  state.register("configuration", scope);
  const panel = {_unsavedState: state, _confirm: async () => { prompts++; return accepted; }};
  assert.equal(await confirmStateSafeNavigation(panel, "assistant/models"), true); assert.equal(prompts, 0);
  assert.equal(await confirmStateSafeNavigation(panel, "overview"), false); assert.equal(value, 1);
  assert.equal(await confirmStateSafeNavigation(panel, null), false, "agent switch is guarded");
  accepted = true; assert.equal(await confirmStateSafeNavigation(panel, "overview"), true); assert.equal(value, 0);
  assert.equal(state.hasChanges(), false);
  assert.equal(same({x: [1, 2]}, {x: [2, 1]}), false);
});
