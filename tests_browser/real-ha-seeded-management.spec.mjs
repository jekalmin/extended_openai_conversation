import {expect, test} from "@playwright/test";
import {expectHarnessClean, trackPageErrors} from "./browser-helpers.mjs";
import {expectContractCalls} from "./real-ha-contract.mjs";

const backendUrl = process.env.REAL_HA_BACKEND_URL;
test.skip(!backendUrl, "requires the dedicated genuine Home Assistant backend bridge");
const fixture = (route) => `/tests_browser/real-ha-fixture.html?route=${encodeURIComponent(route)}&backend=${encodeURIComponent(backendUrl)}`;
const owner = "user:management-acceptance-admin";

test("seeded Management owners mutate through shipped frontend and genuine HA", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(fixture("data-memory/memories"));
  await expect(page.locator("extended-openai-management-panel #add-memory")).toBeVisible();

  const before = await page.evaluate(async ({owner}) => {
    const panel = window.browserHarness.panel;
    const legacy = await panel._call("memories", "list", {scope_id: "__anonymous__"});
    const temporary = await panel._call("memories", "temporary_list", {scope_id: owner});
    const conversations = await panel._call("conversations", "list", {scope_id: owner});
    const active = await panel._call("conversations", "active");
    const haCatalog = await panel._call("tools", "ha_catalog");
    const usage = await panel._call("usage", "runs");
    const summary = await panel._call("usage", "summary");
    return {legacy, temporary, conversations, active, haCatalog, usage, summary};
  }, {owner});
  const legacy = before.legacy.memories.find((item) => item.content === "Nightly legacy memory");
  expect(legacy).toBeTruthy();
  expect(before.temporary.memories).toHaveLength(3);
  const session = before.conversations.sessions.find((item) => item.session_id);
  expect(session).toBeTruthy();
  const active = before.active.active.find((item) => item.key);
  expect(active).toBeTruthy();
  const haTool = before.haCatalog.tools.find((item) => item.name === "acceptance_echo");
  expect(haTool).toBeTruthy();
  expect(before.usage.runs.length).toBeGreaterThan(0);

  const changed = await page.evaluate(async ({owner, legacyId, temporaryIds, sessionId, activeKey, haReference}) => {
    const panel = window.browserHarness.panel;
    const moved = await panel._call("memories", "reassign_legacy", {
      scope_id: "__anonymous__", target_scope_id: owner, memory_ids: [legacyId],
    });
    const updated = await panel._call("memories", "temporary_update", {
      scope_id: owner, memory_id: temporaryIds[0], content: "Nightly temporary fact updated",
    });
    const deleted = await panel._call("memories", "temporary_delete", {
      scope_id: owner, memory_id: temporaryIds[1],
    });
    const cleared = await panel._call("memories", "temporary_clear", {
      scope_id: owner, confirm: true,
    });
    const ended = await panel._call("conversations", "end_active", {continuity_key: activeKey});
    const archiveDeleted = await panel._call("conversations", "delete", {
      scope_id: owner, session_id: sessionId,
    });
    const haAdded = await panel._call("tools", "ha_add", {
      tools: [haReference], group_id: "",
    });
    const usageCleared = await panel._call("usage", "clear_details", {confirm: true});
    return {moved, updated, deleted, cleared, ended, archiveDeleted, haAdded, usageCleared};
  }, {
    owner, legacyId: legacy.memory_id,
    temporaryIds: before.temporary.memories.map((item) => item.memory_id),
    sessionId: session.session_id, activeKey: active.key, haReference: haTool.reference,
  });
  expect(changed.moved.reassigned).toBe(1);
  expect(changed.updated.memory.content).toBe("Nightly temporary fact updated");
  expect(changed.deleted.deleted).toBe(1);
  expect(changed.cleared.deleted).toBe(2);
  expect(changed.ended.ended).toBe(1);
  expect(changed.archiveDeleted.deleted_sessions).toBe(1);
  expect(changed.haAdded.functions.some((item) => item.function?.tool_name === "acceptance_echo")).toBe(true);
  expect(changed.usageCleared.deleted_runs).toBeGreaterThan(0);

  await page.goto(fixture("data-memory/memories"));
  await expect(page.locator("extended-openai-management-panel #add-memory")).toBeVisible();
  const after = await page.evaluate(async ({owner}) => {
    const panel = window.browserHarness.panel;
    return {
      legacy: await panel._call("memories", "list", {scope_id: "__anonymous__"}),
      personal: await panel._call("memories", "list", {scope_id: owner}),
      temporary: await panel._call("memories", "temporary_list", {scope_id: owner}),
      conversations: await panel._call("conversations", "list", {scope_id: owner}),
      active: await panel._call("conversations", "active"),
      configuration: await panel._call("configuration", "get"),
      usage: await panel._call("usage", "runs"),
      summary: await panel._call("usage", "summary"),
    };
  }, {owner});
  expect(after.legacy.memories.some((item) => item.memory_id === legacy.memory_id)).toBe(false);
  expect(after.personal.memories.some((item) => item.memory_id === legacy.memory_id)).toBe(true);
  expect(after.temporary.memories).toHaveLength(0);
  expect(after.conversations.sessions.some((item) => item.session_id === session.session_id)).toBe(false);
  expect(after.active.active.some((item) => item.key === active.key)).toBe(false);
  expect(after.configuration.config.functions.some((item) => item.function?.tool_name === "acceptance_echo")).toBe(true);
  expect(after.usage.runs).toHaveLength(0);
  expect(after.summary.lifetime.api_request_count).toBe(before.summary.lifetime.api_request_count);
  await expectContractCalls(page, "seeded_owners");
  await expectHarnessClean(page, pageErrors);
});
