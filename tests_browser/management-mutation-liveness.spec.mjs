import {expect, test} from "@playwright/test";
import {acceptConfirmation, expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";
import {openDataCollection} from "./data-collection-helpers.mjs";

async function holdFirstMutation(page, section, action) {
  await page.evaluate(({section, action}) => {
    const hass = window.browserHarness.hass;
    const original = hass.callWS.bind(hass);
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    window.browserHarness.livenessMutation = {
      section,
      action,
      calls: 0,
      started: false,
      release,
    };
    hass.callWS = async (message) => {
      if (message.section === section && message.action === action) {
        window.browserHarness.livenessMutation.calls += 1;
        if (window.browserHarness.livenessMutation.calls === 1) {
          window.browserHarness.livenessMutation.started = true;
          await gate;
        }
      }
      return original(message);
    };
  }, {section, action});
}

async function waitForHeldMutation(page) {
  await expect.poll(() => page.evaluate(() => window.browserHarness.livenessMutation?.started)).toBe(true);
}

async function releaseHeldMutation(page) {
  await page.evaluate(() => window.browserHarness.livenessMutation.release());
}

async function expectOverviewLive(panel) {
  await panel.evaluate((host) => host._navigate("overview"));
  await expect(panel.locator(".dashboard-grid")).toBeVisible();
  await expect(panel.locator("main")).not.toHaveAttribute("aria-busy", "true");
}

test("configuration remains live for an immediate second save and route read", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("assistant/basics"));
  const panel = page.locator("extended-openai-management-panel");
  const maxTokens = panel.locator('[data-config="max_tokens"]');
  await expect(maxTokens).toBeVisible();

  await holdFirstMutation(page, "configuration", "save");
  await maxTokens.fill("760");
  await panel.locator("#save-config").click();
  await waitForHeldMutation(page);

  await releaseHeldMutation(page);
  await expect.poll(() => panel.evaluate((host) => host._configDirty)).toBe(false);

  await maxTokens.fill("761");
  await expect(panel.locator("#save-config")).toBeVisible();
  await panel.locator("#save-config").click();
  await expect.poll(() => page.evaluate(() => window.browserHarness.livenessMutation.calls)).toBe(2);
  await expect.poll(() => panel.evaluate((host) => host._configDirty)).toBe(false);

  await expectOverviewLive(panel);
  await expectHarnessClean(page, errors);
});

test("a failed configuration save does not poison the next mutation", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("assistant/basics"));
  const panel = page.locator("extended-openai-management-panel");
  const maxTokens = panel.locator('[data-config="max_tokens"]');
  await expect(maxTokens).toBeVisible();

  await page.evaluate(() => {
    const hass = window.browserHarness.hass;
    const original = hass.callWS.bind(hass);
    window.browserHarness.failedLivenessSaveCalls = 0;
    hass.callWS = async (message) => {
      if (message.section === "configuration" && message.action === "save") {
        window.browserHarness.failedLivenessSaveCalls += 1;
        if (window.browserHarness.failedLivenessSaveCalls === 1) throw new Error("Injected save failure");
      }
      return original(message);
    };
  });

  await maxTokens.fill("762");
  await panel.locator("#save-config").click();
  await expect.poll(() => page.evaluate(() => window.browserHarness.failedLivenessSaveCalls)).toBe(1);
  await expect(panel.locator("#save-config")).toBeEnabled();

  await maxTokens.fill("763");
  await panel.locator("#save-config").click();
  await expect.poll(() => page.evaluate(() => window.browserHarness.failedLivenessSaveCalls)).toBe(2);
  await expect.poll(() => panel.evaluate((host) => host._configDirty)).toBe(false);

  await expectOverviewLive(panel);
  await expectHarnessClean(page, errors);
});

test("Knowledge update completion leaves the collection immediately mutable and navigable", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openDataCollection(page, "knowledge", 3);
  await holdFirstMutation(page, "knowledge", "update");

  await panel.locator('[data-source-id="source-0"] .source-edit-button').click();
  await expect(panel.locator("#knowledge-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#knowledge-description").fill("First liveness update");
  await panel.locator("#knowledge-save").click();
  await waitForHeldMutation(page);

  await releaseHeldMutation(page);
  await expect(panel.locator("#knowledge-dialog")).toHaveJSProperty("open", false);

  await panel.locator('[data-source-id="source-1"] .source-edit-button').click();
  await panel.locator("#knowledge-description").fill("Second liveness update");
  await panel.locator("#knowledge-save").click();
  await expect.poll(() => page.evaluate(() => window.browserHarness.livenessMutation.calls)).toBe(2);
  await expect(panel.locator("#knowledge-dialog")).toHaveJSProperty("open", false);

  await expectOverviewLive(panel);
  await expectHarnessClean(page, errors);
});

test("persistent Memory update completion leaves the next memory immediately mutable", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openDataCollection(page, "persistent", 3);
  await holdFirstMutation(page, "memories", "update");

  await panel.locator('[data-memory-id="memory-0"] .memory-edit-button').click();
  await panel.locator("#memory-content").fill("Memory zero liveness update");
  await panel.locator("#memory-save").click();
  await waitForHeldMutation(page);

  await releaseHeldMutation(page);
  await expect(panel.locator("#memory-dialog")).toHaveJSProperty("open", false);

  await panel.locator('[data-memory-id="memory-1"] .memory-edit-button').click();
  await panel.locator("#memory-content").fill("Memory one liveness update");
  await panel.locator("#memory-save").click();
  await expect.poll(() => page.evaluate(() => window.browserHarness.livenessMutation.calls)).toBe(2);
  await expect(panel.locator("#memory-dialog")).toHaveJSProperty("open", false);

  await expectOverviewLive(panel);
  await expectHarnessClean(page, errors);
});

test("Temporary Memory delete completion does not block the next destructive mutation", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openDataCollection(page, "temporary", 12);
  await holdFirstMutation(page, "memories", "temporary_delete");

  await panel.locator('[data-memory-id="temporary-0"] .delete-temporary').click();
  await acceptConfirmation(panel);
  await waitForHeldMutation(page);

  await releaseHeldMutation(page);
  await expect(panel.locator('[data-memory-id="temporary-0"]')).toHaveCount(0);

  await panel.locator('[data-memory-id="temporary-1"] .delete-temporary').click();
  await acceptConfirmation(panel);
  await expect.poll(() => page.evaluate(() => window.browserHarness.livenessMutation.calls)).toBe(2);
  await expect(panel.locator('[data-memory-id="temporary-1"]')).toHaveCount(0);

  await expectOverviewLive(panel);
  await expectHarnessClean(page, errors);
});

test("Request Rule deletion releases the surface for an immediate update and navigation", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".request-rule-card")).toHaveCount(1);

  await panel.locator(".request-rule-card .rule-duplicate").click();
  await expect(panel.locator(".request-rule-card")).toHaveCount(2);

  await holdFirstMutation(page, "request_rules", "delete");
  await panel.locator(".request-rule-card").nth(1).locator(".rule-delete").click();
  await acceptConfirmation(panel);
  await waitForHeldMutation(page);

  await releaseHeldMutation(page);
  await expect(panel.locator(".request-rule-card")).toHaveCount(1);

  const updatesBefore = await page.evaluate(() => window.browserHarness.calls.filter(
    (call) => call.section === "request_rules" && call.action === "update",
  ).length);
  const enabled = panel.locator(".request-rule-card .rule-enabled");
  if (await enabled.isChecked()) await enabled.uncheck();
  else await enabled.check();
  await expect.poll(() => page.evaluate((before) => window.browserHarness.calls.filter(
    (call) => call.section === "request_rules" && call.action === "update",
  ).length, updatesBefore)).toBe(updatesBefore + 1);

  await expectOverviewLive(panel);
  await expectHarnessClean(page, errors);
});

test("Guest Mode can save again immediately after a completed policy mutation", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/guest-mode"));
  const panel = page.locator("extended-openai-management-panel");
  const toggle = panel.locator("#guest-controls-enabled");
  await toggle.evaluate((input) => { input.closest("details").open = true; });

  await holdFirstMutation(page, "guest_mode", "save_policy");
  await toggle.check();
  await panel.locator("#save-page").click();
  await waitForHeldMutation(page);

  await releaseHeldMutation(page);
  await expect(panel.locator("#save-page")).toHaveCount(0);

  await toggle.uncheck();
  await expect(panel.locator("#save-page")).toBeVisible();
  await panel.locator("#save-page").click();
  await expect.poll(() => page.evaluate(() => window.browserHarness.livenessMutation.calls)).toBe(2);
  await expect(panel.locator("#save-page")).toHaveCount(0);

  await expectOverviewLive(panel);
  await expectHarnessClean(page, errors);
});


test("a completed Knowledge mutation does not block an immediate Memory mutation", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openDataCollection(page, "knowledge", 3);

  await panel.locator('[data-source-id="source-0"] .source-edit-button').click();
  await panel.locator("#knowledge-description").fill("Cross-feature liveness");
  await panel.locator("#knowledge-save").click();
  await expect(panel.locator("#knowledge-dialog")).toHaveJSProperty("open", false);

  await panel.evaluate((host) => {
    host._memoryKind = "persistent";
    return host._navigate("data-memory", "memories");
  });
  await expect(panel.locator('[data-memory-id="memory-0"] .memory-edit-button')).toBeVisible();
  const before = await page.evaluate(() => dataCollectionBackend.calls.filter(
    (call) => call.section === "memories" && call.action === "update",
  ).length);
  await panel.locator('[data-memory-id="memory-0"] .memory-edit-button').click();
  await panel.locator("#memory-content").fill("Cross-feature memory mutation");
  await panel.locator("#memory-save").click();
  await expect.poll(() => page.evaluate((count) => dataCollectionBackend.calls.filter(
    (call) => call.section === "memories" && call.action === "update",
  ).length, before)).toBe(before + 1);
  await expect(panel.locator("#memory-dialog")).toHaveJSProperty("open", false);

  await expectOverviewLive(panel);
  await expectHarnessClean(page, errors);
});

test("a blocked Knowledge background read does not block an unrelated Memory mutation", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openDataCollection(page, "knowledge", 3);

  await page.evaluate(() => {
    const original = browserHarness.hass.callWS;
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    window.browserHarness.releaseHeldKnowledgeRead = release;
    window.browserHarness.heldKnowledgeReadStarted = false;
    browserHarness.hass.callWS = async (message) => {
      if (message.section === "knowledge" && message.action === "list"
          && !window.browserHarness.heldKnowledgeReadStarted) {
        window.browserHarness.heldKnowledgeReadStarted = true;
        await gate;
      }
      return original(message);
    };
  });

  const heldRefresh = panel.evaluate((host) => {
    const key = host._sectionCacheKey();
    host._sectionCache.delete(key);
    host._eocSectionCacheTimes.delete(key);
    return host._loadSection(true);
  });
  await expect.poll(() => page.evaluate(() => window.browserHarness.heldKnowledgeReadStarted)).toBe(true);

  await panel.evaluate((host) => {
    host._memoryKind = "persistent";
    return host._navigate("data-memory", "memories");
  });
  await expect(panel.locator('[data-memory-id="memory-1"] .memory-edit-button')).toBeVisible();
  const before = await page.evaluate(() => dataCollectionBackend.calls.filter(
    (call) => call.section === "memories" && call.action === "update",
  ).length);
  await panel.locator('[data-memory-id="memory-1"] .memory-edit-button').click();
  await panel.locator("#memory-content").fill("Foreground mutation while Knowledge read is held");
  await panel.locator("#memory-save").click();
  await expect.poll(() => page.evaluate((count) => dataCollectionBackend.calls.filter(
    (call) => call.section === "memories" && call.action === "update",
  ).length, before)).toBe(before + 1);

  await page.evaluate(() => window.browserHarness.releaseHeldKnowledgeRead());
  await heldRefresh;
  await expect(panel.getByRole("heading", {name: "Memories", exact: true})).toBeVisible();
  await expectHarnessClean(page, errors);
});

test("Knowledge delete can be followed immediately by create without reloading the route", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openDataCollection(page, "knowledge", 3);

  await panel.locator('[data-source-id="source-0"] .delete-source').click();
  await acceptConfirmation(panel);
  await expect(panel.locator('[data-source-id="source-0"]')).toHaveCount(0);

  const createsBefore = await page.evaluate(() => dataCollectionBackend.calls.filter(
    (call) => call.section === "knowledge" && call.action === "create",
  ).length);
  await panel.locator("#add-source").click();
  await panel.locator("#knowledge-title").fill("Created immediately after delete");
  await panel.locator("#knowledge-description").fill("Liveness create");
  await panel.locator("#knowledge-content").fill("Created without a route reload.");
  await panel.locator("#knowledge-save").click();
  await expect.poll(() => page.evaluate((count) => dataCollectionBackend.calls.filter(
    (call) => call.section === "knowledge" && call.action === "create",
  ).length, createsBefore)).toBe(createsBefore + 1);
  await expect(panel.getByText("Created immediately after delete", {exact: true})).toBeVisible();

  await expectOverviewLive(panel);
  await expectHarnessClean(page, errors);
});

test("a failed Request Rule update leaves a different rule action usable", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = page.locator("extended-openai-management-panel");
  await panel.locator(".request-rule-card .rule-duplicate").click();
  await expect(panel.locator(".request-rule-card")).toHaveCount(2);

  await page.evaluate(() => {
    const original = browserHarness.hass.callWS.bind(browserHarness.hass);
    window.browserHarness.failedRuleUpdateCalls = 0;
    browserHarness.hass.callWS = async (message) => {
      if (message.section === "request_rules" && message.action === "update") {
        window.browserHarness.failedRuleUpdateCalls += 1;
        if (window.browserHarness.failedRuleUpdateCalls === 1) throw new Error("Injected Request Rule update failure");
      }
      return original(message);
    };
  });

  const firstToggle = panel.locator(".request-rule-card .rule-enabled").first();
  const initialEnabled = await firstToggle.isChecked();
  await firstToggle.evaluate((input, checked) => {
    input.checked = !checked;
    input.dispatchEvent(new Event("change", {bubbles: true}));
  }, initialEnabled);
  await expect.poll(() => page.evaluate(() => window.browserHarness.failedRuleUpdateCalls)).toBe(1);
  await expect(firstToggle).toBeEnabled();
  await expect(firstToggle).toBeChecked({checked: initialEnabled});

  const deletesBefore = await page.evaluate(() => browserHarness.calls.filter(
    (call) => call.section === "request_rules" && call.action === "delete",
  ).length);
  await panel.locator(".request-rule-card").nth(1).locator(".rule-delete").click();
  await acceptConfirmation(panel);
  await expect.poll(() => page.evaluate((count) => browserHarness.calls.filter(
    (call) => call.section === "request_rules" && call.action === "delete",
  ).length, deletesBefore)).toBe(deletesBefore + 1);
  await expect(panel.locator(".request-rule-card")).toHaveCount(1);

  await expectOverviewLive(panel);
  await expectHarnessClean(page, errors);
});

test("route-owned settings can save across different Management sections without a dead period", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("usage-maintenance/retention"));
  const panel = page.locator("extended-openai-management-panel");

  const retention = panel.locator('[data-config="usage_request_retention_days"]');
  await expect(retention).toBeVisible();
  await retention.selectOption("7");
  await panel.locator("#save-config").click();
  await expect.poll(() => panel.evaluate((host) => host._configDirty)).toBe(false);

  await panel.evaluate((host) => host._navigate("data-memory", "memory-settings"));
  const memoryLimit = panel.locator('[data-memory-config="memory_auto_retrieve_limit"]');
  await expect(memoryLimit).toBeVisible();
  await memoryLimit.fill("7");
  await panel.locator("#save-config").click();
  await expect.poll(() => panel.evaluate((host) => host._configDirty)).toBe(false);

  await panel.evaluate((host) => host._navigate("capabilities", "quiet-hours"));
  await expect(panel.locator("#qh-start")).toBeVisible();
  await panel.locator("#qh-start").fill("23:00");
  await panel.locator("#save-page").click();
  await expect(panel.locator("#save-page")).toHaveCount(0);

  await expectOverviewLive(panel);
  await expectHarnessClean(page, errors);
});

test("conversation mutations do not leave Management unavailable afterwards", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("data-memory/conversations", "&seed_conversations=1"));
  const panel = page.locator("extended-openai-management-panel");

  await panel.locator('.end-active[data-key="active-1"]').click();
  await acceptConfirmation(panel);
  await expect(panel.getByRole("heading", {name: "Kitchen speaker"})).toHaveCount(0);

  await panel.locator('.delete-session[data-id="session-1"]').click();
  await acceptConfirmation(panel);
  await expect(panel.getByRole("heading", {name: "Kitchen project"})).toHaveCount(0);

  await expectOverviewLive(panel);
  await expectHarnessClean(page, errors);
});

test("rapid mutation to Overview and back does not leave stale loading state", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openDataCollection(page, "knowledge", 3);

  await panel.locator('[data-source-id="source-2"] .source-edit-button').click();
  await panel.locator("#knowledge-description").fill("A-B-A route liveness");
  await panel.locator("#knowledge-save").click();
  await expect(panel.locator("#knowledge-dialog")).toHaveJSProperty("open", false);

  await panel.evaluate((host) => host._navigate("overview"));
  await expect(panel.locator(".dashboard-grid")).toBeVisible();
  await panel.evaluate((host) => host._navigate("data-memory", "knowledge"));
  await expect(panel.getByRole("heading", {name: "Sources", exact: true})).toBeVisible();
  await expect(panel.locator("main")).not.toHaveAttribute("aria-busy", "true");
  await expect(panel.locator('[data-source-id="source-2"]')).toContainText("A-B-A route liveness");

  await expectHarnessClean(page, errors);
});

test("a compact cross-feature mutation chain remains live without intermediate reloads", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openDataCollection(page, "knowledge", 3);

  await panel.locator('[data-source-id="source-0"] .source-edit-button').click();
  await panel.locator("#knowledge-description").fill("Chain knowledge");
  await panel.locator("#knowledge-save").click();

  await panel.evaluate((host) => {
    host._memoryKind = "persistent";
    return host._navigate("data-memory", "memories");
  });
  await panel.locator('[data-memory-id="memory-0"] .memory-edit-button').click();
  await panel.locator("#memory-content").fill("Chain memory");
  await panel.locator("#memory-save").click();

  await panel.evaluate((host) => host._navigate("capabilities", "request-rules"));
  const ruleToggle = panel.locator(".request-rule-card .rule-enabled").first();
  await expect(ruleToggle).toBeVisible();
  const updatesBefore = await page.evaluate(() => browserHarness.calls.filter(
    (call) => call.section === "request_rules" && call.action === "update",
  ).length);
  if (await ruleToggle.isChecked()) await ruleToggle.uncheck();
  else await ruleToggle.check();
  await expect.poll(() => page.evaluate((count) => browserHarness.calls.filter(
    (call) => call.section === "request_rules" && call.action === "update",
  ).length, updatesBefore)).toBe(updatesBefore + 1);

  await expectOverviewLive(panel);
  await expectHarnessClean(page, errors);
});


test("a completed mutation does not leave the agent picker or next agent blocked", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("assistant/basics", "&agents=2"));
  const panel = page.locator("extended-openai-management-panel");
  const picker = panel.locator("#agent");
  await expect(picker).toHaveValue("agent-1");

  const maxTokens = panel.locator('[data-config="max_tokens"]');
  await maxTokens.fill("764");
  await panel.locator("#save-config").click();
  await expect.poll(() => panel.evaluate((host) => host._configDirty)).toBe(false);
  await expect(picker).toBeEnabled();

  await picker.selectOption("scale-agent-1");
  await expect(picker).toHaveValue("scale-agent-1");
  await expect(panel.locator('[data-config="max_tokens"]')).toBeVisible();

  const savesBefore = await page.evaluate(() => browserHarness.calls.filter(
    (call) => call.section === "configuration" && call.action === "save"
      && call.subentry_id === "scale-agent-1",
  ).length);
  await panel.locator('[data-config="max_tokens"]').fill("765");
  await panel.locator("#save-config").click();
  await expect.poll(() => page.evaluate((count) => browserHarness.calls.filter(
    (call) => call.section === "configuration" && call.action === "save"
      && call.subentry_id === "scale-agent-1",
  ).length, savesBefore)).toBe(savesBefore + 1);
  await expect.poll(() => panel.evaluate((host) => host._configDirty)).toBe(false);

  await expectOverviewLive(panel);
  await expectHarnessClean(page, errors);
});
