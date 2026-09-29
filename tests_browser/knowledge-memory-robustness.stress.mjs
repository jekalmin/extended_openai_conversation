import {expect, test} from "@playwright/test";
import {acceptConfirmation, expectHarnessClean, trackPageErrors} from "./browser-helpers.mjs";
import {openDataCollection} from "./data-collection-helpers.mjs";

test("Knowledge CRUD failures, retry, filtering and availability reconcile against the collection", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openDataCollection(page, "knowledge", 120);
  await panel.locator("#list-search").fill("Source 11");
  await expect(panel.locator("[data-source-id]:visible")).toHaveCount(11);
  await panel.locator("#add-source").click();
  await panel.locator("#knowledge-title").fill("Nightly source");
  await panel.locator("#knowledge-description").fill("Retryable reference");
  await panel.locator("#knowledge-content").fill("Authoritative persisted content");
  await page.evaluate(() => {
    const backend = dataCollectionBackend, original = backend.call.bind(backend);
    let fail = true;
    window.knowledgeCreateAttempts = 0;
    backend.call = async message => {
      if (message.section === "knowledge" && message.action === "create") {
        window.knowledgeCreateAttempts++;
        if (fail) { fail = false; throw new Error("temporary create failure"); }
      }
      return original(message);
    };
  });
  await panel.locator("#knowledge-save").click();
  await expect(panel.locator("#knowledge-error")).toContainText("temporary create failure");
  await expect(panel.locator("#knowledge-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#knowledge-save").click();
  await expect(panel.locator("#knowledge-dialog")).not.toHaveJSProperty("open", true);
  await expect(panel.locator("#list-search")).toHaveValue("Source 11");
  await panel.locator("#list-search").fill("Nightly source");
  await expect(panel.locator("[data-source-id]:visible")).toHaveCount(1);
  await panel.locator("[data-source-id]:visible .source-edit-button").click();
  await panel.locator("#knowledge-title").fill("Nightly source edited");
  await panel.locator("#knowledge-save").click();
  await panel.locator("#list-search").fill("Nightly source edited");
  await expect(panel.locator("[data-source-id]:visible")).toHaveCount(1);
  await panel.locator("[data-source-id]:visible .delete-source").click();
  await acceptConfirmation(panel);
  await expect(panel.locator("[data-source-id]:visible")).toHaveCount(0);
  const state = await page.evaluate(() => ({
    sources: dataCollectionBackend.state.sources,
    creates: window.knowledgeCreateAttempts,
    updates: dataCollectionBackend.calls.filter(call => call.action === "update").length,
    deletes: dataCollectionBackend.calls.filter(call => call.action === "delete").length,
  }));
  expect(state.creates).toBe(2);
  expect(state.updates).toBe(1);
  expect(state.deletes).toBe(1);
  expect(state.sources.some(source => source.title.startsWith("Nightly source"))).toBe(false);
  await expectHarnessClean(page, errors);
});

test("Knowledge availability and source mutation can finish out of order without losing either result", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openDataCollection(page, "knowledge", 8);
  await page.evaluate(() => {
    const backend = dataCollectionBackend, original = backend.call.bind(backend);
    let releaseToggle;
    window.knowledgeToggleStarted = new Promise(resolve => { window.knowledgeToggleStartedResolve = resolve; });
    window.releaseKnowledgeToggle = () => releaseToggle?.();
    backend.call = async message => {
      if (message.section === "knowledge" && message.action === "set_enabled") {
        await new Promise(resolve => { releaseToggle = resolve; window.knowledgeToggleStartedResolve(); });
      }
      return original(message);
    };
  });
  await panel.locator("#knowledge-enabled-toggle").setChecked(false);
  await page.evaluate(() => window.knowledgeToggleStarted);
  await panel.locator("#add-source").click();
  await panel.locator("#knowledge-title").fill("Concurrent source");
  await panel.locator("#knowledge-content").fill("Created while availability is pending");
  await panel.locator("#knowledge-save").click();
  await expect(panel.locator("#knowledge-dialog")).not.toHaveJSProperty("open", true);
  await page.evaluate(() => window.releaseKnowledgeToggle());
  await expect.poll(() => page.evaluate(() => dataCollectionBackend.state.sources.some(source => source.title === "Concurrent source"))).toBe(true);
  await expect(panel.locator("[data-source-id]").filter({hasText: "Concurrent source"})).toHaveCount(1);
  await expect(panel.locator("#knowledge-enabled-toggle")).not.toBeChecked();
  await expectHarnessClean(page, errors);
});

test("Memory date metadata survives backend failure, retry and fresh section load", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openDataCollection(page, "persistent", 30);
  await panel.locator('[data-memory-id="memory-1"] .memory-edit-button').click();
  await panel.locator("#memory-content").fill("Temporal fact");
  await panel.locator("#memory-valid-from").fill("2026-09-17T12:30");
  await page.evaluate(() => {
    const backend = dataCollectionBackend, original = backend.call.bind(backend);
    let fail = true;
    backend.call = async message => {
      if (message.section === "memories" && message.action === "update" && fail) { fail = false; throw new Error("invalid valid_from date-time"); }
      return original(message);
    };
  });
  await panel.locator("#memory-save").click();
  await expect(panel.locator("#memory-error")).toBeVisible();
  await expect(panel.locator("#memory-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#memory-subject").fill("Renewal");
  await panel.locator("#memory-key").fill("contract.renewal");
  await panel.locator("#memory-save").click();
  await expect(panel.locator(".memory-list")).toContainText("Temporal fact");
  await expect.poll(() => page.evaluate(() => dataCollectionBackend.state.memories.some(memory => memory.content === "Temporal fact" && memory.valid_from === "2026-09-17T12:30"))).toBe(true);
  await page.evaluate(() => browserHarness.panel._loadSection(true));
  await panel.locator("[data-memory-id]").filter({hasText: "Temporal fact"}).locator(".memory-edit-button").click();
  await expect(panel.locator("#memory-valid-from")).toHaveValue("2026-09-17T12:30");
  await expect(panel.locator("#memory-subject")).toHaveValue("Renewal");
  await expect(panel.locator("#memory-key")).toHaveValue("contract.renewal");
  await expectHarnessClean(page, errors);
});
