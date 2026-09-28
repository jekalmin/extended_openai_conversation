import {expect, test} from "@playwright/test";
import {acceptConfirmation, expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";
import {openDataCollection} from "./data-collection-helpers.mjs";

async function holdRequest(page, section, action, syntheticResult = null) {
  await page.evaluate(({section, action, syntheticResult}) => {
    const hass = window.browserHarness.hass;
    const original = hass.callWS.bind(hass);
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    window.browserHarness.pendingFeedback = {started:false, release};
    hass.callWS = async (message) => {
      if (message.section === section && message.action === action
          && !window.browserHarness.pendingFeedback.started) {
        window.browserHarness.pendingFeedback.started = true;
        await gate;
        if (syntheticResult !== null) return structuredClone(syntheticResult);
      }
      return original(message);
    };
  }, {section, action, syntheticResult});
}

async function waitPending(page) {
  await expect.poll(() => page.evaluate(() => window.browserHarness.pendingFeedback?.started)).toBe(true);
}

async function releasePending(page) {
  await page.evaluate(() => window.browserHarness.pendingFeedback.release());
}

test("Knowledge delete visibly enters and exits a pending state", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openDataCollection(page, "knowledge", 3);
  await holdRequest(page, "knowledge", "delete");

  const button = panel.locator('[data-source-id="source-0"] .delete-source');
  await button.click();
  await acceptConfirmation(panel);
  await waitPending(page);

  await expect(button).toBeDisabled();
  await expect(button).toHaveText("Deleting…");

  await releasePending(page);
  await expect(panel.locator('[data-source-id="source-0"]')).toHaveCount(0);
  await expectHarnessClean(page, errors);
});

test("persistent Memory delete visibly enters and exits a pending state", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openDataCollection(page, "persistent", 3);
  await holdRequest(page, "memories", "delete");

  const button = panel.locator('[data-memory-id="memory-0"] .delete-memory');
  await button.click();
  await acceptConfirmation(panel);
  await waitPending(page);

  await expect(button).toBeDisabled();
  await expect(button).toHaveText("Deleting…");

  await releasePending(page);
  await expect(panel.locator('[data-memory-id="memory-0"]')).toHaveCount(0);
  await expectHarnessClean(page, errors);
});

test("Temporary Memory delete and clear expose control-level pending feedback", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openDataCollection(page, "temporary", 12);

  await holdRequest(page, "memories", "temporary_delete");
  const deleteButton = panel.locator('[data-memory-id="temporary-0"] .delete-temporary');
  await deleteButton.click();
  await acceptConfirmation(panel);
  await waitPending(page);
  await expect(deleteButton).toBeDisabled();
  await expect(deleteButton).toHaveText("Deleting…");
  await releasePending(page);
  await expect(panel.locator('[data-memory-id="temporary-0"]')).toHaveCount(0);

  await holdRequest(page, "memories", "temporary_clear");
  const clear = panel.locator("#clear-temporary");
  await clear.click();
  await acceptConfirmation(panel);
  await waitPending(page);
  await expect(clear).toBeDisabled();
  await expect(clear).toHaveText("Clearing…");
  await releasePending(page);
  await expect(panel.locator(".memory-list article")).toHaveCount(0);

  await expectHarnessClean(page, errors);
});

test("conversation end and delete actions expose pending feedback", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("data-memory/conversations", "&seed_conversations=1"));
  const panel = page.locator("extended-openai-management-panel");

  await holdRequest(page, "conversations", "end_active");
  const end = panel.locator('.end-active[data-key="active-1"]');
  await end.click();
  await acceptConfirmation(panel);
  await waitPending(page);
  await expect(end).toBeDisabled();
  await expect(end).toHaveText("Ending…");
  await releasePending(page);
  await expect(panel.getByRole("heading", {name:"Kitchen speaker"})).toHaveCount(0);

  await holdRequest(page, "conversations", "delete");
  const remove = panel.locator('.delete-session[data-id="session-1"]');
  await remove.click();
  await acceptConfirmation(panel);
  await waitPending(page);
  await expect(remove).toBeDisabled();
  await expect(remove).toHaveText("Deleting…");
  await releasePending(page);
  await expect(panel.getByRole("heading", {name:"Kitchen project"})).toHaveCount(0);

  await expectHarnessClean(page, errors);
});

test("Usage clear details exposes a pending state and restores normally", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("usage-maintenance/usage"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator("#clear-details")).toBeVisible();

  await holdRequest(page, "usage", "clear_details", {deleted_runs:1, deleted_requests:2});
  const clear = panel.locator("#clear-details");
  await clear.click();
  await acceptConfirmation(panel);
  await waitPending(page);

  await expect(clear).toBeDisabled();
  await expect(clear).toHaveText("Clearing…");

  await releasePending(page);
  await expect(clear).toBeEnabled();
  await expect(clear).not.toHaveText("Clearing…");
  await expectHarnessClean(page, errors);
});

test("legacy Memory reassignment exposes a pending state", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("data-memory/memories"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name:"Memories", exact:true})).toBeVisible();

  await panel.evaluate((host) => host._openReassign("memory-1"));
  const assign = panel.locator("#reassign-save");
  await expect(assign).toBeVisible();
  await holdRequest(page, "memories", "reassign_legacy", {reassigned:1});

  await assign.click();
  await waitPending(page);
  await expect(assign).toBeDisabled();
  await expect(assign).toHaveText("Assigning…");

  await releasePending(page);
  await expect(panel.locator("#reassign-dialog")).toHaveJSProperty("open", false);
  await expectHarnessClean(page, errors);
});

test("failed destructive action restores its control after showing pending feedback", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openDataCollection(page, "knowledge", 3);

  await page.evaluate(() => {
    const original = browserHarness.hass.callWS.bind(browserHarness.hass);
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    window.browserHarness.failedPendingFeedback = {started:false, release};
    browserHarness.hass.callWS = async (message) => {
      if (message.section === "knowledge" && message.action === "delete"
          && !window.browserHarness.failedPendingFeedback.started) {
        window.browserHarness.failedPendingFeedback.started = true;
        await gate;
        throw new Error("Injected delete failure");
      }
      return original(message);
    };
  });

  const button = panel.locator('[data-source-id="source-0"] .delete-source');
  await button.click();
  await acceptConfirmation(panel);
  await expect.poll(() => page.evaluate(() => window.browserHarness.failedPendingFeedback.started)).toBe(true);
  await expect(button).toBeDisabled();
  await expect(button).toHaveText("Deleting…");

  await page.evaluate(() => window.browserHarness.failedPendingFeedback.release());
  await expect(button).toBeEnabled();
  await expect(button).toHaveText("Delete");
  await expect(panel.locator('[data-source-id="source-0"]')).toBeVisible();
  await expectHarnessClean(page, errors);
});
