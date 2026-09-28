import {expect, test} from "@playwright/test";
import {expectHarnessClean, trackPageErrors} from "./browser-helpers.mjs";
import {expectContractCalls} from "./real-ha-contract.mjs";

const backendUrl = process.env.REAL_HA_BACKEND_URL;
test.skip(!backendUrl, "requires the Enhanced genuine Home Assistant backend bridge");
const realFixtureUrl = (route) => `/tests_browser/real-ha-fixture.html?route=${encodeURIComponent(route)}&backend=${encodeURIComponent(backendUrl)}`;

test("shipped frontend duplicates, updates, and imports agents through genuine HA", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(realFixtureUrl("assistant/basics"));
  await expect(page.locator('extended-openai-management-panel [data-config="__title"]')).toBeVisible();
  const result = await page.evaluate(async () => {
    const panel = window.browserHarness.panel;
    const original = await panel._call("configuration", "get");
    const exported = await panel._call("configuration", "export");
    const updated = await panel._call("configuration", "update", {
      config: {current_datetime_enabled: false}, revision: original.revision,
    });
    const duplicate = await panel._call("configuration", "duplicate", {title: "Nightly duplicate"});
    const imported = await panel._call("configuration", "import", {
      document: {...exported.document, title: "Nightly imported"}, mode: "new", confirm: false,
    });
    return {updated, duplicate, imported};
  });
  expect(result.updated.config.current_datetime_enabled).toBe(false);
  expect(result.duplicate.status).toBe("created");
  expect(result.imported.status).toBe("created");
  expect(result.duplicate.subentry_id).not.toBe(result.imported.subentry_id);

  await page.goto(realFixtureUrl("assistant/basics"));
  await expect(page.locator('extended-openai-management-panel [data-config="__title"]')).toBeVisible();
  const persisted = await page.evaluate(async ({duplicateId, importedId}) => {
    const panel = window.browserHarness.panel;
    await panel._loadAgents(duplicateId);
    const duplicate = await panel._call("configuration", "get");
    await panel._loadAgents(importedId);
    const imported = await panel._call("configuration", "get");
    return {duplicate, imported};
  }, {duplicateId: result.duplicate.subentry_id, importedId: result.imported.subentry_id});
  expect(persisted.duplicate.title).toBe("Nightly duplicate");
  expect(persisted.imported.title).toBe("Nightly imported");
  await expectContractCalls(page, "configuration_extended");
  await expectHarnessClean(page, pageErrors);
});

test("shipped frontend persists an empty Request Rule wording list through genuine HA", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(realFixtureUrl("capabilities/request-rules"));
  let panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Request Rules", exact: true})).toBeVisible();
  await panel.locator(".wording-editor summary").click();
  await panel.locator("#wording-add").click();
  await panel.locator(".wording-group").last().locator(".wording-canonical").fill("nightly wording");
  await panel.locator("#save-page").click();
  await page.goto(realFixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".wording-canonical").last()).toHaveValue("nightly wording");
  await panel.locator(".wording-editor summary").click();
  while (await panel.locator(".wording-remove").count()) {
    await panel.locator(".wording-remove").first().click();
  }
  await expect(panel.locator(".wording-group")).toHaveCount(0);
  await expect.poll(() => panel.evaluate((element) => element._rulesSettingsDraft?.wording_groups)).toEqual([]);
  await panel.locator("#save-page").click();
  await page.goto(realFixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".wording-group")).toHaveCount(0);
  await expectContractCalls(page, "request_rules_empty");
  await expectHarnessClean(page, pageErrors);
});

test("shipped frontend starts and ends Guest Mode through genuine HA", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(realFixtureUrl("capabilities/guest-mode"));
  await page.waitForFunction(() => window.browserHarness?.panel?._selectedAgent?.());
  const started = await page.evaluate(() => window.browserHarness.panel._call(
    "guest_mode", "update", {indefinite: true},
  ));
  expect(started.status.state).toBe("active_indefinitely");
  await page.goto(realFixtureUrl("capabilities/guest-mode"));
  await page.waitForFunction(() => window.browserHarness?.panel?._selectedAgent?.());
  const active = await page.evaluate(() => window.browserHarness.panel._call("guest_mode", "get"));
  expect(active.status.state).toBe("active_indefinitely");
  const ended = await page.evaluate(() => window.browserHarness.panel._call("guest_mode", "disable"));
  expect(ended.status.state).toBe("inactive");
  await page.goto(realFixtureUrl("capabilities/guest-mode"));
  await page.waitForFunction(() => window.browserHarness?.panel?._selectedAgent?.());
  const inactive = await page.evaluate(() => window.browserHarness.panel._call("guest_mode", "get"));
  expect(inactive.status.state).toBe("inactive");
  await expectContractCalls(page, "guest_operations");
  await expectHarnessClean(page, pageErrors);
});

