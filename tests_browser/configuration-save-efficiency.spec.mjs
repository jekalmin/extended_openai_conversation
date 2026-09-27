import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

test("Backup paints before transfer code or configuration data loads", async ({page}) => {
  const errors = trackPageErrors(page);
  const assets = [];
  page.on("request", request => assets.push(new URL(request.url()).pathname.split("/").pop()));
  await page.goto(fixtureUrl("usage-maintenance/backup-restore"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator("#create-backup-transfer")).toBeVisible();
  expect(assets.some(name => name.startsWith("backup-transfer-ui") || name.startsWith("agent-config-editor"))).toBe(false);
  expect(await page.evaluate(() => browserHarness.calls.some(call => call.section === "configuration" && call.action === "get"))).toBe(false);
  await panel.locator("#transfer-export-mode").selectOption("full");
  const download = page.waitForEvent("download");
  await panel.locator("#create-backup-transfer").click();
  await download;
  expect(assets.some(name => name.startsWith("backup-transfer-ui"))).toBe(true);
  await expectHarnessClean(page, errors);
});

test("Backup keeps an existing unsaved configuration draft while exporting saved data", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("assistant/basics"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator('[data-config="__title"]')).toBeVisible();
  await panel.locator('[data-config="__title"]').fill("Draft agent name");
  await panel.evaluate(host => host._navigate("usage-maintenance", "backup-restore"));
  await expect(panel.locator("#create-backup-transfer")).toBeEnabled();
  await expect(panel.getByText("Exports use saved configuration.", {exact: false})).toBeVisible();
  await expect(panel.locator("#confirm-dialog")).toHaveJSProperty("open", false);
  await panel.evaluate(host => host._navigate("assistant", "basics"));
  await expect(panel.locator('[data-config="__title"]')).toHaveValue("Draft agent name");
  await expectHarnessClean(page, errors);
});

test("Retention reads a narrow projection and saves only its changed field", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("usage-maintenance/retention"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator('[data-config="usage_request_retention_days"]')).toBeVisible();
  expect(await page.evaluate(() => browserHarness.calls.filter(call => call.section === "configuration").map(call => call.action))).toEqual(["retention_get"]);
  await panel.locator('[data-config="usage_request_retention_days"]').selectOption("7");
  await panel.locator("#save-config").click();
  const save = await page.evaluate(() => browserHarness.calls.find(call => call.section === "configuration" && call.action === "save"));
  expect(save).toMatchObject({config:{usage_request_retention_days:7}, revision:"fixture-7"});
  expect(Object.keys(save.config)).toEqual(["usage_request_retention_days"]);
  expect(save).not.toHaveProperty("title");
  await expect(panel.locator('[data-config="usage_request_retention_days"]')).toHaveValue("7");
  await expectHarnessClean(page, errors);
});

test("Leaving the Retention projection loads the full configuration editor", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("usage-maintenance/retention"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator('[data-config="usage_run_retention_days"]')).toBeVisible();
  await panel.evaluate(host => host._navigate("assistant", "basics"));
  await expect(panel.locator('[data-config="chat_model"]')).toHaveValue("gpt-5-mini");
  expect(await page.evaluate(() => browserHarness.calls.filter(call => call.section === "configuration").map(call => call.action))).toEqual(["retention_get", "get"]);
  await expectHarnessClean(page, errors);
});

test("Configuration save sends dirty fields and consumes the normalized response", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("assistant/basics"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator('[data-config="max_tokens"]')).toBeVisible();
  await page.evaluate(() => {
    const original = browserHarness.hass.callWS.bind(browserHarness.hass);
    browserHarness.hass.callWS = async message => {
      const result = await original(message);
      if (message.section === "configuration" && message.action === "save") return {...result, config:{...result.config, max_tokens:777}};
      return result;
    };
  });
  await panel.locator('[data-config="max_tokens"]').fill("760");
  await panel.locator("#save-config").click();
  await expect(panel.locator('[data-config="max_tokens"]')).toHaveValue("777");
  const save = await page.evaluate(() => browserHarness.calls.find(call => call.section === "configuration" && call.action === "save"));
  expect(save).toMatchObject({config:{max_tokens:760}, revision:"fixture-7"});
  expect(Object.keys(save.config)).toEqual(["max_tokens"]);
  expect(save).not.toHaveProperty("title");
  const reads = await page.evaluate(() => browserHarness.calls.filter(
    call => call.section === "configuration" && call.action === "get",
  ).length);
  await panel.locator('.top-nav button[data-page="usage-maintenance"]').click();
  await expect(panel.locator("#usage-window")).toBeVisible();
  await panel.locator('.top-nav button[data-page="assistant"]').click();
  await expect(panel.locator('[data-config="max_tokens"]')).toHaveValue("777");
  expect(await page.evaluate(() => browserHarness.calls.filter(
    call => call.section === "configuration" && call.action === "get",
  ).length)).toBe(reads);
  await expectHarnessClean(page, errors);
});
