import {expect, test} from "@playwright/test";
import {acceptConfirmation, expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

test("Backup export modes validate selection, preserve settings and reset on reopen", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("usage-maintenance/backup-restore"));
  let panel = page.locator("extended-openai-management-panel");
  for (const mode of ["setup", "custom", "full"]) {
    await panel.locator("#transfer-export-mode").selectOption(mode);
    if (mode === "setup") {
      await panel.locator("#create-backup-transfer").click();
      await expect.poll(() => page.evaluate(() => browserHarness.calls.filter(call => call.action === "setup_export").length)).toBe(1);
    } else if (mode === "custom") {
      await expect(panel.locator("#transfer-custom-options")).toBeVisible();
      for (const input of await panel.locator(".transfer-custom-section").all()) await input.uncheck();
      const downloads = page.waitForEvent("download");
      await panel.locator("#create-backup-transfer").click();
      await expect(panel.getByText("Select at least one section for the custom backup")).toBeVisible();
      await expect(panel.locator("#create-backup-transfer")).toBeEnabled();
      for (const input of await panel.locator(".transfer-custom-section").all()) await input.check();
      await expect(panel.locator(".transfer-custom-section:checked")).toHaveCount(8);
      for (const input of await panel.locator(".transfer-custom-section").all()) await input.uncheck();
      await panel.locator(".transfer-custom-section").first().check();
      await panel.locator("#create-backup-transfer").click();
      await downloads;
    } else {
      const download = page.waitForEvent("download");
      await panel.locator("#create-backup-transfer").click();
      await expect(await download).toBeTruthy();
    }
    await page.goto(fixtureUrl("assistant/basics"));
    await page.goto(fixtureUrl("usage-maintenance/backup-restore"));
    panel = page.locator("extended-openai-management-panel");
    await expect(panel.locator("#transfer-export-mode")).toHaveValue("setup");
    await expect(panel.locator("#transfer-custom-options")).toBeHidden();
  }
  await expectHarnessClean(page, errors);
});

test("Restore rejects corrupt input, supports cancel and requires file reselection after failure", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("usage-maintenance/backup-restore"));
  const panel = page.locator("extended-openai-management-panel");
  await panel.locator("#backup-file-transfer").setInputFiles({name: "corrupt.json", mimeType: "application/json", buffer: Buffer.from("{broken")});
  await expect(panel.locator("#restore-dialog")).toHaveJSProperty("open", false);
  await expect(panel.locator("#toast")).toContainText(/JSON/i);

  await panel.locator("#transfer-export-mode").selectOption("full");
  const download = page.waitForEvent("download");
  await panel.locator("#create-backup-transfer").click();
  const path = await (await download).path();
  await panel.locator("#restore-backup-transfer").click();
  await panel.locator("#backup-file-transfer").setInputFiles(path);
  await expect(panel.locator("#restore-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#restore-transfer-cancel").click();
  await expect(panel.locator("#restore-dialog")).toHaveJSProperty("open", false);
  await expect.poll(() => page.evaluate(() => browserHarness.calls.filter(call => call.action === "import_cancel").length)).toBeGreaterThan(0);

  await panel.locator("#restore-backup-transfer").click();
  await panel.locator("#backup-file-transfer").setInputFiles(path);
  await expect(panel.locator("#restore-dialog")).toHaveJSProperty("open", true);
  await page.evaluate(() => {
    const original = browserHarness.hass.callWS;
    window.restoreFailureCalls = 0;
    browserHarness.hass.callWS = message => {
      if (message.type === "extended_openai_conversation_responses/management/backup_transfer" && message.action === "import_restore" && window.restoreFailureCalls++ === 0) return Promise.reject(new Error("injected restore failure"));
      return original(message);
    };
  });
  await panel.locator("#restore-transfer-apply").click();
  await acceptConfirmation(panel);
  await expect(panel.getByText(/Re-select the transfer file to retry/)).toBeVisible();
  expect(await panel.evaluate(host => host._backupTransferSession)).toBeNull();
  await panel.locator("#restore-transfer-apply").click();
  expect(await page.evaluate(() => window.restoreFailureCalls)).toBe(1);
  await expectHarnessClean(page, errors);
});

test("Restore selection preview follows the latest section choice and successful restore reloads", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("usage-maintenance/backup-restore"));
  const panel = page.locator("extended-openai-management-panel");
  await panel.locator("#transfer-export-mode").selectOption("full");
  const download = page.waitForEvent("download");
  await panel.locator("#create-backup-transfer").click();
  const path = await (await download).path();
  await panel.locator("#backup-file-transfer").setInputFiles(path);
  await expect(panel.locator("#restore-dialog")).toHaveJSProperty("open", true);
  await panel.locator('.transfer-restore-section[value="configuration"]').uncheck();
  await panel.locator('.transfer-restore-section[value="request_rules"]').check();
  await expect(panel.locator("#restore-transfer-apply")).toBeEnabled();
  await panel.locator("#restore-transfer-apply").click();
  await acceptConfirmation(panel);
  await expect(panel.locator("#restore-dialog")).toHaveJSProperty("open", false);
  expect(await page.evaluate(() => browserHarness.calls.filter(call => call.action === "import_restore").at(-1)?.data?.sections)).toContain("request_rules");
  await expectHarnessClean(page, errors);
});
