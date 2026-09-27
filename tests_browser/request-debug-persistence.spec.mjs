import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

test("Request debugging capture policy persists across a fresh panel", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("usage-maintenance/request-debug"));
  let debug = page.locator("extended-openai-debug-panel");
  await expect(debug.locator("#enabled")).toBeEnabled();
  await expect(debug.locator("#enabled")).not.toBeChecked();
  await debug.locator("#enabled").check();
  await expect(debug.locator("#enabled")).toBeChecked();
  await debug.locator("#limit").selectOption("25");
  await expect.poll(() => page.evaluate(() => window.browserHarness.getState().requestDebug)).toMatchObject({enabled: true, limit: 25});
  await page.goto(fixtureUrl("usage-maintenance/request-debug"));
  debug = page.locator("extended-openai-debug-panel");
  await expect(debug.locator("#enabled")).toBeChecked();
  await expect(debug.locator("#limit")).toHaveValue("25");
  await expectHarnessClean(page, errors);
});
