import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

test("Diagnostics renders the selected agent and a real result from its management action", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("usage-maintenance/diagnostics"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator("[data-eoc-main]")).toContainText("Jarvis");
  await panel.locator("#test-agent").click();
  await expect(panel.locator("#test-result .diagnostic-summary.passed")).toBeVisible();
  await expect(panel.locator("#test-result .diagnostic-check")).toContainText("Model access");
  expect(await page.evaluate(() => window.browserHarness.calls.filter(call =>
    call.section === "diagnostics" && call.action === "test_agent").length)).toBe(1);
  await expectHarnessClean(page, errors);
});
