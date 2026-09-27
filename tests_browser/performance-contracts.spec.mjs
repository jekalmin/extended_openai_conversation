import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

for (const agentCount of [1, 50]) {
  test(`Overview, configuration, and save have bounded work with ${agentCount} agents`, async ({page}) => {
    const errors = trackPageErrors(page);
    const assets = [];
    page.on("request", request => {
      if (request.url().includes("/frontend/")) assets.push(new URL(request.url()).pathname);
    });
    await page.goto(fixtureUrl("overview", `&agents=${agentCount}&bundle=1`));
    const panel = page.locator("extended-openai-management-panel");
    await expect(panel.locator(".dashboard-grid")).toBeVisible();
    await expect.poll(() => page.evaluate(() => browserHarness.panel._data?.agents?.length)).toBe(agentCount);
    const overview = await page.evaluate(() => browserHarness.calls.map(call => `${call.section || "root"}/${call.action}`));
    expect(overview.filter(call => call === "root/agents")).toHaveLength(1);
    expect(overview.filter(call => call === "overview/primary").length).toBeLessThanOrEqual(1);
    expect(overview.filter(call => call === "overview/summary").length).toBeLessThanOrEqual(1);
    expect(overview).not.toContain("configuration/get");

    await panel.evaluate(host => host._navigate("assistant", "basics"));
    const tokens = panel.locator('[data-config="max_tokens"]');
    await expect(tokens).toBeVisible();
    const beforeSave = await page.evaluate(() => browserHarness.calls.length);
    await tokens.fill("1300");
    await panel.locator("#save-config").click();
    await expect(tokens).toHaveValue("1300");
    const calls = await page.evaluate(start => browserHarness.calls.slice(start).map(call => `${call.section || "root"}/${call.action}`), beforeSave);
    expect(calls.filter(call => call === "configuration/save")).toHaveLength(1);
    expect(calls).not.toContain("root/agents");
    expect(calls).not.toContain("configuration/validate");
    console.log("Performance contract", {agentCount, overview, saveCalls:calls, assetRequests:assets.length});
    expect(new Set(assets).size).toBe(assets.length);
    await expectHarnessClean(page, errors);
  });
}
