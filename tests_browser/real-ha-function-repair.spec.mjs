import {expect, test} from "@playwright/test";
import {expectHarnessClean, trackPageErrors} from "./browser-helpers.mjs";
import {expectContractCalls} from "./real-ha-contract.mjs";

const backendUrl = process.env.REAL_HA_BACKEND_URL;
test.skip(!backendUrl, "requires the dedicated genuine Home Assistant backend bridge");
const fixture = `/tests_browser/real-ha-fixture.html?route=capabilities%2Ffunctions&backend=${encodeURIComponent(backendUrl)}`;

test("quarantined Function Tools repair through shipped frontend and genuine HA", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(fixture);
  await page.waitForFunction(() => window.browserHarness?.panel?._selectedAgent?.());
  const before = await page.evaluate(() => window.browserHarness.panel._call("function_repair", "get"));
  expect(before.invalid_tools.map((item) => item.index)).toEqual([0, 2, 3]);

  const safeEdit = await page.evaluate(() => window.browserHarness.panel._call("configuration", "save", {
    config: {prompt: "Nightly safe edit while tools need repair"},
  }));
  expect(safeEdit.valid).toBe(true);

  const changed = await page.evaluate(async () => {
    const panel = window.browserHarness.panel;
    const first = await panel._call("function_repair", "get");
    const replacement = structuredClone(first.tools[1]);
    replacement.spec.name = "nightly_repaired_tool";
    const savedOne = await panel._call("function_repair", "save_one", {
      index: 0, tool: replacement, revision: first.revision,
    });
    const second = await panel._call("function_repair", "get");
    const deletedOne = await panel._call("function_repair", "delete_one", {
      index: 2, revision: second.revision,
    });
    const third = await panel._call("function_repair", "get");
    const saved = await panel._call("function_repair", "save", {
      tools: third.tools.slice(0, 2), revision: third.revision,
    });
    return {savedOne, deletedOne, saved};
  });
  expect(changed.savedOne.revision).toBeTruthy();
  expect(changed.deletedOne.revision).toBeTruthy();
  expect(changed.saved.valid).toBe(true);

  await page.goto(fixture);
  await page.waitForFunction(() => window.browserHarness?.panel?._selectedAgent?.());
  const after = await page.evaluate(async () => {
    const panel = window.browserHarness.panel;
    return panel._call("configuration", "get");
  });
  expect(after.function_repair).toBeUndefined();
  expect(after.config.functions.map((item) => item.spec.name)).toContain("nightly_repaired_tool");
  expect(after.config.prompt).toBe("Nightly safe edit while tools need repair");
  await expectContractCalls(page, "function_repair");
  await expectHarnessClean(page, pageErrors);
});
