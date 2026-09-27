import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

async function openFunctions(page) {
  await page.goto(fixtureUrl("capabilities/functions"));
  const panel = page.locator("extended-openai-management-panel");
  await panel.locator('.function-group-card[data-group-id="baseline-group"] summary').click();
  await expect(panel.locator(".edit-tool").first()).toBeVisible();
  await page.evaluate(() => {
    const {hass} = window.browserHarness;
    const original = hass.callWS.bind(hass);
    window.toolEditorLoads = [];
    window.toolEditorSettled = 0;
    hass.callWS = async (message) => {
      if (message.section !== "tools" || message.action !== "serialize") return original(message);
      try {
        await new Promise((resolve, reject) => window.toolEditorLoads.push({resolve, reject}));
        return await original(message);
      } finally { window.toolEditorSettled++; }
    };
  });
  return panel;
}

const editedYaml = `spec:
  name: baseline_tool
  description: Edited after loading
  parameters:
    type: object
    properties: {}
function:
  type: script
  sequence: []
`;

test("Function Tool editing waits for delayed YAML and persists the user's edit", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openFunctions(page);
  await panel.locator(".edit-tool").first().click();
  await expect(panel.locator("#tool-dialog")).toHaveJSProperty("open", true);
  const editor = panel.locator("#tool-yaml");
  await expect(editor).not.toBeEditable();
  await expect(editor.locator("..")).toHaveJSProperty("inert", true);
  await expect(panel.locator("#tool-save")).toBeDisabled();
  await expect(panel.locator("#tool-validate")).toBeDisabled();
  await expect(panel.locator("#tool-cancel")).toBeEnabled();
  await expect.poll(() => page.evaluate(() => window.toolEditorLoads.length)).toBe(1);
  await page.evaluate(() => window.toolEditorLoads[0].resolve());
  await expect(editor).toBeEditable();
  await editor.fill(editedYaml);
  await panel.locator("#tool-save").click();
  await expect(panel.locator("#tool-dialog")).toHaveJSProperty("open", false);
  const sentYaml = await page.evaluate(() => window.browserHarness.calls
    .filter((call) => call.section === "tools" && call.action === "validate_yaml")
    .at(-1)?.yaml);
  expect(sentYaml).toBe(editedYaml);
  // The harness rewrites the URL to the application route; re-enter its fixture
  // while retaining the mock backend persisted in localStorage.
  await page.goto(fixtureUrl("capabilities/functions"));
  await panel.locator('.function-group-card[data-group-id="baseline-group"] summary').click();
  await expect(panel.locator(".tool-card").filter({hasText: "baseline_tool"})).toContainText("Edited after loading");
  await expectHarnessClean(page, errors);
});

test("cancelled editor loads cannot overwrite a reopened dialog", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openFunctions(page);
  await panel.locator(".edit-tool").first().click();
  await expect.poll(() => page.evaluate(() => window.toolEditorLoads.length)).toBe(1);
  await panel.locator("#tool-cancel").click();
  await panel.locator(".edit-tool").first().click();
  await expect.poll(() => page.evaluate(() => window.toolEditorLoads.length)).toBe(2);
  await page.evaluate(() => window.toolEditorLoads[1].resolve());
  const editor = panel.locator("#tool-yaml");
  await expect(editor).toBeEditable();
  await editor.fill(editedYaml);
  await page.evaluate(() => window.toolEditorLoads[0].resolve());
  await expect.poll(() => page.evaluate(() => window.toolEditorSettled)).toBe(2);
  await expect(editor).toHaveValue(editedYaml);
  await panel.locator("#tool-save").click();
  await expect(panel.locator("#tool-dialog")).toHaveJSProperty("open", false);
  const sentYaml = await page.evaluate(() => window.browserHarness.calls
    .filter((call) => call.section === "tools" && call.action === "validate_yaml")
    .at(-1)?.yaml);
  expect(sentYaml).toBe(editedYaml);
  // The harness rewrites the URL to the application route; re-enter its fixture
  // while retaining the mock backend persisted in localStorage.
  await page.goto(fixtureUrl("capabilities/functions"));
  await panel.locator('.function-group-card[data-group-id="baseline-group"] summary').click();
  await expect(panel.locator(".tool-card").filter({hasText: "baseline_tool"})).toContainText("Edited after loading");
  await expectHarnessClean(page, errors);
});

test("failed editor loading is visible and can be retried without saving a blank tool", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openFunctions(page);
  await panel.locator(".edit-tool").first().click();
  await expect.poll(() => page.evaluate(() => window.toolEditorLoads.length)).toBe(1);
  await page.evaluate(() => window.toolEditorLoads[0].reject(new Error("YAML temporarily unavailable")));
  await expect(panel.locator("#tool-error")).toHaveText("YAML temporarily unavailable");
  await expect(panel.locator("#tool-dialog")).toHaveAttribute("aria-busy", "false");
  await expect(panel.locator("#tool-save")).toBeDisabled();
  await panel.locator("#tool-cancel").click();
  await panel.locator(".edit-tool").first().click();
  await expect.poll(() => page.evaluate(() => window.toolEditorLoads.length)).toBe(2);
  await page.evaluate(() => window.toolEditorLoads[1].resolve());
  await expect(panel.locator("#tool-yaml")).toBeEditable();
  await expect(panel.locator("#tool-save")).toBeEnabled();
  await panel.locator("#tool-cancel").click();
  await expectHarnessClean(page, errors);
});


for (const kind of ["knowledge", "memory"]) {
  for (const bundled of [false, true]) {
    test(`${kind} initial focus cannot steal input on a later frame (${bundled ? "bundle" : "source"})`, async ({page}) => {
      const errors = trackPageErrors(page);
      await page.goto(fixtureUrl(`data-memory/${kind === "knowledge" ? "knowledge" : "memories"}`, bundled ? "&bundle=1" : ""));
      const panel = page.locator("extended-openai-management-panel");
      await expect(panel.locator(kind === "knowledge" ? "#add-source" : "#add-memory")).toBeVisible();
      // Hold only callbacks scheduled by opening the editor, reproducing the CI
      // ordering without relying on machine speed or adding arbitrary sleeps.
      await panel.evaluate(async (host, kind) => {
        const original = window.requestAnimationFrame;
        window.editorOpeningFrames = [];
        window.requestAnimationFrame = callback => window.editorOpeningFrames.push(callback);
        try {
          if (kind === "knowledge") await host._openKnowledge();
          else await host._openMemory();
        } finally { window.requestAnimationFrame = original; }
      }, kind);
      const first = panel.locator(kind === "knowledge" ? "#knowledge-title" : "#memory-content");
      const next = panel.locator(kind === "knowledge" ? "#knowledge-content" : "#memory-subject");
      await first.fill("First field");
      await next.focus();
      await page.evaluate(() => { for (const callback of window.editorOpeningFrames) callback(performance.now()); });
      await expect(next).toBeFocused();
      await page.keyboard.insertText("Second field");
      await expect(first).toHaveValue("First field");
      await expect(next).toHaveValue("Second field");
      await expectHarnessClean(page, errors);
    });
  }
}
