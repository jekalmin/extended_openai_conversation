import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

const validYaml = (name = "nightly_tool", description = "Nightly Function Tool") => `spec:
  name: ${name}
  description: ${description}
  parameters:
    type: object
    properties: {}
function:
  type: native
  name: get_user_from_user_id
`;

async function openAddTool(panel) {
  await panel.locator("#add-tool").click();
  await expect(panel.locator("#tool-dialog")).toHaveJSProperty("open", true);
  await expect(panel.locator("#tool-yaml")).toBeEditable();
  return panel.locator("#tool-yaml");
}

test("nightly Function Tool validation rejects malformed/schema/dependency cases without saving", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/functions"));
  const panel = page.locator("extended-openai-management-panel");
  const editor = await openAddTool(panel);

  await panel.evaluate((host) => {
    const original = host._call.bind(host);
    window.functionValidationMode = "valid";
    window.functionValidationAttempts = [];
    host._call = async (section, action, payload) => {
      if (section === "tools" && action === "validate_yaml") {
        window.functionValidationAttempts.push({mode:window.functionValidationMode, yaml:payload.yaml});
        const cases = {
          malformed: {valid:false, errors:{yaml:"YAML could not be parsed"}},
          schema: {valid:false, errors:{"spec.parameters":"must be an object schema"}},
          unsupported: {valid:false, errors:{"function.type":"unsupported Function Tool implementation"}},
          dependency: {valid:false, errors:{"function.entity_id":"Referenced Home Assistant entity is unavailable"}},
        };
        if (cases[window.functionValidationMode]) return structuredClone(cases[window.functionValidationMode]);
      }
      return original(section, action, payload);
    };
  });

  for (const [mode, yaml, message] of [
    ["malformed", "spec: [", "YAML could not be parsed"],
    ["schema", validYaml("schema_probe"), "must be an object schema"],
    ["unsupported", validYaml("unsupported_probe"), "unsupported Function Tool implementation"],
    ["dependency", validYaml("dependency_probe"), "Referenced Home Assistant entity is unavailable"],
  ]) {
    await page.evaluate((value) => { functionValidationMode = value; }, mode);
    await editor.fill(yaml);
    await panel.locator("#tool-validate").click();
    await expect(panel.locator("#tool-error")).toHaveClass(/invalid/);
    await expect(panel.locator("#tool-error")).toContainText(message);
    await expect(panel.locator("#tool-dialog")).toHaveJSProperty("open", true);
    expect(await page.evaluate(() => browserHarness.calls.filter(
      call => call.section === "tools" && call.action === "save",
    ).length)).toBe(0);
  }

  await page.evaluate(() => { functionValidationMode = "valid"; });
  await editor.fill(validYaml("validated_tool"));
  await panel.locator("#tool-validate").click();
  await expect(panel.locator("#tool-error")).toHaveClass(/valid/);
  await expect(panel.locator("#tool-error")).toContainText("validated_tool");
  await panel.locator("#tool-save").click();
  await expect(panel.locator(".tool-card").filter({hasText:"validated_tool"})).toBeVisible();
  expect(await page.evaluate(() => functionValidationAttempts.length)).toBeGreaterThanOrEqual(5);
  await expectHarnessClean(page, errors);
});

test("nightly Function Tool save failure preserves the editor and a corrected retry succeeds", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/functions"));
  const panel = page.locator("extended-openai-management-panel");
  const editor = await openAddTool(panel);

  await panel.evaluate((host) => {
    const original = host._call.bind(host);
    window.injectedToolSaveAttempts = 0;
    host._call = async (section, action, payload) => {
      if (section === "tools" && action === "save") {
        window.injectedToolSaveAttempts += 1;
        if (window.injectedToolSaveAttempts === 1) throw new Error("Function name already exists");
      }
      return original(section, action, payload);
    };
  });

  await editor.fill(validYaml("baseline_tool", "Duplicate name probe"));
  await panel.locator("#tool-save").click();
  await expect(panel.locator("#tool-dialog")).toHaveJSProperty("open", true);
  await expect(panel.locator("#tool-error")).toContainText("Function name already exists");
  await expect(panel.locator("#tool-save")).toBeEnabled();
  await expect(editor).toHaveValue(/baseline_tool/);

  await editor.fill(validYaml("recovered_tool", "Recovered after duplicate"));
  await panel.locator("#tool-save").click();
  await expect(panel.locator("#tool-dialog")).not.toHaveJSProperty("open", true);
  await expect(panel.locator(".tool-card").filter({hasText:"recovered_tool"})).toContainText("Recovered after duplicate");
  expect(await page.evaluate(() => injectedToolSaveAttempts)).toBe(2);
  await expectHarnessClean(page, errors);
});

test("nightly saved-configuration validation reports mixed failures and recovers", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/functions"));
  const panel = page.locator("extended-openai-management-panel");
  const baselineCount = await panel.locator(".tool-card").count();

  await panel.evaluate((host) => {
    const original = host._call.bind(host);
    window.validateCurrentMode = 0;
    host._call = async (section, action, payload) => {
      if (section === "tools" && action === "validate_current") {
        window.validateCurrentMode += 1;
        if (window.validateCurrentMode === 1) return {
          valid:false,
          errors:{
            baseline_tool:"Missing dependency: sensor.unavailable",
            disabled_tool:"Unavailable implementation is retained but disabled",
          },
        };
        if (window.validateCurrentMode === 2) throw new Error("Validation service temporarily unavailable");
        return {valid:true, errors:{}};
      }
      return original(section, action, payload);
    };
  });

  const button = panel.locator("#validate-tools");
  await button.click();
  await expect(panel.locator("#tool-status")).toHaveClass(/invalid/);
  await expect(panel.locator("#tool-status")).toContainText("Missing dependency");
  await expect(panel.locator("#tool-status")).toContainText("disabled_tool");

  await button.click();
  await expect(panel.locator("#tool-status")).toContainText("Validation service temporarily unavailable");
  await button.click();
  await expect(panel.locator("#tool-status")).toHaveClass(/valid/);
  await expect(panel.locator("#tool-status")).toHaveText("All saved tools and groups are valid");
  await expect(panel.locator(".tool-card")).toHaveCount(baselineCount);
  await expectHarnessClean(page, errors);
});

test("nightly built-in preset loading and replacement failure paths remain recoverable", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/functions"));
  const panel = page.locator("extended-openai-management-panel");

  await panel.evaluate((host) => {
    const original = host._call.bind(host);
    window.catalogAttempts = 0;
    const presetYaml = (name) => `spec:
  name: ${name}
  description: Preset Function Tool
  parameters:
    type: object
    properties: {}
function:
  type: native
  name: get_user_from_user_id
`;
    host._call = async (section, action, payload) => {
      if (section === "tools" && action === "built_in_catalog") {
        window.catalogAttempts += 1;
        if (window.catalogAttempts === 1) throw new Error("Built-in catalogue unavailable");
        return {
          functions:[
            {
              implementation:"configured_preset",
              label:"Configured preset",
              yaml:presetYaml("configured_tool"),
              already_configured:true,
            },
            {
              implementation:"available_preset",
              label:"Available preset",
              yaml:presetYaml("preset_tool"),
              already_configured:false,
            },
          ],
        };
      }
      return original(section, action, payload);
    };
  });

  await panel.locator("#add-tool").click();
  await expect(panel.locator("#tool-error")).toContainText("Built-in catalogue unavailable");
  await expect(panel.locator("#tool-save")).toBeDisabled();
  await panel.locator("#tool-cancel").click();

  const editor = await openAddTool(panel);
  await expect(panel.locator('#built-in-function option[value="configured_preset"]')).toBeDisabled();
  await expect(panel.locator('#built-in-function option[value="available_preset"]')).toBeEnabled();

  const unsaved = validYaml("custom_unsaved", "Keep me");
  await editor.fill(unsaved);
  await panel.locator("#built-in-function").selectOption("available_preset");
  await expect(panel.locator("#confirm-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#confirm-cancel").click();
  await expect(editor).toHaveValue(unsaved);
  await expect(panel.locator("#built-in-function")).toHaveValue("");

  await panel.locator("#built-in-function").selectOption("available_preset");
  await panel.locator("#confirm-accept").click();
  await expect(editor).toHaveValue(/preset_tool/);
  await expect(panel.locator("#tool-error")).toHaveClass(/valid/);
  await panel.locator("#tool-save").click();
  await expect(panel.locator(".tool-card").filter({hasText:"preset_tool"})).toBeVisible();
  expect(await page.evaluate(() => window.catalogAttempts)).toBe(2);
  await expectHarnessClean(page, errors);
});
