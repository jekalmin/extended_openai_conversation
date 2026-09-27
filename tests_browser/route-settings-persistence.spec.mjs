import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

// Each case edits a control owned by its route, not a shared header control.
// This complements the deeper focused journeys linked from the inventory.
const cases = [
  ["assistant/model-responses", "shorten_tool_call_id", "checkbox", true],
  ["assistant/conversation", "context_threshold", "number", "65"],
  ["assistant/prompt-context", "prompt", "text", "Route-specific prompt saved by browser acceptance"],
  ["assistant/voice", "voice_scope_policy", "select", "shared"],
  ["assistant/speech", "speech_processing_enabled", "checkbox", true],
  ["capabilities/home-assistant", "local_intents_enabled", "checkbox", true],
  ["capabilities/web-skills", "web_search", "checkbox", true],
];

for (const [route, field, kind, value] of cases) {
  test(`route setting persists: ${route}`, async ({page}) => {
    const errors = trackPageErrors(page);
    await page.goto(fixtureUrl(route));
    let panel = page.locator("extended-openai-management-panel");
    let control = panel.locator(`[data-eoc-main] [data-config="${field}"]`);
    await expect(control).toBeVisible();
    if (kind === "checkbox") await control.setChecked(value);
    else if (kind === "select") await control.selectOption(value);
    else await control.fill(value);
    await expect(panel.getByText("Unsaved changes", {exact: true})).toBeVisible();
    await panel.getByRole("button", {name: "Save changes", exact: true}).click();
    await expect(panel.getByText("Unsaved changes", {exact: true})).toHaveCount(0);
    const persisted = await page.evaluate(key => window.browserHarness.getState().configuration.config[key], field);
    expect(persisted).toBe(kind === "number" ? Number(value) : value);
    await page.goto(fixtureUrl(route));
    panel = page.locator("extended-openai-management-panel");
    control = panel.locator(`[data-eoc-main] [data-config="${field}"]`);
    if (kind === "checkbox") await expect(control).toBeChecked({checked: value});
    else await expect(control).toHaveValue(String(value));
    await expectHarnessClean(page, errors);
  });
}

for (const [route, selector, kind, value, expected] of [
  ["capabilities/quiet-hours", "#qh-start", "text", "23:00", "23:00"],
  ["capabilities/guest-mode", "#guest-controls-enabled", "checkbox", true, true],
  ["data-memory/memory-settings", '[data-memory-config="memory_auto_retrieve_limit"]', "text", "7", 7],
  ["usage-maintenance/retention", '[data-config="usage_request_retention_days"]', "select", "7", 7],
]) {
  test(`route setting persists: ${route}`, async ({page}) => {
    const errors = trackPageErrors(page);
    await page.goto(fixtureUrl(route));
    let panel = page.locator("extended-openai-management-panel");
    let control = panel.locator(`[data-eoc-main] ${selector}`);
    if (kind === "checkbox") {
      await control.evaluate(input => input.closest("details")?.setAttribute("open", ""));
      await expect(control).toBeVisible();
      await control.setChecked(value);
    } else if (kind === "select") { await expect(control).toBeVisible(); await control.selectOption(value); }
    else { await expect(control).toBeVisible(); await control.fill(value); }
    await panel.locator(route === "capabilities/quiet-hours" || route === "capabilities/guest-mode" ? "#save-page" : "#save-config").click();
    await expect.poll(() => page.evaluate(fn => {
      const state = window.browserHarness.getState();
      if (fn === "quiet") return state.quiet.config.start;
      if (fn === "guest") return state.guest.config.guest_mode_enabled;
      if (fn === "memory") return state.configuration.config.memory_auto_retrieve_limit;
      return state.configuration.config.usage_request_retention_days;
    }, route.includes("quiet") ? "quiet" : route.includes("guest") ? "guest" : route.includes("memory") ? "memory" : "retention")).toBe(expected);
    await page.goto(fixtureUrl(route));
    panel = page.locator("extended-openai-management-panel");
    control = panel.locator(`[data-eoc-main] ${selector}`);
    if (kind === "checkbox") await expect(control).toBeChecked();
    else await expect(control).toHaveValue(value);
    await expectHarnessClean(page, errors);
  });
}
