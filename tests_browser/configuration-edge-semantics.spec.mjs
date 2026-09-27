import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

const cases = [
  {route: "assistant/model-responses", field: "shorten_tool_call_id", selector: '[data-config="shorten_tool_call_id"]', populated: true, cleared: false},
  {route: "data-memory/memory-settings", field: "memory_auto_retrieve_limit", selector: '[data-memory-config="memory_auto_retrieve_limit"]', populated: 7, cleared: 0},
  {route: "assistant/prompt-context", field: "prompt", selector: '[data-config="prompt"]', populated: "Custom prompt", cleared: ""},
  {route: "capabilities/web-skills", field: "skills", selector: '[data-config="skills"]', populated: ["weather", "calendar"], cleared: []},
];

function inputValue(field, value) {
  return field === "skills" ? value.join("\n") : String(value);
}

async function edit(control, field, value) {
  if (typeof value === "boolean") await control.setChecked(value);
  else await control.fill(inputValue(field, value));
}

for (const {route, field, selector, populated, cleared} of cases) {
  test(`shipped configuration control preserves ${field} clearing semantics`, async ({page}) => {
    const errors = trackPageErrors(page);
    const url = fixtureUrl(route, "&bundle=1");
    await page.goto(url);
    let panel = page.locator("extended-openai-management-panel");
    let control = panel.locator(selector);
    await expect(control).toBeVisible();
    await edit(control, field, populated);
    await panel.getByRole("button", {name: "Save changes", exact: true}).click();
    await expect.poll(() => page.evaluate((key) => window.browserHarness.getState().configuration.config[key], field)).toEqual(populated);

    await edit(control, field, cleared);
    await expect(panel.getByText("Unsaved changes", {exact: true})).toBeVisible();
    await panel.getByRole("button", {name: "Save changes", exact: true}).click();
    await expect.poll(() => page.evaluate((key) => window.browserHarness.getState().configuration.config[key], field)).toEqual(cleared);
    const writes = await page.evaluate(() => window.browserHarness.calls.filter((call) => call.section === "configuration" && call.action === "save"));
    expect(writes).toHaveLength(2);
    expect(writes.at(-1).config).toEqual({[field]: cleared});
    expect(await page.evaluate(() => window.browserHarness.getState().configuration.config.chat_model)).toBe("gpt-5-mini");

    await page.goto(url);
    panel = page.locator("extended-openai-management-panel");
    control = panel.locator(selector);
    if (typeof cleared === "boolean") await expect(control).toBeChecked({checked: cleared});
    else await expect(control).toHaveValue(inputValue(field, cleared));
    await expectHarnessClean(page, errors);
  });
}

test("disabling Web Search retains its saved detail preference", async ({page}) => {
  const errors = trackPageErrors(page);
  const url = fixtureUrl("capabilities/web-skills", "&bundle=1");
  await page.goto(url);
  let panel = page.locator("extended-openai-management-panel");
  const search = panel.locator('[data-config="web_search"]');
  let detail = panel.locator('[data-config="web_search_context"]');
  await expect(detail).toBeDisabled();
  await search.check();
  await expect(detail).toBeEnabled();
  await detail.selectOption("high");
  await panel.getByRole("button", {name: "Save changes", exact: true}).click();
  await expect.poll(() => page.evaluate(() => window.browserHarness.getState().configuration.config.web_search_context)).toBe("high");
  await search.uncheck();
  await expect(detail).toBeDisabled();
  await panel.getByRole("button", {name: "Save changes", exact: true}).click();
  const writes = await page.evaluate(() => window.browserHarness.calls.filter((call) => call.section === "configuration" && call.action === "save"));
  expect(writes.at(-1).config).toEqual({web_search: false});
  expect(await page.evaluate(() => window.browserHarness.getState().configuration.config.web_search_context)).toBe("high");
  await page.goto(url);
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator('[data-config="web_search"]')).toBeChecked({checked: false});
  detail = panel.locator('[data-config="web_search_context"]');
  await expect(detail).toBeDisabled();
  await expect(detail).toHaveValue("high");
  await expectHarnessClean(page, errors);
});
