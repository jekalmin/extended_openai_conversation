import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

async function beforeUnloadIsBlocked(page) {
  return page.evaluate(() => {
    const event = new Event("beforeunload", {cancelable: true});
    const dispatched = window.dispatchEvent(event);
    return event.defaultPrevented || dispatched === false;
  });
}

async function navigateSection(panel, section) {
  const select = panel.locator("#local-section");
  await select.selectOption(section, {force: true});
}

test("dirty navigation survives subsection changes and clears when the draft returns to baseline", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(fixtureUrl("assistant/basics"));

  const panel = page.locator("extended-openai-management-panel");
  const title = panel.locator('[data-config="__title"]');
  await expect(title).toHaveValue("Jarvis");
  await expect(panel.getByText("Unsaved changes", {exact: true})).toHaveCount(0);

  await title.fill("Navigation draft");
  await expect(panel.getByText("Unsaved changes", {exact: true})).toBeVisible();
  await expect(panel.locator('.top-nav button[data-page="assistant"]')).toHaveClass(/eoc-has-unsaved/);
  await expect(panel.locator('#local-section option[value="basics"]')).toHaveText(/Basics\s+•$/);
  expect(await beforeUnloadIsBlocked(page)).toBe(true);

  await navigateSection(panel, "prompt-context");
  await expect(page).toHaveURL(/\/extended-openai\/assistant\/prompt-context$/);
  await expect(panel.locator("#confirm-dialog")).toHaveJSProperty("open", false);
  await expect(panel.getByText("Unsaved changes", {exact: true})).toBeVisible();
  await expect(panel.locator('.top-nav button[data-page="assistant"]')).toHaveClass(/eoc-has-unsaved/);
  await expect(panel.locator('#local-section option[value="basics"]')).toHaveText(/Basics\s+•$/);

  await navigateSection(panel, "basics");
  await expect(page).toHaveURL(/\/extended-openai\/assistant\/basics$/);
  await panel.locator('[data-config="__title"]').fill("Jarvis");
  await expect(panel.getByText("Unsaved changes", {exact: true})).toHaveCount(0);
  await expect(panel.locator('.top-nav button[data-page="assistant"]')).not.toHaveClass(/eoc-has-unsaved/);
  await expect(panel.locator('#local-section option[value="basics"]')).toHaveText("Basics");
  expect(await beforeUnloadIsBlocked(page)).toBe(false);

  await expectHarnessClean(page, pageErrors);
});

test("internal navigation preserves a dirty configuration without a refetch", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(fixtureUrl("assistant/basics"));

  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator('[data-config="__title"]')).toHaveValue("Jarvis");
  const initialReads = await page.evaluate(() => browserHarness.calls.filter(
    (call) => call.section === "configuration" && call.action === "get",
  ).length);
  await panel.locator('[data-config="__title"]').fill("Unsaved navigation title");
  await expect(panel.getByText("Unsaved changes", {exact: true})).toBeVisible();

  await panel.locator('.top-nav button[data-page="overview"]').click();
  await expect(page).toHaveURL(/\/extended-openai\/overview$/);
  await expect(panel.locator("#confirm-dialog")).toHaveJSProperty("open", false);
  expect(await page.evaluate(() => window.browserHarness.getState().configuration.title)).toBe("Jarvis");

  await panel.locator('.top-nav button[data-page="assistant"]').click();
  await expect(page).toHaveURL(/\/extended-openai\/assistant\/basics$/);
  await expect(panel.locator('[data-config="__title"]')).toHaveValue("Unsaved navigation title");
  await expect(panel.getByText("Unsaved changes", {exact: true})).toBeVisible();
  expect(await page.evaluate(() => browserHarness.calls.filter(
    (call) => call.section === "configuration" && call.action === "get",
  ).length)).toBe(initialReads);
  await expect(panel.locator('.top-nav button[data-page="assistant"]')).toHaveClass(/eoc-has-unsaved/);

  await expectHarnessClean(page, pageErrors);
});

test("a clean Assistant snapshot is reusable after visiting Usage", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(fixtureUrl("assistant/basics"));
  console.log(pageErrors);
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator('[data-config="__title"]')).toHaveValue("Jarvis");
  const initialReads = await page.evaluate(() => browserHarness.calls.filter(
    (call) => call.section === "configuration" && call.action === "get",
  ).length);
  await panel.locator('.top-nav button[data-page="usage-maintenance"]').click();
  await expect(panel.locator("#usage-window")).toBeVisible();
  await panel.locator('.top-nav button[data-page="assistant"]').click();
  await expect(panel.locator('[data-config="__title"]')).toHaveValue("Jarvis");
  expect(await page.evaluate(() => browserHarness.calls.filter(
    (call) => call.section === "configuration" && call.action === "get",
  ).length)).toBe(initialReads);
  await expectHarnessClean(page, pageErrors);
});
