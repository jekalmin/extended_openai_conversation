import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

for (const bundled of [false, true]) {
  test(`Prompt controls remain editable while attribute code loads (${bundled ? "bundle" : "source"})`, async ({page}) => {
    const errors = trackPageErrors(page);
    let release;
    await page.route(/\/exposed-attributes-ui(?:-[^/]+)?\.js$/, async (route) => {
      await new Promise((resolve) => { release = resolve; });
      await route.continue();
    });
    await page.goto(fixtureUrl("assistant/prompt-context", bundled ? "&bundle=1" : ""), {waitUntil: "domcontentloaded"});
    const panel = page.locator("extended-openai-management-panel");
    await expect(panel.locator("#prompt-editor")).toBeVisible();
    await expect.poll(() => Boolean(release)).toBe(true);
    await expect(panel.locator("[data-exposed-feature]")).toContainText("Loading Assist-exposed entity choices");
    await panel.locator("#prompt-editor").fill("Keep this unsaved prompt");
    await panel.evaluate((host) => { host._draft.exposed_entity_attributes = {"registry:missing": ["brightness"]}; });
    release();
    await expect(panel.locator(".exposed-attribute-settings")).toContainText("Additional entity attributes");
    await expect(panel.locator(".exposed-attribute-settings")).not.toContainText("Loading Assist-exposed entity choices");
    expect(await panel.evaluate((host) => ({prompt: host._draft.prompt, attributes: host._draft.exposed_entity_attributes}))).toEqual({
      prompt: "Keep this unsaved prompt", attributes: {"registry:missing": ["brightness"]},
    });
    await expectHarnessClean(page, errors);
  });
}

test("saved device assignments hydrate after the Voice policy core and survive route changes", async ({page}) => {
  const errors = trackPageErrors(page);
  let release;
  await page.route(/\/voice-identity-ui\.js$/, async (route) => {
    await new Promise((resolve) => { release = resolve; });
    await route.continue();
  });
  await page.goto(fixtureUrl("assistant/basics"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator('[data-config="chat_model"]')).toBeVisible();
  await panel.evaluate(async (host) => {
    host._draft.voice_scope_policy = "device_mapping";
    host._draft.voice_device_mappings = {"device-kitchen": "user:test-user"};
    host._configData.config.voice_scope_policy = "device_mapping";
    host._configData.config.voice_device_mappings = {"device-kitchen": "user:test-user"};
    await host._navigate("assistant", "voice");
  });
  await expect(panel.locator(".voice-identity-flow")).toBeHidden();
  await expect(panel.locator(".voice-identity-help > summary")).toBeVisible();
  await expect(panel.locator("#voice-current-summary")).toBeVisible();
  await expect.poll(() => Boolean(release)).toBe(true);
  await expect(panel.locator("[data-voice-mapping-feature]")).toContainText("Loading saved assignments");
  expect(await panel.evaluate((host) => host._draft.voice_device_mappings)).toEqual({"device-kitchen": "user:test-user"});
  release();
  await expect(panel.locator('#voice-mappings [data-voice-mapping-row]')).toHaveCount(1);
  await expect(panel.locator(".voice-device-id")).toHaveValue("device-kitchen");
  await panel.evaluate((host) => host._navigate("assistant", "basics"));
  await expect(panel.locator("#voice-mappings")).toHaveCount(0);
  await expectHarnessClean(page, errors);
});

test("a delayed mapping module cannot decorate a later route", async ({page}) => {
  let release;
  await page.route(/\/voice-identity-ui\.js$/, async (route) => {
    await new Promise((resolve) => { release = resolve; });
    await route.continue();
  });
  await page.goto(fixtureUrl("assistant/voice"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".voice-identity-flow")).toBeHidden();
  await expect(panel.locator(".voice-identity-help > summary")).toBeVisible();
  await expect(panel.locator("#voice-current-summary")).toBeVisible();
  await panel.locator('[data-config="voice_scope_policy"]').selectOption("device_mapping");
  await expect.poll(() => Boolean(release)).toBe(true);
  await panel.evaluate((host) => host._navigate("assistant", "basics"));
  await expect(panel.locator('[data-config="chat_model"]')).toBeVisible();
  release();
  await expect(panel.locator("#voice-mappings")).toHaveCount(0);
  expect(await panel.evaluate((host) => host._viewKey())).toBe("assistant/basics");
});
