import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

async function trackRenders(panel) {
  await panel.evaluate((host) => {
    const original = host._render.bind(host);
    window.pageSaveRenderCount = 0;
    host._render = (...args) => {
      window.pageSaveRenderCount++;
      return original(...args);
    };
    window.pageSaveMain = host.shadowRoot.querySelector("main");
  });
}

test("Request Rules page save preserves the route DOM", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = page.locator("extended-openai-management-panel");
  await panel.locator(".rule-settings details").first().evaluate((node) => { node.open = true; });
  await expect(panel.locator("#rules-default-fuzzy")).toBeVisible();
  await trackRenders(panel);

  await panel.locator(".wording-editor").evaluate((node) => { node.open = true; });
  await panel.locator("#rules-default-fuzzy").check();
  await expect(panel.locator(".save-bar")).toBeVisible();
  await panel.locator("#save-page").click();
  await expect(panel.locator(".save-bar")).toHaveCount(0);

  expect(await panel.evaluate((host) => ({
    sameMain: host.shadowRoot.querySelector("main") === window.pageSaveMain,
    renders: window.pageSaveRenderCount,
    wordingOpen: host.shadowRoot.querySelector(".wording-editor")?.open,
    fuzzy: host.shadowRoot.querySelector("#rules-default-fuzzy")?.checked,
  }))).toEqual({sameMain:true, renders:0, wordingOpen:true, fuzzy:true});
  await expectHarnessClean(page, errors);
});

test("Quiet Hours page save preserves DOM and patches saved status", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/quiet-hours"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator("#qh-enabled")).toBeVisible();
  await trackRenders(panel);

  const original = await panel.locator("#qh-enabled").isChecked();
  await panel.locator("#qh-enabled").setChecked(!original);
  await expect(panel.locator(".save-bar")).toBeVisible();
  await panel.locator("#save-page").click();
  await expect(panel.locator(".save-bar")).toHaveCount(0);

  const state = await panel.evaluate((host) => ({
    sameMain: host.shadowRoot.querySelector("main") === window.pageSaveMain,
    renders: window.pageSaveRenderCount,
    checked: host.shadowRoot.querySelector("#qh-enabled")?.checked,
    status: host.shadowRoot.querySelector(".qh-status strong")?.textContent,
  }));
  expect(state.sameMain).toBe(true);
  expect(state.renders).toBe(0);
  expect(state.checked).toBe(!original);
  expect(state.status).toBe(!original ? "Outside Quiet Hours" : "Quiet Hours schedule disabled");
  await expectHarnessClean(page, errors);
});

test("Guest Mode policy save preserves DOM", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/guest-mode"));
  const panel = page.locator("extended-openai-management-panel");
  const advanced = panel.locator(".guest-advanced:has(#guest-controls-enabled)");
  await advanced.evaluate((node) => { node.open = true; });
  await expect(panel.locator("#guest-controls-enabled")).toBeVisible();
  await trackRenders(panel);
  const toggle = panel.locator("#guest-controls-enabled");
  const original = await toggle.isChecked();
  await toggle.setChecked(!original);
  await expect(panel.locator(".save-bar")).toBeVisible();
  await panel.locator("#save-page").click();
  await expect(panel.locator(".save-bar")).toHaveCount(0);

  expect(await panel.evaluate((host) => ({
    sameMain: host.shadowRoot.querySelector("main") === window.pageSaveMain,
    renders: window.pageSaveRenderCount,
    advancedOpen: host.shadowRoot.querySelector("#guest-controls-enabled")?.closest("details")?.open,
    checked: host.shadowRoot.querySelector("#guest-controls-enabled")?.checked,
  }))).toEqual({sameMain:true, renders:0, advancedOpen:true, checked:!original});
  await expectHarnessClean(page, errors);
});
