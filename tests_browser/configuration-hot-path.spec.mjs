import {expect, test} from "@playwright/test";
import {fixtureUrl as routeFixture, trackPageErrors, expectHarnessClean} from "./browser-helpers.mjs";

for (const bundle of [false, true]) test.describe(bundle ? "production configuration inputs" : "source configuration inputs", () => {
const fixtureUrl = (route) => routeFixture(route, bundle ? "&bundle=1" : "");

test("first and subsequent edits retain the draft, read no other controls and propagate normally", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("assistant/basics"));
  const host = page.locator("extended-openai-management-panel");
  await expect(host.locator('[data-config="__title"]')).toBeVisible();
  const result = await page.evaluate(() => {
    const {panel} = window.browserHarness;
    const root = panel.shadowRoot, draft = panel._draft;
    const control = root.querySelector('[data-config="__title"]');
    const original = root.querySelectorAll;
    let broadReads = 0, propagated = 0;
    root.querySelectorAll = function(selector) {
      if (selector.includes("[data-config]") || selector.includes("[data-memory-config]")) broadReads++;
      return original.call(this, selector);
    };
    root.addEventListener("input", () => propagated++);
    try {
      control.value = "First edit";
      control.dispatchEvent(new Event("input", {bubbles:true, composed:true}));
      control.value = "Second edit";
      control.dispatchEvent(new Event("input", {bubbles:true, composed:true}));
      return {broadReads, propagated, sameDraft:panel._draft === draft, title:panel._draftTitle};
    } finally { root.querySelectorAll = original; }
  });
  console.log("Configuration hot path", result);
  expect(result).toEqual({broadReads:0, propagated:2, sameDraft:true, title:"Second edit"});
  await expect(host.locator(".save-bar")).toBeVisible();
  await expectHarnessClean(page, errors);
});

test("returning to baseline restores saved-state actions without rebuilding the form", async ({page}) => {
  await page.goto(fixtureUrl("assistant/basics"));
  const host = page.locator("extended-openai-management-panel");
  const title = host.locator('[data-config="__title"]');
  await expect(title).toBeVisible();
  const baseline = await title.inputValue();
  await title.fill("Changed title");
  await expect(host.locator("#duplicate-agent")).toBeDisabled();
  await page.evaluate(async () => {
    const {panel} = window.browserHarness;
    await panel._navigate("assistant", "conversation");
    await panel._navigate("assistant", "basics");
    window.cleanInput = panel.shadowRoot.querySelector('[data-config="__title"]');
  });
  await title.fill(baseline);
  await expect(host.locator("#duplicate-agent")).toBeEnabled();
  await expect(host.locator("#export-agent")).toBeEnabled();
  await expect(host.locator(".save-bar")).toHaveCount(0);
  expect(await page.evaluate(() => window.cleanInput === window.browserHarness.panel.shadowRoot.querySelector('[data-config="__title"]'))).toBe(true);
});

test("Skills uses newline parsing on every edit, preserving commas in names", async ({page}) => {
  await page.goto(fixtureUrl("capabilities/web-skills"));
  const skills = page.locator('extended-openai-management-panel [data-config="skills"]');
  await skills.fill("weather, indoor\ncalendar");
  await skills.fill("weather, indoor\ncalendar\nlocal notes");
  expect(await page.evaluate(() => window.browserHarness.panel._draft.skills)).toEqual(["weather, indoor", "calendar", "local notes"]);
});

test("delegation owns dynamically inserted controls once across input and change", async ({page}) => {
  await page.goto(fixtureUrl("assistant/basics"));
  await expect(page.locator('extended-openai-management-panel [data-config="__title"]')).toBeVisible();
  const result = await page.evaluate(async () => {
    const {bindConfigurationInputs} = await import("/custom_components/extended_openai_conversation_responses/frontend/configuration-inputs.js");
    const {panel} = window.browserHarness;
    bindConfigurationInputs(panel);
    bindConfigurationInputs(panel);
    let mutations = 0;
    panel._draft = new Proxy(panel._draft, {set(target, key, value) { if (key === "skills") mutations++; target[key] = value; return true; }});
    const input = document.createElement("textarea");
    input.dataset.config = "skills";
    input.value = "weather, indoor\ncalendar";
    panel.shadowRoot.querySelector("main").append(input);
    input.dispatchEvent(new Event("input", {bubbles:true}));
    input.dispatchEvent(new Event("change", {bubbles:true}));
    return {mutations, skills:panel._draft.skills, dirty:[...panel._eocDirtyConfigKeys]};
  });
  expect(result).toEqual({mutations:1, skills:["weather, indoor","calendar"], dirty:["skills"]});
});

test("model selection has one lookup/validation path and retains reasoning defaults", async ({page}) => {
  await page.goto(fixtureUrl("assistant/basics"));
  const model = page.locator('extended-openai-management-panel [data-config="chat_model"]');
  await expect(model).toBeVisible();
  await expect.poll(() => page.evaluate(() => Boolean(window.browserHarness.panel._modelCatalogData))).toBe(true);
  await page.evaluate(() => {
    const {panel} = window.browserHarness;
    const call = panel._hass.callWS.bind(panel._hass);
    window.modelCalls = [];
    panel._hass.callWS = async (message) => {
      window.modelCalls.push(message);
      const response = await call(message);
      if (message.type.endsWith("/model_catalog")) response.model_metadata = {reasoning:{supported:true, efforts:["low","high"]}, recommended_profile:{reasoning_effort:"high"}};
      return response;
    };
  });
  await model.fill("gpt-4.1");
  await model.press("Tab");
  await expect.poll(() => page.evaluate(() => window.modelCalls.filter((m) => m.section === "configuration" && m.action === "validate").length)).toBe(1);
  const result = await page.evaluate(() => {
    const {panel} = window.browserHarness;
    return {lookups:window.modelCalls.filter((m) => m.type.endsWith("/model_catalog")).length, model:panel._draft.chat_model, requested:panel._modelCatalogData.requested_model, dirty:panel._eocDirtyConfigKeys.has("chat_model"), reasoning:panel._draft.reasoning_effort};
  });
  expect(result).toEqual({lookups:1, model:"gpt-4.1", requested:"gpt-4.1", dirty:true, reasoning:"high"});
});

test("a late model catalog response does not redirect typing from the title into the model", async ({page}) => {
  await page.goto(fixtureUrl("assistant/basics"));
  const panel = page.locator("extended-openai-management-panel");
  const title = panel.locator('[data-config="__title"]');
  const model = panel.locator('[data-config="chat_model"]');
  await expect(title).toBeVisible();
  await expect.poll(() => page.evaluate(() => Boolean(window.browserHarness.panel._modelCatalogData))).toBe(true);
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const originalModel = await model.inputValue();
  await page.evaluate(() => {
    const {panel} = window.browserHarness;
    const call = panel._hass.callWS.bind(panel._hass);
    panel._hass.callWS = (message) => message.type.endsWith("/model_catalog")
      ? new Promise((resolve, reject) => { window.resolveLateCatalog = () => call(message).then(resolve, reject); })
      : call(message);
    panel._modelCatalogData = null;
    panel._render();
  });
  await expect.poll(() => page.evaluate(() => typeof window.resolveLateCatalog)).toBe("function");
  await title.fill("Late catalog title");
  await expect.poll(() => page.evaluate(() => window.browserHarness.panel.shadowRoot.activeElement?.dataset.config)).toBe("__title");
  await page.evaluate(() => window.resolveLateCatalog());
  await page.keyboard.insertText("!");
  const typedTitle = await title.inputValue();
  expect(typedTitle).toMatch(/^(?:!Late catalog title|Late catalog title!)$/);
  await expect(model).toHaveValue(originalModel);
  await expect.poll(() => page.evaluate(() => window.browserHarness.panel.shadowRoot.activeElement?.dataset.config)).toBe("__title");
  await title.press("Tab");
  await expect.poll(() => page.evaluate(() => window.browserHarness.panel._modelCatalogData?.requested_model)).toBe(originalModel);
  await expect(title).toHaveValue(typedTitle);
  await expect(model).toHaveValue(originalModel);
});

test("a Web Search toggle refreshes capability controls", async ({page}) => {
  await page.goto(fixtureUrl("capabilities/web-skills"));
  const web = page.locator('extended-openai-management-panel [data-config="web_search"]');
  await expect(web).toBeVisible();
  const result = await page.evaluate(async () => {
    const {panel} = window.browserHarness;
    const input = panel.shadowRoot.querySelector('[data-config="web_search"]');
    const next = !input.checked;
    input.checked = next;
    input.dispatchEvent(new Event("input", {bubbles:true}));
    input.dispatchEvent(new Event("change", {bubbles:true}));
    await Promise.resolve();
    return {
      draft: panel._draft.web_search,
      checked: panel.shadowRoot.querySelector('[data-config="web_search"]').checked,
      detailInactive: panel.shadowRoot.querySelector('[data-dependent="web_search"]').classList.contains("is-disabled"),
    };
  });
  expect(result.checked).toBe(result.draft);
  expect(result.detailInactive).toBe(!result.draft);
});

test("Memory controls share delegation, preserve the draft and toggle their dependent field", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("data-memory/memory-settings"));
  const host = page.locator("extended-openai-management-panel");
  const retrieval = host.locator('[data-memory-config="memory_retrieval_mode"]');
  await expect(retrieval).toBeVisible();
  await page.evaluate(() => { window.initialMemoryDraft = window.browserHarness.panel._draft; });
  await retrieval.selectOption("hybrid");
  await expect(host.locator('[data-memory-config="memory_embedding_model"]')).toBeEnabled();
  await host.locator('[data-memory-config="memory_auto_retrieve_limit"]').fill("3");
  expect(await page.evaluate(() => ({same:window.initialMemoryDraft === window.browserHarness.panel._draft, value:window.browserHarness.panel._draft.memory_auto_retrieve_limit}))).toEqual({same:true, value:3});
  await expect(host.locator("#save-config")).toBeVisible();
  await host.locator("#revert-config").click();
  await expect(host.locator(".save-bar")).toHaveCount(0);
  await expectHarnessClean(page, errors);
});

test("conversation presets update only the numeric field and Custom retains its value", async ({page}) => {
  await page.goto(fixtureUrl("assistant/conversation"));
  const host = page.locator("extended-openai-management-panel");
  await expect(host.locator("#conversation-timeout-preset")).toBeVisible();
  await page.evaluate(() => {
    const {panel} = window.browserHarness;
    Object.assign(panel._configData.options, {conversation_continuity:[{value:"ha_default",label:"Home Assistant"},{value:"device",label:"Device"}], conversation_timeout_minutes:[{value:10,label:"10 minutes"},{value:30,label:"30 minutes"}]});
    panel._draft.conversation_continuity = "ha_default";
    panel._draft.conversation_timeout_minutes = 10;
    panel._render();
  });
  await host.locator('[data-config="conversation_continuity"]').selectOption("device");
  const preset = host.locator("#conversation-timeout-preset");
  await preset.selectOption("30");
  expect(await page.evaluate(() => window.browserHarness.panel._draft.conversation_timeout_minutes)).toBe(30);
  await preset.selectOption("custom");
  await expect(host.locator('[data-config="conversation_timeout_minutes"]')).toBeFocused();
  expect(await page.evaluate(() => window.browserHarness.panel._draft.conversation_timeout_minutes)).toBe(30);
});

});
