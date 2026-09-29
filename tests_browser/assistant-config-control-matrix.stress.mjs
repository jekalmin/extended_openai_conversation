import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

const routes = [
  "assistant/basics", "assistant/conversation", "assistant/model-responses", "assistant/prompt-context",
  "assistant/voice", "assistant/speech", "capabilities/home-assistant", "capabilities/web-skills",
  "data-memory/memory-settings", "usage-maintenance/retention",
];
const panelFor = page => page.locator("extended-openai-management-panel");

function controlValues(panel) {
  return panel.evaluate(host => [...host.shadowRoot.querySelectorAll("[data-config], [data-memory-config]")]
    .filter(control => !control.disabled)
    .map(control => ({
      key:control.dataset.config || control.dataset.memoryConfig,
      kind:control.type || control.tagName.toLowerCase(),
      value:control.type === "checkbox" ? control.checked : control.value,
    })));
}

async function changeControl(panel, key) {
  const control = panel.locator(`[data-config="${key}"], [data-memory-config="${key}"]`).first();
  if (!(await control.count())) return null;
  if (!(await control.isVisible())) return null;
  const state = await control.evaluate(element => ({
    tag:element.tagName.toLowerCase(), type:element.type, disabled:element.disabled,
    min:element.min, max:element.max, value:element.value,
    options:element.tagName === "SELECT" ? [...element.options].filter(option => !option.disabled).map(option => option.value) : [],
  }));
  if (state.disabled) return null;
  if (state.type === "checkbox") {
    await control.setChecked(!(await control.isChecked()));
  } else if (state.tag === "select") {
    const next = state.options.find(value => value !== state.value);
    if (next === undefined) return null;
    await control.selectOption(next);
  } else if (state.type === "number" || state.type === "range") {
    const min = state.min === "" ? null : Number(state.min), max = state.max === "" ? null : Number(state.max);
    if (state.type === "number") {
      await control.fill("");
      await control.dispatchEvent("change");
    }
    if (min !== null) {
      await control.fill(String(min));
      expect(await control.evaluate(element => element.validity.valid), `${key} minimum should be accepted`).toBe(true);
    }
    if (max !== null) {
      await control.fill(String(max));
      expect(await control.evaluate(element => element.validity.valid), `${key} maximum should be accepted`).toBe(true);
    }
    const current = Number(state.value || 0);
    const target = min !== null && current !== min ? min : max !== null && current !== max ? max : current + 1;
    await control.fill(String(target));
    await control.dispatchEvent("change");
    expect(await control.evaluate(element => element.validity.valid), `${key} should accept its selected numeric value`).toBe(true);
  } else if (state.tag === "textarea") {
    await control.fill("Nightly control matrix content");
  } else {
    await control.fill(`${state.value || ""} nightly`);
  }
  return control;
}

test("nightly assistant configuration control matrix edits, saves, and reloads every ordinary field", async ({page}) => {
  const errors = trackPageErrors(page);
  const coverage = {};
  for (const route of routes) {
    await page.goto(fixtureUrl(route));
    const panel = panelFor(page);
    await expect(panel.locator("main")).toBeVisible();
    const initial = await controlValues(panel);
    expect(initial.length, `${route} must expose ordinary configuration controls`).toBeGreaterThan(0);
    const keys = [...new Set(initial.map(item => item.key))].sort((a, b) => {
      const priority = key => ({api_mode:0, memory_retrieval_mode:0, web_search:0}[key] ?? 1);
      const byFeature = priority(a) - priority(b);
      if (byFeature) return byFeature;
      const checkbox = key => initial.find(item => item.key === key)?.kind === "checkbox" ? 1 : 0;
      return checkbox(a) - checkbox(b);
    });
    const changed = [];
    for (const key of keys) {
      const control = await changeControl(panel, key);
      if (control) changed.push(key);
    }
    for (const {key} of await controlValues(panel)) {
      if (keys.includes(key) || changed.includes(key)) continue;
      const control = await changeControl(panel, key);
      if (control) changed.push(key);
    }
    expect(changed.length, `${route} should exercise its enabled controls through the UI`).toBeGreaterThan(0);
    await expect(panel.locator("#save-config")).toBeVisible();
    const visibleBeforeSave = await controlValues(panel);
    await panel.locator("#save-config").click();
    await expect(panel.locator("#save-config")).toHaveCount(0);
    const saved = await panel.evaluate(host => ({
      config:structuredClone(host._configData?.config || host._result?.config || {}),
      title:host._configData?.title,
      draftTitle:host._draftTitle,
    }));
    coverage[route] = {keys, changed, dependentDisabled:keys.filter(key => !changed.includes(key)), visibleBeforeSave};

    await page.goto(fixtureUrl(route));
    const reloaded = await controlValues(panelFor(page));
    for (const field of visibleBeforeSave) {
      if (!changed.includes(field.key)) continue;
      const actual = reloaded.find(item => item.key === field.key);
      expect(actual, `${route}/${field.key} should remain present after fresh load`).toBeTruthy();
      if (field.key === "__title") expect(actual.value).toBe(saved.title);
      else if (Object.hasOwn(saved.config, field.key)) {
        const expected = saved.config[field.key];
        if (typeof field.value === "boolean") expect(actual.value).toBe(Boolean(expected));
        else if (Array.isArray(expected)) expect(actual.value).toBe(expected.join(", "));
        else expect(String(actual.value), `${route}/${field.key}: saved=${JSON.stringify(expected)} loaded=${JSON.stringify(actual.value)}`).toBe(String(expected ?? ""));
      }
    }
  }
  expect(Object.keys(coverage)).toEqual(routes);
  await expectHarnessClean(page, errors);
});

test("nightly configuration validation maps backend field errors onto the matching control", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("assistant/basics"));
  const panel = panelFor(page);
  await panel.locator('[data-config="max_tokens"]').fill("1300");
  await panel.evaluate(host => {
    const original = host._hass.callWS.bind(host._hass);
    host._hass.callWS = async message => {
      if (message.section === "configuration" && message.action === "save") {
        return {...host._configData, valid:false, errors:{max_tokens:"Nightly validation: token limit rejected."}};
      }
      return original(message);
    };
  });
  await panel.locator("#save-config").click();
  await expect(panel.locator('[data-error="max_tokens"]')).toHaveText("Nightly validation: token limit rejected.");
  await expect(panel.locator("#toast")).toContainText("Fix the highlighted configuration errors");
  await expect(panel.locator("#save-config")).toBeVisible();
  await expectHarnessClean(page, errors);
});

test("nightly configuration parent controls reveal and enable their dependent settings", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = panelFor(page);
  await page.goto(fixtureUrl("capabilities/web-skills"));
  await expect(panel.locator('[data-config="web_search_context"]')).toBeDisabled();
  await panel.locator('[data-config="web_search"]').check();
  await expect(panel.locator('[data-config="web_search_context"]')).toBeEnabled();
  await panel.locator("#save-config").click();
  await expect(panel.locator("#save-config")).toHaveCount(0);

  await page.goto(fixtureUrl("data-memory/memory-settings"));
  await expect(panel.locator('[data-memory-config="memory_embedding_model"]')).toBeDisabled();
  await panel.locator('[data-memory-config="memory_retrieval_mode"]').selectOption("hybrid");
  await expect(panel.locator('[data-memory-config="memory_embedding_model"]')).toBeEnabled();
  await panel.locator("#save-config").click();
  await expect(panel.locator("#save-config")).toHaveCount(0);

  await page.goto(fixtureUrl("assistant/speech"));
  await expect(panel.locator('[data-config="speech_strip_markdown"]')).toBeDisabled();
  await panel.locator('[data-config="speech_processing_enabled"]').check();
  await expect(panel.locator('[data-config="speech_strip_markdown"]')).toBeEnabled();
  await expect(panel.locator('[data-config="speech_strip_urls"]')).toBeEnabled();
  await panel.locator("#save-config").click();
  await expect(panel.locator("#save-config")).toHaveCount(0);

  await page.goto(fixtureUrl("assistant/voice"));
  await expect(panel.locator('[data-config="voice_unmapped_policy"]')).toBeDisabled();
  await panel.locator('[data-config="voice_scope_policy"]').selectOption("device_mapping");
  await expect(panel.locator('[data-config="voice_unmapped_policy"]')).toBeEnabled();
  await expect(panel.locator("#add-voice-mapping")).toBeVisible();
  await panel.locator("#add-voice-mapping").click();
  await expect(panel.locator("[data-voice-mapping-row]")).toHaveCount(1);
  await expectHarnessClean(page, errors);
});
