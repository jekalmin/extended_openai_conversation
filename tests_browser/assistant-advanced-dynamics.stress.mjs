import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

test("nightly model capability transitions and dedicated reset stay coherent", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("assistant/model-responses"));
  const panel = page.locator("extended-openai-management-panel");

  await panel.evaluate((host) => {
    const capabilities = {
      supports_temperature: true,
      supports_top_p: true,
      supports_reasoning_effort: true,
      supports_service_tier: true,
      limits: {max_output_tokens: 4096},
      reasoning: {
        supported: true,
        efforts: ["low", "high"],
        by_api: {
          responses: {efforts: ["low", "high"]},
          chat_completions: {efforts: ["low"]},
        },
      },
      temperature: {support: "conditional", allowed_reasoning_efforts: ["low"]},
      top_p: {support: "always"},
      api: {responses: true, chat_completions: false},
      evaluations: {
        responses: {
          low: {reasoning: true, function: true, web_search: true},
          high: {reasoning: true, function: true, web_search: true},
        },
      },
      recommended_profile: {reasoning_effort: "low"},
    };
    Object.assign(host._draft, {
      chat_model: "dynamic-model",
      api_mode: "responses",
      temperature: 0.7,
      top_p: 0.8,
      reasoning_effort: "high",
      service_tier: "flex",
      shorten_tool_call_id: true,
    });
    Object.assign(host._result.defaults, {
      temperature: 0.2,
      top_p: 0.9,
      reasoning_effort: "low",
      service_tier: "default",
      shorten_tool_call_id: false,
      memory_auto_retrieve_limit: 0,
      memory_retrieval_mode: "lexical",
      memory_embedding_model: "text-embedding-3-small",
    });
    host._configData.options = {
      ...host._configData.options,
      api_mode: [
        {value: "auto", label: "Auto"},
        {value: "responses", label: "Responses"},
        {value: "chat_completions", label: "Chat Completions"},
      ],
      reasoning_effort: [
        {value: "low", label: "Low"},
        {value: "high", label: "High"},
      ],
      service_tier: [
        {value: "default", label: "Default"},
        {value: "flex", label: "Flex"},
      ],
    };
    host._result.options = host._configData.options;
    host._result.model_capabilities = capabilities;
    host._modelCatalogData = {
      requested_model: "dynamic-model",
      model_capabilities: capabilities,
      catalog_models: [{id: "dynamic-model", display_name: "Dynamic Model", status: "current"}],
    };
    host._render();
  });

  await expect(panel.locator("#config-temperature")).toBeVisible();
  await expect(panel.locator("#config-temperature")).toBeDisabled();
  await expect(panel.locator('[data-field="temperature"] .capability-note')).toContainText("inactive");
  await expect(panel.locator("#config-top_p")).toBeEnabled();
  await expect(panel.locator('#config-api_mode option[value="chat_completions"]')).toBeDisabled();
  await expect(panel.locator("#config-max_tokens")).toHaveAttribute("max", "4096");

  await panel.locator("#config-reasoning_effort").selectOption("low");
  await panel.evaluate((host) => host._render());
  await expect(panel.locator("#config-temperature")).toBeEnabled();

  await panel.locator("#reset-model-parameters").click();
  await expect(panel.locator("#config-temperature")).toHaveValue("0.2");
  await expect(panel.locator("#config-top_p")).toHaveValue("0.9");
  await expect(panel.locator("#config-reasoning_effort")).toHaveValue("low");
  await expect(panel.locator("#config-service_tier")).toHaveValue("default");
  await expect(panel.locator("#config-shorten_tool_call_id")).not.toBeChecked();
  await expect(panel.getByText("Unsaved changes", {exact: true})).toBeVisible();
  await expectHarnessClean(page, errors);
});

test("nightly request preview ignores an older completion after a newer preview", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("assistant/prompt-context"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator("#prompt-editor")).toBeVisible();

  await panel.evaluate((host) => {
    const original = host._call.bind(host);
    window.requestPreviewResolvers = [];
    host._call = (section, action, payload) => {
      if (section === "configuration" && action === "request_preview") {
        return new Promise((resolve) => requestPreviewResolvers.push({resolve, payload}));
      }
      return original(section, action, payload);
    };
  });

  await panel.locator("#prompt-editor").fill("First preview prompt");
  await panel.locator("#preview-request").click();
  await expect.poll(() => page.evaluate(() => requestPreviewResolvers.length)).toBe(1);
  await panel.locator("#prompt-preview-close").click();

  await panel.locator("#prompt-editor").fill("Second preview prompt");
  await panel.locator("#preview-request").click();
  await expect.poll(() => page.evaluate(() => requestPreviewResolvers.length)).toBe(2);

  await page.evaluate(() => requestPreviewResolvers[1].resolve({
    total_character_count: 12,
    function_group_savings: {characters: 3, percent: 20},
    sections: [{label: "System", content: "new preview", character_count: 11}],
    notes: ["new note"],
  }));
  await expect(panel.locator(".request-preview-output")).toHaveValue("new preview");

  await page.evaluate(() => requestPreviewResolvers[0].resolve({
    total_character_count: 999,
    function_group_savings: {characters: 0, percent: 0},
    sections: [{label: "System", content: "stale preview", character_count: 13}],
    notes: ["stale note"],
  }));
  await page.waitForTimeout(50);
  await expect(panel.locator(".request-preview-output")).toHaveValue("new preview");
  expect(await panel.evaluate((host) => host._effectiveRequestPreview.sections[0].content)).toBe("new preview");

  await panel.locator("#prompt-preview-close").click();
  await panel.locator("#reset-prompt").click();
  await expect(panel.locator("#prompt-editor")).toHaveValue("");
  await expect(panel.locator('[data-config="current_datetime_enabled"]')).toBeChecked();
  await expect(panel.locator('[data-config="exposed_entities_enabled"]')).toBeChecked();
  await expectHarnessClean(page, errors);
});

test("nightly exposed attribute preferences survive context disable and save", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("assistant/prompt-context"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator("#prompt-editor")).toBeVisible();
  await expect.poll(() => panel.evaluate((host) => Boolean(host._result?.exposed_attribute_catalog))).toBe(true);

  await panel.evaluate((host) => {
    host._draft.exposed_entities_enabled = true;
    host._draft.exposed_entity_attributes = {};
    host._result.exposed_attribute_catalog = {
      entities: [{
        entity_id: "light.kitchen",
        reference: "registry:kitchen-light",
        name: "Kitchen light",
        attributes: ["brightness", "color_temp_kelvin"],
        selected_attributes: [],
        durable_selection_available: true,
      }],
      saved_unexposed: [],
    };
    host._render();
  });

  const fallback = panel.locator("#exposed-entity-picker-fallback");
  await expect(fallback).toBeVisible();
  await fallback.selectOption("light.kitchen");
  await expect(panel.locator('[data-exposed-attribute][data-attribute="brightness"]')).toBeVisible();
  await panel.locator('[data-exposed-attribute][data-attribute="brightness"]').check();
  expect(await panel.evaluate((host) => host._draft.exposed_entity_attributes)).toEqual({
    "registry:kitchen-light": ["brightness"],
  });

  await panel.locator('[data-config="exposed_entities_enabled"]').uncheck();
  await expect(panel.locator(".exposed-inactive-notice")).toBeVisible();
  expect(await panel.evaluate((host) => host._draft.exposed_entity_attributes)).toEqual({
    "registry:kitchen-light": ["brightness"],
  });

  await panel.getByRole("button", {name: "Save changes", exact: true}).click();
  await expect.poll(() => page.evaluate(() => browserHarness.getState().configuration.config.exposed_entity_attributes)).toEqual({
    "registry:kitchen-light": ["brightness"],
  });
  await expectHarnessClean(page, errors);
});

test("nightly voice mappings follow policy dependencies and persist", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("assistant/voice"));
  let panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator("#voice-current-summary")).toBeVisible();

  await panel.evaluate((host) => {
    host._configData.options = {
      ...host._configData.options,
      voice_scope_policy: ["unretained", "shared", "default_user", "device_mapping"].map((value) => ({value, label:value})),
      voice_unmapped_policy: ["unretained", "shared", "default_user"].map((value) => ({value, label:value})),
    };
    host._result.options = host._configData.options;
    Object.assign(host._draft, {
      voice_scope_policy: "unretained",
      voice_unmapped_policy: "unretained",
      voice_default_user_id: "",
      voice_device_mappings: {},
    });
    host.__voiceEntityRegistry = [{
      entity_id: "assist_satellite.kitchen",
      device_id: "device-kitchen",
    }];
    host._render();
  });

  await panel.locator('[data-config="voice_scope_policy"]').selectOption("device_mapping");
  await expect(panel.locator("[data-voice-fallback-card]")).not.toHaveClass(/is-disabled/);
  await expect(panel.locator("[data-voice-mappings-card]")).not.toHaveClass(/is-disabled/);
  await panel.locator("#add-voice-mapping").click();
  const row = panel.locator("[data-voice-mapping-row]").last();
  await row.locator(".voice-satellite-picker").evaluate((picker) => {
    picker.value = "assist_satellite.kitchen";
    picker.dispatchEvent(new CustomEvent("value-changed", {
      detail:{value:"assist_satellite.kitchen"},
      bubbles:true,
      composed:true,
    }));
  });
  await expect(row.locator(".voice-device-id")).toHaveValue("device-kitchen");
  await row.locator(".voice-owner-type").selectOption("shared");
  expect(await panel.evaluate((host) => host._draft.voice_device_mappings)).toEqual({
    "device-kitchen": "shared:household",
  });

  await panel.getByRole("button", {name: "Save changes", exact: true}).click();
  await page.goto(fixtureUrl("assistant/voice"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator("[data-voice-mapping-row]")).toHaveCount(1);
  await expect(panel.locator(".voice-device-id")).toHaveValue("device-kitchen");
  await expect(panel.locator(".voice-owner-type")).toHaveValue("shared");
  await expectHarnessClean(page, errors);
});

test("nightly speech regex lifecycle and preview ordering remain stable", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("assistant/speech"));
  let panel = page.locator("extended-openai-management-panel");

  await panel.locator('[data-config="speech_processing_enabled"]').check();
  await panel.locator("#add-regex").click();
  await panel.locator("#add-regex").click();
  let rows = panel.locator("#regex-rules .rule-row");
  await expect(rows).toHaveCount(2);
  await rows.nth(0).locator(".regex-pattern").fill("ETA");
  await rows.nth(0).locator(".regex-replacement").fill("estimated time");
  await rows.nth(1).locator(".regex-pattern").fill("URL");
  await rows.nth(1).locator(".regex-replacement").fill("link");
  await rows.nth(1).locator('.move-regex[data-direction="-1"]').click();
  rows = panel.locator("#regex-rules .rule-row");
  await expect(rows.nth(0).locator(".regex-pattern")).toHaveValue("URL");
  await rows.nth(1).locator(".delete-regex").click();
  await expect(rows).toHaveCount(1);

  await panel.evaluate((host) => {
    const original = host._call.bind(host);
    window.speechPreviewResolvers = [];
    host._call = (section, action, payload) => {
      if (section === "configuration" && action === "speech_preview") {
        return new Promise((resolve) => speechPreviewResolvers.push({resolve, payload}));
      }
      return original(section, action, payload);
    };
  });
  await panel.locator("#speech-sample").fill("first sample");
  await panel.locator("#preview-speech").click();
  await expect.poll(() => page.evaluate(() => speechPreviewResolvers.length)).toBe(1);
  await panel.locator("#speech-sample").fill("second sample");
  await panel.locator("#preview-speech").click();
  await expect.poll(() => page.evaluate(() => speechPreviewResolvers.length)).toBe(2);

  await page.evaluate(() => speechPreviewResolvers[1].resolve({speech_text:"new speech"}));
  await expect(panel.locator("#speech-output")).toHaveValue("new speech");
  await page.evaluate(() => speechPreviewResolvers[0].resolve({speech_text:"stale speech"}));
  await page.waitForTimeout(50);
  await expect(panel.locator("#speech-output")).toHaveValue("new speech");

  await panel.getByRole("button", {name: "Save changes", exact: true}).click();
  await page.goto(fixtureUrl("assistant/speech"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator('[data-config="speech_processing_enabled"]')).toBeChecked();
  await expect(panel.locator("#regex-rules .rule-row")).toHaveCount(1);
  await expect(panel.locator("#regex-rules .regex-pattern")).toHaveValue("URL");
  await expect(panel.locator("#regex-rules .regex-replacement")).toHaveValue("link");
  await expectHarnessClean(page, errors);
});
