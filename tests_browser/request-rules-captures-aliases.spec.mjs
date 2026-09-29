import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

async function setActions(panel, value) {
  const selector = panel.locator("#rule-action-sequence-host ha-selector");
  await selector.evaluate((node, next) => {
    node.value = next;
    node.dispatchEvent(new CustomEvent("value-changed", {
      detail: {value: next},
      bubbles: true,
      composed: true,
    }));
  }, value);
}

test("captured AI input stays valid only while every sentence pattern provides the selected slot", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  let panel = page.locator("extended-openai-management-panel");

  await panel.getByRole("button", {name:"Create rule", exact:true}).first().click();
  await panel.locator("#rule-name").fill("Captured question");
  await panel.locator("#rule-phrases").fill("deep think {question}\ncarefully answer {question}");
  await panel.locator("#rule-match").selectOption("sentence_pattern");
  await panel.locator("#rule-action-type").selectOption("model_routing");
  await panel.locator("#rule-model").fill("gpt-5-mini");
  await panel.locator("#rule-continue-to-ai").check();
  await panel.locator("#rule-ai-input-mode").selectOption("capture");
  await expect(panel.locator("#rule-ai-input-capture")).toHaveValue("question");
  await expect(panel.locator("#rule-ai-input-capture")).toHaveAttribute("aria-invalid", "false");
  await panel.locator("#rule-save").click();

  let card = panel.locator(".request-rule-card").filter({hasText:"Captured question"});
  await card.locator(".rule-edit").click();
  await expect(panel.locator("#rule-ai-input-mode")).toHaveValue("capture");
  await expect(panel.locator("#rule-ai-input-capture")).toHaveValue("question");

  await panel.locator("#rule-phrases").fill("deep think {question}\ncarefully answer {topic}");
  await expect(panel.locator("#rule-ai-input-capture")).toHaveValue("question");
  await expect(panel.locator("#rule-ai-input-capture")).toHaveAttribute("aria-invalid", "true");
  await expect(panel.locator("#rule-ai-input-help")).toContainText("No captured values are available for every trigger");
  await panel.locator("#rule-save").click();
  await expect(panel.locator("#rule-dialog")).toHaveJSProperty("open", true);
  await expect(panel.locator("#rule-error")).toContainText("No captured values are available for every trigger");

  await panel.locator("#rule-ai-input-mode").selectOption("original");
  await expect(panel.locator("#rule-ai-capture-label")).toBeHidden();
  await panel.locator("#rule-save").click();
  await expect(panel.locator("#rule-dialog")).not.toHaveJSProperty("open", true);

  await page.goto(fixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  card = panel.locator(".request-rule-card").filter({hasText:"Captured question"});
  await card.locator(".rule-edit").click();
  await expect(panel.locator("#rule-ai-input-mode")).toHaveValue("original");
  await expect(panel.locator("#rule-phrases")).toHaveValue("deep think {question}\ncarefully answer {topic}");

  await panel.locator("#rule-phrases").fill("deep think {question}\ncarefully answer {question}");
  await panel.locator("#rule-ai-input-mode").selectOption("capture");
  await expect(panel.locator("#rule-ai-input-capture")).toHaveValue("question");
  await expect(panel.locator("#rule-ai-input-capture")).toHaveAttribute("aria-invalid", "false");
  await panel.locator("#rule-save").click();

  await page.goto(fixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  await panel.locator(".request-rule-card").filter({hasText:"Captured question"}).locator(".rule-edit").click();
  await expect(panel.locator("#rule-ai-input-mode")).toHaveValue("capture");
  await expect(panel.locator("#rule-ai-input-capture")).toHaveValue("question");
  await expectHarnessClean(page, errors);
});

test("Function result aliases round-trip and rename response references by stable step id", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  let panel = page.locator("extended-openai-management-panel");

  await panel.getByRole("button", {name:"Create rule", exact:true}).first().click();
  await panel.locator("#rule-name").fill("Aliased Function result");
  await panel.locator("#rule-phrases").fill("check aliased result");

  const actions = [
    {
      action:"extended_openai_conversation_responses.call_function",
      data:{function:"baseline_tool", arguments:{}, result_alias:"battery", step_id:"battery-step"},
    },
    {
      action:"light.turn_on",
      target:{entity_id:"light.alias_probe"},
      data:{note:"{battery.level}"},
    },
  ];
  await setActions(panel, actions);
  await expect(panel.locator(".rule-result-alias")).toHaveCount(1);
  await expect(panel.locator(".rule-result-alias")).toHaveValue("battery");
  await panel.locator("#rule-success").fill("Battery {battery.level}");
  await panel.locator("#rule-save").click();

  let card = panel.locator(".request-rule-card").filter({hasText:"Aliased Function result"});
  await card.locator(".rule-edit").click();
  await expect(panel.locator(".rule-result-alias")).toHaveValue("battery");
  await expect(panel.locator("#rule-success")).toHaveValue("Battery {battery.level}");

  await panel.locator(".rule-result-alias").fill("power");
  await panel.locator(".rule-result-alias").press("Tab");
  await panel.locator("#rule-save").click();

  card = panel.locator(".request-rule-card").filter({hasText:"Aliased Function result"});
  await card.locator(".rule-edit").click();
  await expect(panel.locator(".rule-result-alias")).toHaveValue("power");
  await expect(panel.locator("#rule-success")).toHaveValue("Battery {power.level}");
  const renamedActions = await panel.locator("#rule-action-sequence-host ha-selector").evaluate((selector) => selector.value);
  expect(renamedActions).toEqual([
    {
      action:"extended_openai_conversation_responses.call_function",
      data:{function:"baseline_tool", arguments:{}, result_alias:"power", step_id:"battery-step"},
    },
    {
      action:"light.turn_on",
      target:{entity_id:"light.alias_probe"},
      data:{note:"{power.level}"},
    },
  ]);
  await panel.locator(".rule-close").first().click();

  await page.goto(fixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  await panel.locator(".request-rule-card").filter({hasText:"Aliased Function result"}).locator(".rule-edit").click();
  await expect(panel.locator(".rule-result-alias")).toHaveValue("power");
  await expect(panel.locator("#rule-success")).toHaveValue("Battery {power.level}");
  await expectHarnessClean(page, errors);
});

test("multiple Function calls expose independent alias fields and preserve their step identity", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = page.locator("extended-openai-management-panel");

  await panel.getByRole("button", {name:"Create rule", exact:true}).first().click();
  await panel.locator("#rule-name").fill("Two Function aliases");
  await panel.locator("#rule-phrases").fill("two aliased calls");
  const actions = [
    {
      action:"extended_openai_conversation_responses.call_function",
      data:{function:"baseline_tool", arguments:{}, step_id:"first-step"},
    },
    {
      action:"extended_openai_conversation_responses.call_function",
      data:{function:"baseline_tool", arguments:{}, step_id:"second-step"},
    },
  ];
  await setActions(panel, actions);
  const aliases = panel.locator(".rule-result-alias");
  await expect(aliases).toHaveCount(2);
  await expect(aliases.nth(0)).toHaveAttribute("placeholder", "baseline_tool");
  await expect(aliases.nth(1)).toHaveAttribute("placeholder", "baseline_tool_2");

  await aliases.nth(0).fill("first_result");
  await aliases.nth(0).press("Tab");
  await aliases.nth(1).fill("second_result");
  await aliases.nth(1).press("Tab");
  await panel.locator("#rule-success").fill("{first_result.value} / {second_result.value}");
  await panel.locator("#rule-save").click();

  const card = panel.locator(".request-rule-card").filter({hasText:"Two Function aliases"});
  await card.locator(".rule-edit").click();
  await expect(panel.locator(".rule-result-alias").nth(0)).toHaveValue("first_result");
  await expect(panel.locator(".rule-result-alias").nth(1)).toHaveValue("second_result");
  const persisted = await panel.locator("#rule-action-sequence-host ha-selector").evaluate((selector) => selector.value);
  expect(persisted.map((step) => step.data.step_id)).toEqual(["first-step", "second-step"]);
  await expect(panel.locator("#rule-success")).toHaveValue("{first_result.value} / {second_result.value}");
  await expectHarnessClean(page, errors);
});
