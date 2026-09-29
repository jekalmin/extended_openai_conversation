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

test("nightly capture and result-alias transitions remain stable across repeated edits", async ({page}, testInfo) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  let panel = page.locator("extended-openai-management-panel");

  await panel.getByRole("button", {name:"Create rule", exact:true}).first().click();
  await panel.locator("#rule-name").fill("Nightly capture alias");
  await panel.locator("#rule-phrases").fill("ask {question}\nanswer {question}");
  await panel.locator("#rule-match").selectOption("sentence_pattern");
  await panel.locator("#rule-action-type").selectOption("model_routing");
  await panel.locator("#rule-model").fill("gpt-5-mini");
  await panel.locator("#rule-continue-to-ai").check();
  await panel.locator("#rule-ai-input-mode").selectOption("capture");
  await expect(panel.locator("#rule-ai-input-capture")).toHaveValue("question");
  await panel.locator("#rule-save").click();

  for (let index = 0; index < 3; index += 1) {
    const card = panel.locator(".request-rule-card").filter({hasText:"Nightly capture alias"});
    await card.locator(".rule-edit").click();
    await panel.locator("#rule-phrases").fill(`ask {question}\nanswer {${index % 2 ? "question" : "topic"}}`);
    if (index % 2 === 0) {
      await expect(panel.locator("#rule-ai-input-capture")).toHaveAttribute("aria-invalid", "true");
      await panel.locator("#rule-ai-input-mode").selectOption("original");
    } else {
      await panel.locator("#rule-ai-input-mode").selectOption("capture");
      await expect(panel.locator("#rule-ai-input-capture")).toHaveValue("question");
      await expect(panel.locator("#rule-ai-input-capture")).toHaveAttribute("aria-invalid", "false");
    }
    await panel.locator("#rule-save").click();
  }

  await panel.getByRole("button", {name:"Create rule", exact:true}).first().click();
  await panel.locator("#rule-name").fill("Nightly result alias");
  await panel.locator("#rule-phrases").fill("nightly alias result");
  await setActions(panel, [{
    action:"extended_openai_conversation_responses.call_function",
    data:{function:"baseline_tool", arguments:{}, result_alias:"result_0", step_id:"stable-step"},
  }]);
  await panel.locator("#rule-success").fill("Value {result_0.value}");
  await panel.locator("#rule-save").click();

  for (let index = 1; index <= 3; index += 1) {
    const card = panel.locator(".request-rule-card").filter({hasText:"Nightly result alias"});
    await card.locator(".rule-edit").click();
    const alias = `result_${index}`;
    await panel.locator(".rule-result-alias").fill(alias);
    await panel.locator(".rule-result-alias").press("Tab");
    await panel.locator("#rule-save").click();
    await card.locator(".rule-edit").click();
    await expect(panel.locator(".rule-result-alias")).toHaveValue(alias);
    await expect(panel.locator("#rule-success")).toHaveValue(`Value {${alias}.value}`);
    const actions = await panel.locator("#rule-action-sequence-host ha-selector").evaluate((selector) => selector.value);
    expect(actions[0].data.step_id).toBe("stable-step");
    await panel.locator(".rule-close").first().click();
  }

  await page.goto(fixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  await panel.locator(".request-rule-card").filter({hasText:"Nightly result alias"}).locator(".rule-edit").click();
  await expect(panel.locator(".rule-result-alias")).toHaveValue("result_3");
  await expect(panel.locator("#rule-success")).toHaveValue("Value {result_3.value}");

  await testInfo.attach("request-rule-capture-alias-stress", {
    body:JSON.stringify({captureTransitions:3, aliasRenames:3, freshLoadVerified:true}, null, 2),
    contentType:"application/json",
  });
  console.log("ENHANCED REQUEST_RULE_CAPTURE_ALIAS capture_transitions=3 alias_renames=3");
  await expectHarnessClean(page, errors);
});
