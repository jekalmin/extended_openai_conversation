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

async function setConditions(panel, value) {
  await panel.locator("#rule-add-conditions").click();
  const selector = panel.locator("#rule-condition-host ha-selector");
  await selector.evaluate((node, next) => {
    node.value = next;
    node.dispatchEvent(new CustomEvent("value-changed", {
      detail: {value: next},
      bubbles: true,
      composed: true,
    }));
  }, value);
}

async function createLocalRule(panel, {name, phrase, action, success, continueMatching = false, continueAi = false, conditions = null}) {
  await panel.getByRole("button", {name:"Create rule", exact:true}).first().click();
  await panel.locator("#rule-name").fill(name);
  await panel.locator("#rule-phrases").fill(phrase);
  await panel.locator("#rule-match").selectOption("equals");
  await setActions(panel, action);
  await panel.locator("#rule-success").fill(success);
  if (continueAi) await panel.locator("#rule-local-continue-to-ai").check();
  if (conditions) await setConditions(panel, conditions);
  if (continueMatching) {
    const advanced = panel.locator("#rule-advanced");
    if (!(await advanced.evaluate(node => node.open))) await advanced.locator("summary").click();
    await panel.locator("#rule-continue-matching").check();
  }
  await panel.locator("#rule-save").click();
  await expect(panel.locator(".request-rule-card").filter({hasText:name})).toBeVisible();
}

test("Request Rules UI authors and reloads a three-rule Continue Matching chain", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  let panel = page.locator("extended-openai-management-panel");

  await createLocalRule(panel, {
    name:"Chain first",
    phrase:"run chain",
    action:[{action:"light.turn_on", target:{entity_id:"light.chain_first"}}],
    success:"First complete",
    continueMatching:true,
  });
  await createLocalRule(panel, {
    name:"Chain conditional",
    phrase:"run chain",
    action:[{action:"light.turn_on", target:{entity_id:"light.chain_conditional"}}],
    success:"Conditional complete",
    continueMatching:true,
    conditions:[{condition:"state", entity_id:"input_boolean.chain_allowed", state:"on"}],
  });
  await createLocalRule(panel, {
    name:"Chain final",
    phrase:"run chain",
    action:[{action:"light.turn_on", target:{entity_id:"light.chain_final"}}],
    success:"Final complete",
  });

  let cards = panel.locator(".request-rule-card");
  await expect(cards.locator("h2")).toContainText(["Baseline rule", "Chain first", "Chain conditional", "Chain final"]);

  for (const [name, expected] of [["Chain first", true], ["Chain conditional", true], ["Chain final", false]]) {
    const card = panel.locator(".request-rule-card").filter({hasText:name});
    await card.locator(".rule-edit").click();
    await expect(panel.locator("#rule-continue-matching")).toBeChecked({checked:expected});
    await expect(panel.locator("#rule-local-continue-to-ai")).not.toBeChecked();
    await panel.locator(".rule-close").first().click();
  }

  await panel.locator(".request-rule-card").filter({hasText:"Chain conditional"}).locator(".rule-edit").click();
  await expect(panel.locator("#rule-conditions-body")).toBeVisible();
  expect(await panel.locator("#rule-condition-host ha-selector").evaluate((selector) => selector.value)).toEqual([
    {condition:"state", entity_id:"input_boolean.chain_allowed", state:"on"},
  ]);
  await panel.locator(".rule-close").first().click();

  await page.goto(fixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  cards = panel.locator(".request-rule-card");
  await expect(cards.locator("h2")).toContainText(["Baseline rule", "Chain first", "Chain conditional", "Chain final"]);
  await panel.locator(".request-rule-card").filter({hasText:"Chain first"}).locator(".rule-edit").click();
  await expect(panel.locator("#rule-continue-matching")).toBeChecked();
  await panel.locator(".rule-close").first().click();
  await panel.locator(".request-rule-card").filter({hasText:"Chain final"}).locator(".rule-edit").click();
  await expect(panel.locator("#rule-continue-matching")).not.toBeChecked();
  await panel.locator(".rule-close").first().click();

  const stored = await page.evaluate(() => browserHarness.getState().requestRules.rules
    .filter((rule) => rule.name.startsWith("Chain "))
    .map((rule) => ({
      name:rule.name,
      order:rule.order,
      continue_matching:Boolean(rule.continue_matching),
      conditions:rule.conditions || [],
    })));
  expect(stored.map((rule) => rule.name)).toEqual(["Chain first", "Chain conditional", "Chain final"]);
  expect(stored.map((rule) => rule.continue_matching)).toEqual([true, true, false]);
  expect(stored[1].conditions).toEqual([{condition:"state", entity_id:"input_boolean.chain_allowed", state:"on"}]);

  await expectHarnessClean(page, errors);
});
