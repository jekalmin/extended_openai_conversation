import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

const panelFor = (page) => page.locator("extended-openai-management-panel");

async function createGroup(panel, name) {
  await panel.locator("#rule-groups-manage").click();
  await expect(panel.locator("#rule-groups-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#rule-new-group-name").fill(name);
  await panel.locator("#rule-group-add").click();
  const id = await panel.evaluate((host, label) => host._result.groups.find((group) => group.name === label)?.id || null, name);
  expect(id).toBeTruthy();
  await panel.locator("#rule-groups-done").click();
  return id;
}

async function setHaSelector(locator, value) {
  await locator.evaluate((selector, next) => {
    selector.value = next;
    selector.dispatchEvent(new CustomEvent("value-changed", {
      detail: {value: next},
      bubbles: true,
      composed: true,
    }));
  }, value);
}

async function openRule(panel, name) {
  const card = panel.locator(".request-rule-card").filter({hasText:name});
  await expect(card).toBeVisible();
  await card.locator(".rule-edit").click();
  await expect(panel.locator("#rule-dialog")).toHaveJSProperty("open", true);
  return card;
}

async function assertLocalRuleEditor(panel, {groupId, conditions, actions}) {
  await expect(panel.locator("#rule-name")).toHaveValue("Round-trip local");
  await expect(panel.locator("#rule-phrases")).toHaveValue("local round trip\nlocal alternate");
  await expect(panel.locator("#rule-match")).toHaveValue("contains");
  await expect(panel.locator("#rule-action-type")).toHaveValue("local_action");
  await expect(panel.locator("#rule-group")).toHaveValue(groupId);
  await expect(panel.locator("#rule-local-continue-to-ai")).not.toBeChecked();
  await expect(panel.locator("#rule-success")).toHaveValue("Local success");
  await expect(panel.locator("#rule-failure")).toHaveValue("Local failure");
  await expect(panel.locator("#rule-ai-input-mode")).toHaveValue("original");
  await expect(panel.locator("#rule-continue-matching")).toBeChecked();
  await expect(panel.locator("#rule-matching-behavior")).toHaveValue("custom");
  await expect(panel.locator("#rule-word-forms")).not.toBeChecked();
  await expect(panel.locator("#rule-wording")).not.toBeChecked();
  await expect(panel.locator("#rule-fuzzy")).toBeChecked();
  await expect(panel.locator("#rule-threshold")).toHaveValue("83");
  await expect(panel.locator("#rule-conditions-body")).toBeVisible();
  expect(await panel.locator("#rule-condition-host ha-selector").evaluate((selector) => selector.value)).toEqual(conditions);
  expect(await panel.locator("#rule-action-sequence-host ha-selector").evaluate((selector) => selector.value)).toEqual(actions);
}

async function assertRoutingRuleEditor(panel, {groupId}) {
  await expect(panel.locator("#rule-name")).toHaveValue("Round-trip routing");
  await expect(panel.locator("#rule-phrases")).toHaveValue("route round trip");
  await expect(panel.locator("#rule-match")).toHaveValue("starts_with");
  await expect(panel.locator("#rule-action-type")).toHaveValue("model_routing");
  await expect(panel.locator("#rule-group")).toHaveValue(groupId);
  await expect(panel.locator("#rule-model")).toHaveValue("gpt-5-mini");
  await expect(panel.locator("#rule-reasoning")).toHaveValue("medium");
  await expect(panel.locator("#rule-scope")).toHaveValue("request");
  await expect(panel.locator("#rule-reset")).not.toBeChecked();
  await expect(panel.locator("#rule-continue-to-ai")).toBeChecked();
  await expect(panel.locator("#rule-ai-input-mode")).toHaveValue("original");
  await expect(panel.locator("#rule-continue-matching")).not.toBeChecked();
  await expect(panel.locator("#rule-matching-behavior")).toHaveValue("defaults");
}

test("Request Rule editor round-trips local and routing forms through reopen and fresh load", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  let panel = panelFor(page);
  const groupId = await createGroup(panel, "Round-trip group");

  const actions = [{
    action: "light.turn_on",
    target: {entity_id: "light.round_trip"},
    data: {brightness_pct: 42},
  }];
  const conditions = [{
    condition: "and",
    conditions: [
      {condition: "state", entity_id: "input_boolean.round_trip_ready", state: "on"},
      {condition: "numeric_state", entity_id: "sensor.round_trip_level", above: 10},
    ],
  }];

  await panel.getByRole("button", {name:"Create rule", exact:true}).first().click();
  await panel.locator("#rule-name").fill("Round-trip local");
  await panel.locator("#rule-phrases").fill("local round trip\nlocal alternate");
  await panel.locator("#rule-match").selectOption("contains");
  await panel.locator("#rule-group").selectOption(groupId);
  await setHaSelector(panel.locator("#rule-action-sequence-host ha-selector"), actions);
  await panel.locator("#rule-success").fill("Local success");
  await panel.locator("#rule-failure").fill("Local failure");
  await panel.locator("#rule-ai-input-mode").selectOption("original");
  await panel.locator("#rule-add-conditions").click();
  await setHaSelector(panel.locator("#rule-condition-host ha-selector"), conditions);
  await panel.locator("#rule-advanced summary").click();
  await panel.locator("#rule-continue-matching").check();
  await panel.locator("#rule-matching-behavior").selectOption("custom");
  await panel.locator("#rule-word-forms").uncheck();
  await panel.locator("#rule-wording").uncheck();
  await panel.locator("#rule-fuzzy").check();
  await panel.locator("#rule-threshold").fill("83");
  await panel.locator("#rule-save").click();

  await openRule(panel, "Round-trip local");
  await assertLocalRuleEditor(panel, {groupId, conditions, actions});
  await panel.locator(".rule-close").first().click();

  await panel.getByRole("button", {name:"Create rule", exact:true}).first().click();
  await panel.locator("#rule-name").fill("Round-trip routing");
  await panel.locator("#rule-phrases").fill("route round trip");
  await panel.locator("#rule-match").selectOption("starts_with");
  await panel.locator("#rule-action-type").selectOption("model_routing");
  await panel.locator("#rule-group").selectOption(groupId);
  await panel.locator("#rule-model").fill("gpt-5-mini");
  await expect.poll(() => panel.locator("#rule-reasoning option").allTextContents()).toContain("Medium");
  await panel.locator("#rule-reasoning").selectOption("medium");
  await panel.locator("#rule-scope").selectOption("request");
  await panel.locator("#rule-continue-to-ai").check();
  await panel.locator("#rule-ai-input-mode").selectOption("original");
  await panel.locator("#rule-save").click();

  await openRule(panel, "Round-trip routing");
  await assertRoutingRuleEditor(panel, {groupId});

  // Reset is a distinct routing mode: the backend intentionally normalizes any
  // model/reasoning values away while preserving the reset/scope/handoff fields.
  await panel.locator("#rule-reset").check();
  await panel.locator("#rule-continue-to-ai").uncheck();
  await expect(panel.locator("#rule-scope")).toHaveValue("conversation");
  await panel.locator("#rule-routing-success").fill("Defaults restored");
  await panel.locator("#rule-save").click();

  await openRule(panel, "Round-trip routing");
  await expect(panel.locator("#rule-reset")).toBeChecked();
  await expect(panel.locator("#rule-continue-to-ai")).not.toBeChecked();
  await expect(panel.locator("#rule-scope")).toHaveValue("conversation");
  await expect(panel.locator("#rule-model")).toHaveValue("");
  await expect(panel.locator("#rule-reasoning")).toHaveValue("");
  await expect(panel.locator("#rule-routing-success")).toHaveValue("Defaults restored");
  await panel.locator(".rule-close").first().click();

  await page.goto(fixtureUrl("capabilities/request-rules"));
  panel = panelFor(page);
  await openRule(panel, "Round-trip local");
  await assertLocalRuleEditor(panel, {groupId, conditions, actions});
  await panel.locator(".rule-close").first().click();

  await openRule(panel, "Round-trip routing");
  await expect(panel.locator("#rule-reset")).toBeChecked();
  await expect(panel.locator("#rule-continue-to-ai")).not.toBeChecked();
  await expect(panel.locator("#rule-scope")).toHaveValue("conversation");
  await expect(panel.locator("#rule-model")).toHaveValue("");
  await expect(panel.locator("#rule-reasoning")).toHaveValue("");
  await expect(panel.locator("#rule-routing-success")).toHaveValue("Defaults restored");
  await panel.locator(".rule-close").first().click();

  await expectHarnessClean(page, errors);
});
