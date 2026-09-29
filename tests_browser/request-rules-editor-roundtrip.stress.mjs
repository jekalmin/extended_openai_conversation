import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

const MATCH_CASES = [
  {match:"equals", action:"model_routing", continueAi:false, scope:"conversation", reset:true, continueMatching:false, custom:false},
  {match:"starts_with", action:"model_routing", continueAi:true, scope:"request", reset:false, continueMatching:true, custom:true},
  {match:"ends_with", action:"local_action", continueAi:false, continueMatching:false, custom:true, conditions:true},
  {match:"contains", action:"local_action", continueAi:true, continueMatching:true, custom:true},
  {match:"sentence_pattern", action:"model_routing", continueAi:true, scope:"conversation", reset:false, continueMatching:false, custom:false},
];

const nameFor = (index, item) => `Nightly editor ${index + 1} ${item.match}`;
const phraseFor = (index, item) => item.match === "sentence_pattern"
  ? `nightly ${index + 1} set {room} mode`
  : `nightly ${index + 1} ${item.match} phrase`;

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

async function createGroup(panel) {
  await panel.locator("#rule-groups-manage").click();
  await panel.locator("#rule-new-group-name").fill("Nightly editor group");
  await panel.locator("#rule-group-add").click();
  const row = panel.locator(".rule-group-row").filter({hasText:"Nightly editor group"});
  await expect(row).toHaveCount(1);
  const groupId = await row.getAttribute("data-group-id");
  await panel.locator("#rule-groups-done").click();
  return groupId;
}

async function configureCase(panel, item, index, groupId) {
  await panel.getByRole("button", {name:"Create rule", exact:true}).first().click();
  await panel.locator("#rule-name").fill(nameFor(index, item));
  await panel.locator("#rule-phrases").fill(phraseFor(index, item));
  await panel.locator("#rule-match").selectOption(item.match);
  if (index % 2 === 0) await panel.locator("#rule-group").selectOption(groupId);
  await panel.locator("#rule-ai-input-mode").selectOption("original");

  if (item.action === "local_action") {
    const actions = [{
      action: "light.turn_on",
      target: {entity_id: `light.nightly_editor_${index + 1}`},
      data: {brightness_pct: 20 + index},
    }];
    await setHaSelector(panel.locator("#rule-action-sequence-host ha-selector"), actions);
    await panel.locator("#rule-success").fill(`Local success ${index + 1}`);
    await panel.locator("#rule-failure").fill(`Local failure ${index + 1}`);
    if (item.continueAi) await panel.locator("#rule-local-continue-to-ai").check();
    if (item.conditions) {
      const conditions = [{
        condition: "state",
        entity_id: "input_boolean.nightly_editor_ready",
        state: "on",
      }];
      await panel.locator("#rule-add-conditions").click();
      await setHaSelector(panel.locator("#rule-condition-host ha-selector"), conditions);
    }
  } else {
    await panel.locator("#rule-action-type").selectOption("model_routing");
    if (!item.reset) {
      await panel.locator("#rule-model").fill("gpt-5-mini");
      await expect.poll(() => panel.locator("#rule-reasoning option").allTextContents()).toContain("Medium");
      await panel.locator("#rule-reasoning").selectOption(index % 2 ? "medium" : "low");
    } else {
      await panel.locator("#rule-reset").check();
    }
    if (!item.continueAi) {
      await panel.locator("#rule-continue-to-ai").uncheck();
      await expect(panel.locator("#rule-scope")).toHaveValue("conversation");
      await panel.locator("#rule-routing-success").fill(`Routing acknowledgement ${index + 1}`);
    } else {
      await panel.locator("#rule-continue-to-ai").check();
      await panel.locator("#rule-scope").selectOption(item.scope);
    }
  }

  await panel.locator("#rule-advanced summary").click();
  if (item.continueMatching) await panel.locator("#rule-continue-matching").check();
  if (item.custom) {
    await panel.locator("#rule-matching-behavior").selectOption("custom");
    if (index % 2 === 0) await panel.locator("#rule-word-forms").uncheck();
    if (index % 2 !== 0) await panel.locator("#rule-wording").uncheck();
    await panel.locator("#rule-fuzzy").check();
    await panel.locator("#rule-threshold").fill(String(80 + index));
  }
  await panel.locator("#rule-save").click();
  await expect(panel.getByRole("heading", {name:nameFor(index, item), exact:true})).toBeVisible();
}

async function assertCase(panel, item, index, groupId) {
  const card = panel.locator(".request-rule-card").filter({hasText:nameFor(index, item)});
  await card.locator(".rule-edit").click();
  await expect(panel.locator("#rule-name")).toHaveValue(nameFor(index, item));
  await expect(panel.locator("#rule-phrases")).toHaveValue(phraseFor(index, item));
  await expect(panel.locator("#rule-match")).toHaveValue(item.match);
  await expect(panel.locator("#rule-action-type")).toHaveValue(item.action);
  await expect(panel.locator("#rule-ai-input-mode")).toHaveValue("original");
  await expect(panel.locator("#rule-group")).toHaveValue(index % 2 === 0 ? groupId : "");
  await expect(panel.locator("#rule-continue-matching")).toBeChecked({checked:item.continueMatching});

  if (item.action === "local_action") {
    await expect(panel.locator("#rule-local-continue-to-ai")).toBeChecked({checked:item.continueAi});
    await expect(panel.locator("#rule-success")).toHaveValue(`Local success ${index + 1}`);
    await expect(panel.locator("#rule-failure")).toHaveValue(`Local failure ${index + 1}`);
    const actions = await panel.locator("#rule-action-sequence-host ha-selector").evaluate((selector) => selector.value);
    expect(actions).toEqual([{
      action:"light.turn_on",
      target:{entity_id:`light.nightly_editor_${index + 1}`},
      data:{brightness_pct:20 + index},
    }]);
    if (item.conditions) {
      await expect(panel.locator("#rule-conditions-body")).toBeVisible();
      expect(await panel.locator("#rule-condition-host ha-selector").evaluate((selector) => selector.value)).toEqual([{
        condition:"state",
        entity_id:"input_boolean.nightly_editor_ready",
        state:"on",
      }]);
    }
  } else {
    await expect(panel.locator("#rule-reset")).toBeChecked({checked:item.reset});
    await expect(panel.locator("#rule-continue-to-ai")).toBeChecked({checked:item.continueAi});
    await expect(panel.locator("#rule-scope")).toHaveValue(item.continueAi ? item.scope : "conversation");
    if (item.reset) {
      await expect(panel.locator("#rule-model")).toHaveValue("");
      await expect(panel.locator("#rule-reasoning")).toHaveValue("");
      await expect(panel.locator("#rule-routing-success")).toHaveValue(`Routing acknowledgement ${index + 1}`);
    } else {
      await expect(panel.locator("#rule-model")).toHaveValue("gpt-5-mini");
      await expect(panel.locator("#rule-reasoning")).toHaveValue(index % 2 ? "medium" : "low");
    }
  }

  if (item.match === "sentence_pattern") {
    await expect(panel.locator("#rule-matching-behavior")).toBeDisabled();
  } else if (item.custom) {
    await expect(panel.locator("#rule-matching-behavior")).toHaveValue("custom");
    await expect(panel.locator("#rule-fuzzy")).toBeChecked();
    await expect(panel.locator("#rule-threshold")).toHaveValue(String(80 + index));
  } else {
    await expect(panel.locator("#rule-matching-behavior")).toHaveValue("defaults");
  }

  await panel.locator(".rule-close").first().click();
}

test("nightly Request Rule editor matrix round-trips all matchers and major form modes", async ({page}, testInfo) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  let panel = page.locator("extended-openai-management-panel");
  const groupId = await createGroup(panel);

  for (const [index, item] of MATCH_CASES.entries()) {
    await configureCase(panel, item, index, groupId);
    await assertCase(panel, item, index, groupId);
  }

  await page.goto(fixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  for (const [index, item] of MATCH_CASES.entries()) {
    await assertCase(panel, item, index, groupId);
  }

  const state = await page.evaluate(() => browserHarness.getState().requestRules);
  const created = state.rules.filter((rule) => rule.name.startsWith("Nightly editor "));
  expect(created).toHaveLength(MATCH_CASES.length);
  expect(new Set(created.map((rule) => rule.match_type))).toEqual(new Set(MATCH_CASES.map((item) => item.match)));

  await testInfo.attach("request-rule-editor-matrix", {
    body: JSON.stringify({
      matchers:MATCH_CASES.map((item) => item.match),
      actionTypes:[...new Set(MATCH_CASES.map((item) => item.action))],
      rules:created.length,
      freshLoadVerified:true,
    }, null, 2),
    contentType:"application/json",
  });
  console.log(`ENHANCED REQUEST_RULE_EDITOR_MATRIX rules=${created.length} matchers=${MATCH_CASES.length}`);
  await expectHarnessClean(page, errors);
});
