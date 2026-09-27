import {expect, test} from "@playwright/test";
import {acceptConfirmation, expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

test("Knowledge sources support create, reload, edit, and delete", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("data-memory/knowledge"));
  let panel = page.locator("extended-openai-management-panel");
  await panel.locator("#add-source").click();
  await panel.locator("#knowledge-title").fill("Browser reference");
  await panel.locator("#knowledge-description").fill("First revision");
  await panel.locator("#knowledge-content").fill("Reference facts for a browser journey.");
  await panel.locator("#knowledge-save").click();
  await expect(panel.getByText("Browser reference", {exact: true})).toBeVisible();
  await page.goto(fixtureUrl("data-memory/knowledge"));
  panel = page.locator("extended-openai-management-panel");
  let card = panel.locator(".list-card").filter({hasText: "Browser reference"});
  await expect(card).toBeVisible();
  await card.locator(".source-edit-button").click();
  await panel.locator("#knowledge-description").fill("Second revision");
  await panel.locator("#knowledge-save").click();
  await page.goto(fixtureUrl("data-memory/knowledge"));
  panel = page.locator("extended-openai-management-panel");
  card = panel.locator(".list-card").filter({hasText: "Browser reference"});
  await expect(card).toContainText("Second revision");
  await card.locator(".delete-source").click();
  await acceptConfirmation(panel);
  await page.goto(fixtureUrl("data-memory/knowledge"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByText("Browser reference", {exact: true})).toHaveCount(0);
  expect(await page.evaluate(() => window.browserHarness.getState().knowledgeSources)).toEqual([]);
  await expectHarnessClean(page, errors);
});

test("persistent memories support create, reload, edit, and delete", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(fixtureUrl("data-memory/memories"));

  let panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Memories", exact: true})).toBeVisible();
  await panel.locator("#add-memory").click();
  await expect(panel.locator("#memory-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#memory-content").fill("Browser journey memory");
  await panel.locator("#memory-category").fill("testing");
  await panel.locator("#memory-save").click();
  await expect(panel.getByText("Browser journey memory", {exact: true})).toBeVisible();

  await page.goto(fixtureUrl("data-memory/memories"));
  panel = page.locator("extended-openai-management-panel");
  let card = panel.locator(".list-card").filter({hasText: "Browser journey memory"});
  await expect(card).toBeVisible();
  await card.locator(".memory-edit-button").click();
  await panel.locator("#memory-content").fill("Browser journey memory edited");
  await panel.locator("#memory-category").fill("testing-edited");
  await panel.locator("#memory-save").click();

  await page.goto(fixtureUrl("data-memory/memories"));
  panel = page.locator("extended-openai-management-panel");
  card = panel.locator(".list-card").filter({hasText: "Browser journey memory edited"});
  await expect(card).toContainText("testing-edited");
  await card.locator(".delete-memory").click();
  await acceptConfirmation(panel);
  await expect(panel.getByText("Browser journey memory edited", {exact: true})).toHaveCount(0);

  await page.goto(fixtureUrl("data-memory/memories"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByText("Browser journey memory edited", {exact: true})).toHaveCount(0);
  await expect(panel.getByText("Baseline browser fixture memory", {exact: true})).toBeVisible();
  await expectHarnessClean(page, pageErrors);
});

test("Request Rules support create, precedence changes, reload, edit, and delete", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));

  let panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Request Rules", exact: true})).toBeVisible();
  await panel.getByRole("button", {name: "Create rule", exact: true}).first().click();
  await expect(panel.locator("#rule-dialog")).toHaveJSProperty("open", true);
  await expect(panel.locator("#rule-enabled-edit")).toHaveCount(0);
  await panel.locator("#rule-name").fill("Browser rule");
  await panel.locator("#rule-phrases").fill("browser route");
  await panel.locator("#rule-match").selectOption("contains");
  await panel.locator("#rule-action-type").selectOption("model_routing");
  await panel.locator("#rule-model").fill("gpt-5-mini");
  await panel.locator("#rule-reasoning").selectOption("medium");
  await panel.locator("#rule-scope").selectOption("request");
  await panel.locator("#rule-save").click();
  await expect(panel.getByRole("heading", {name: "Browser rule", exact: true})).toBeVisible();

  let card = panel.locator(".request-rule-card").filter({hasText: "Browser rule"});
  await expect(card.locator(".rule-enabled")).toBeChecked();
  await card.locator('.rule-move[data-direction="up"]').click();
  await expect.poll(async () => panel.locator(".request-rule-card h2").allTextContents()).toEqual(["Browser rule", "Baseline rule"]);

  await page.goto(fixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  await expect.poll(async () => panel.locator(".request-rule-card h2").allTextContents()).toEqual(["Browser rule", "Baseline rule"]);
  card = panel.locator(".request-rule-card").filter({hasText: "Browser rule"});
  await card.locator(".rule-edit").click();
  await expect(panel.locator("#rule-enabled-edit")).toHaveCount(0);
  await panel.locator("#rule-name").fill("Browser rule edited");
  await panel.locator("#rule-model").fill("gpt-5-nano");
  await panel.locator("#rule-save").click();
  await expect(panel.getByRole("heading", {name: "Browser rule edited", exact: true})).toBeVisible();

  await page.goto(fixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  card = panel.locator(".request-rule-card").filter({hasText: "Browser rule edited"});
  await expect(card).toContainText("gpt-5-nano");
  await card.locator(".rule-delete").click();
  await acceptConfirmation(panel);
  await expect(panel.getByRole("heading", {name: "Browser rule edited", exact: true})).toHaveCount(0);

  await page.goto(fixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Browser rule edited", exact: true})).toHaveCount(0);
  await expect(panel.getByRole("heading", {name: "Baseline rule", exact: true})).toBeVisible();
  await expectHarnessClean(page, pageErrors);
});

test("Rule Sharing reviews before importing disabled rules at the bottom", async ({page}) => {
  const pageErrors=trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel=page.locator("extended-openai-management-panel");
  const sharing=panel.locator("#rule-sharing");
  await expect(sharing).not.toHaveAttribute("open", "");
  await expect(panel.locator("#rule-pack-file")).toHaveCount(0);
  await sharing.locator("summary").click();
  await expect(panel.locator("#rule-pack-file")).toBeVisible();
  const pack={format:"extended_openai_request_rule_pack",version:1,groups:[],rules:[{
    id:"portable-rule",name:"Shared route",enabled:true,phrases:["shared route"],match_type:"equals",action_type:"model_routing",
    action:{model:"gpt-5-mini",reasoning_effort:"",scope:"request",reset:false,continue_to_ai:true,success_response:"Updated"},
    matching_behavior:"custom",matching:{word_forms:true,wording_alternatives:true,fuzzy:false,fuzzy_threshold:90},order:0,
    conditions:[],group_id:null,continue_matching:true,ai_input_mode:"original",ai_input_capture:null,
  }]};
  await panel.locator("#rule-pack-file").setInputFiles({name:"rules.json",mimeType:"application/json",buffer:Buffer.from(JSON.stringify(pack))});
  await panel.locator("#rule-pack-review-button").click();
  await expect(panel.getByRole("heading",{name:"Review Rule Pack"})).toBeVisible();
  await expect(panel.getByRole("heading",{name:"Shared route"})).toHaveCount(0);
  await panel.locator("#rule-pack-confirm").click();
  const card=panel.locator(".request-rule-card").filter({hasText:"Shared route"});
  await expect(card).toBeVisible();
  await expect(card.locator(".rule-enabled")).not.toBeChecked();
  await expect.poll(async()=>panel.locator(".request-rule-card h2").allTextContents()).toEqual(["Baseline rule","Shared route"]);
  await panel.locator("#rule-pack-selection").selectOption("selected");
  await panel.locator("#rule-pack-export").click();
  await expect(panel.locator("#rule-pack-select-dialog")).toHaveJSProperty("open",true);
  await expect(panel.locator("#rule-pack-rule-list input")).toHaveCount(2);
  await panel.locator("#rule-pack-select-cancel").click();
  await expectHarnessClean(page,pageErrors);
});

test("Captured AI input keeps an invalid selection visible for repair", async ({page}) => {
  const pageErrors=trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel=page.locator("extended-openai-management-panel");
  await panel.getByRole("button",{name:"Create rule",exact:true}).first().click();
  await panel.locator("#rule-phrases").fill("deep think {question}\nthink {question}");
  await panel.locator("#rule-match").selectOption("sentence_pattern");
  await panel.locator("#rule-action-type").selectOption("model_routing");
  await panel.locator("#rule-ai-input-mode").selectOption("capture");
  await expect(panel.locator("#rule-ai-input-capture option")).toHaveCount(1);
  await expect(panel.locator("#rule-ai-input-capture")).toHaveValue("question");
  await panel.locator("#rule-phrases").fill("deep think {question}\nthink carefully");
  await expect(panel.locator("#rule-ai-input-capture")).toHaveValue("question");
  await expect(panel.locator("#rule-ai-input-capture")).toHaveAttribute("aria-invalid","true");
  await expect(panel.locator("#rule-ai-input-help")).toBeVisible();
  await panel.locator("#rule-match").selectOption("equals");
  await expect(panel.locator("#rule-match")).toHaveValue("equals");
  await expect(panel.locator("#rule-ai-input-capture")).toHaveValue("question");
  await panel.locator(".rule-close").first().click();
  await expectHarnessClean(page,pageErrors);
});

test("Request Rule condition selector, local continuation, and group survive reload", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  let panel = page.locator("extended-openai-management-panel");
  await panel.locator("#rule-groups-manage").click();
  await expect(panel.locator("#rule-groups-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#rule-new-group-name").fill("Kitchen");
  await panel.locator("#rule-group-add").click();
  await expect(panel.locator(".rule-group-row")).toHaveCount(1);
  await expect(panel.locator(".rule-group-row .rule-group-count")).toHaveText("0 rules");
  await panel.locator("#rule-groups-done").click();
  await panel.getByRole("button", {name:"Create rule", exact:true}).first().click();
  await panel.locator("#rule-name").fill("Conditional local");
  await panel.locator("#rule-phrases").fill("good kitchen");
  await panel.locator("#rule-group").selectOption({label:"Kitchen"});
  await panel.locator("#rule-local-continue-to-ai").check();
  await panel.locator("#rule-advanced summary").click();
  await panel.locator("#rule-continue-matching").check();
  await expect(panel.locator("#rule-action-sequence-host ha-selector")).toHaveJSProperty("value", []);
  await panel.locator("#rule-save").click();
  await expect(panel.locator("#rule-error")).toContainText("Add at least one action before saving this rule.");
  await panel.locator("#rule-action-sequence-host ha-selector").evaluate((selector) => {
    const value=[{action:"light.turn_on",target:{entity_id:"light.kitchen"}}];
    selector.value=value;
    selector.dispatchEvent(new CustomEvent("value-changed", {detail:{value},bubbles:true,composed:true}));
  });
  const condition = [{condition:"and",conditions:[{condition:"state",entity_id:"input_boolean.kitchen_ready",state:"on"},{condition:"not",conditions:[{condition:"state",entity_id:"input_boolean.kitchen_busy",state:"on"}]}]}];
  await panel.locator("#rule-condition-host ha-selector").evaluate((selector, value) => {
    selector.value=value;
    selector.dispatchEvent(new CustomEvent("value-changed", {detail:{value},bubbles:true,composed:true}));
  }, condition);
  await panel.locator("#rule-save").click();
  await expect(panel.getByRole("heading", {name:"Conditional local", exact:true})).toBeVisible();
  await page.goto(fixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  const card = panel.locator(".request-rule-card").filter({hasText:"Conditional local"});
  await expect(card).toContainText("Kitchen");
  await expect(card).toContainText("continues to AI");
  await card.locator(".rule-edit").click();
  await expect(panel.locator("#rule-local-continue-to-ai")).toBeChecked();
  await expect(panel.locator("#rule-continue-matching")).toBeChecked();
  await expect(panel.locator("#rule-group")).toHaveValue(await panel.locator(".rule-group-row").getAttribute("data-group-id"));
  const saved = await panel.locator("#rule-condition-host ha-selector").evaluate((selector) => selector.value);
  expect(saved).toEqual(condition);
  await panel.locator(".rule-close").first().click();
  await expectHarnessClean(page, pageErrors);
});

test("Request Rule Show filter keeps global priorities and group changes preserve order", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = page.locator("extended-openai-management-panel");
  await panel.locator("#rule-groups-manage").click();
  for (const name of ["Lighting", "Media"]) {
    await panel.locator("#rule-new-group-name").fill(name);
    await panel.locator("#rule-group-add").click();
  }
  await panel.locator("#rule-groups-done").click();
  for (const [name, group] of [["Lamp rule", "Lighting"], ["Music rule", "Media"]]) {
    await panel.getByRole("button", {name:"Create rule", exact:true}).first().click();
    await panel.locator("#rule-name").fill(name);
    await panel.locator("#rule-phrases").fill(name.toLowerCase());
    await panel.locator("#rule-group").selectOption({label:group});
    await panel.locator("#rule-action-type").selectOption("model_routing");
    await panel.locator("#rule-save").click();
  }
  const cards = panel.locator(".request-rule-card");
  await expect(cards).toHaveCount(3);
  await expect(cards.locator(".rule-card-heading .meta")).toContainText(["#1", "#2", "#3"]);
  await expect(cards.locator(".rule-group-chip")).toHaveText(["Ungrouped", "Lighting", "Media"]);
  await panel.locator("#rule-group-filter").selectOption({label:"Lighting"});
  await expect(panel.locator(".request-rule-card:visible")).toHaveCount(1);
  await expect(panel.locator(".request-rule-card:visible")).toContainText("#2");
  await expect(panel.locator(".request-rule-card:visible .rule-move").first()).toBeDisabled();
  await panel.locator("#rule-group-filter").selectOption({label:"All"});
  const music=cards.filter({hasText:"Music rule"}), lamp=cards.filter({hasText:"Lamp rule"});
  await page.setViewportSize({width:1280,height:1200});
  await music.evaluate(node => { window.__musicCard = node; });
  await music.dragTo(lamp,{targetPosition:{x:20,y:5}});
  await expect(cards.locator(".rule-group-chip")).toHaveText(["Ungrouped", "Media", "Lighting"]);
  expect(await music.evaluate(node => node === window.__musicCard)).toBe(true);
  await panel.locator("#rule-groups-manage").click();
  const lighting = panel.locator(".rule-group-row").filter({has:page.locator('input[value="Lighting"]')});
  await expect(lighting.locator(".rule-group-count")).toHaveText("1 rule");
  await lighting.locator(".rule-group-name").fill("Lights");
  await lighting.locator(".rule-group-rename").click();
  await expect(cards.locator(".rule-group-chip")).toHaveText(["Ungrouped", "Media", "Lights"]);
  await panel.locator(".rule-group-row").filter({has:page.locator('input[value="Lights"]')}).locator(".rule-group-delete").click();
  await expect(panel.locator("#confirm-dialog")).toContainText("become Ungrouped and keep their existing priority and order");
  await acceptConfirmation(panel);
  await expect(cards.locator(".rule-group-chip")).toHaveText(["Ungrouped", "Media", "Ungrouped"]);
  await expect(cards.filter({hasText:"Lamp rule"})).toContainText("#3");
  await panel.locator("#rule-groups-done").click();
  await page.goto(fixtureUrl("capabilities/request-rules"));
  await expect(cards.locator(".rule-group-chip")).toHaveText(["Ungrouped", "Media", "Ungrouped"]);
  await expect(cards.filter({hasText:"Music rule"})).toContainText("#2");
  await expectHarnessClean(page, pageErrors);
});
