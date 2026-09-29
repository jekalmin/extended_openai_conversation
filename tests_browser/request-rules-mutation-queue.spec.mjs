import {expect, test} from "@playwright/test";
import {acceptConfirmation, expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

async function createRoutingRule(panel, name, phrase) {
  await panel.getByRole("button", {name:"Create rule", exact:true}).first().click();
  await panel.locator("#rule-name").fill(name);
  await panel.locator("#rule-phrases").fill(phrase);
  await panel.locator("#rule-action-type").selectOption("model_routing");
  await panel.locator("#rule-model").fill("gpt-5-mini");
  await panel.locator("#rule-save").click();
  await expect(panel.locator(".request-rule-card").filter({hasText:name})).toBeVisible();
}

async function gateFirstRequestRuleMutation(page) {
  await page.evaluate(() => {
    const hass = browserHarness.hass;
    const original = hass.callWS.bind(hass);
    let release;
    const gate = new Promise(resolve => { release = resolve; });
    window.ruleMutationGate = {started:[], release};
    hass.callWS = async request => {
      if (request.section === "request_rules" && [
        "update", "move", "duplicate", "delete", "groups",
      ].includes(request.action)) {
        window.ruleMutationGate.started.push(structuredClone(request));
        if (window.ruleMutationGate.started.length === 1) await gate;
      }
      return original(request);
    };
  });
}

test("different Request Rule controls serialize and resolve revision when each mutation starts", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = page.locator("extended-openai-management-panel");
  await createRoutingRule(panel, "Queue second", "queue second");

  const startRevision = await page.evaluate(() => browserHarness.getState().requestRules.revision);
  const cards = panel.locator(".request-rule-card");
  const firstToggle = cards.filter({hasText:"Baseline rule"}).locator(".rule-enabled");
  const secondToggle = cards.filter({hasText:"Queue second"}).locator(".rule-enabled");

  await gateFirstRequestRuleMutation(page);
  await firstToggle.uncheck();
  await expect.poll(() => page.evaluate(() => ruleMutationGate.started.length)).toBe(1);

  // A distinct control remains usable, but its backend mutation must wait for
  // the shared Request Rules revision to advance.
  await secondToggle.uncheck();
  await page.waitForTimeout(50);
  expect(await page.evaluate(() => ruleMutationGate.started.length)).toBe(1);

  await page.evaluate(() => ruleMutationGate.release());
  await expect.poll(() => page.evaluate(() => ruleMutationGate.started.length)).toBe(2);

  const requests = await page.evaluate(() => ruleMutationGate.started.map(
    ({action, rule_id, revision}) => ({action, rule_id, revision}),
  ));
  expect(requests.map(item => item.action)).toEqual(["update", "update"]);
  expect(requests.map(item => item.revision)).toEqual([startRevision, startRevision + 1]);
  await expect(firstToggle).not.toBeChecked();
  await expect(secondToggle).not.toBeChecked();
  expect(await page.evaluate(() => browserHarness.getState().requestRules.revision)).toBe(startRevision + 2);
  await expectHarnessClean(page, errors);
});

test("move followed by duplicate uses the post-move revision and preserves both results", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = page.locator("extended-openai-management-panel");
  await createRoutingRule(panel, "Queue A", "queue a");
  await createRoutingRule(panel, "Queue B", "queue b");
  const startRevision = await page.evaluate(() => browserHarness.getState().requestRules.revision);

  const a = panel.locator(".request-rule-card").filter({hasText:"Queue A"});
  const b = panel.locator(".request-rule-card").filter({hasText:"Queue B"});
  await gateFirstRequestRuleMutation(page);

  await a.locator(".rule-move-menu > summary").click();
  await a.locator('.rule-move[data-direction="up"]').click();
  await expect.poll(() => page.evaluate(() => ruleMutationGate.started.length)).toBe(1);

  await b.locator(".rule-duplicate").click();
  await page.waitForTimeout(50);
  expect(await page.evaluate(() => ruleMutationGate.started.length)).toBe(1);

  await page.evaluate(() => ruleMutationGate.release());
  await expect.poll(() => page.evaluate(() => ruleMutationGate.started.length)).toBe(2);
  const requests = await page.evaluate(() => ruleMutationGate.started.map(
    ({action, revision}) => ({action, revision}),
  ));
  expect(requests).toEqual([
    {action:"move", revision:startRevision},
    {action:"duplicate", revision:startRevision + 1},
  ]);
  await expect(panel.locator(".request-rule-card").filter({hasText:"Queue B"})).toHaveCount(2);
  expect(await page.evaluate(() => browserHarness.getState().requestRules.revision)).toBe(startRevision + 2);
  await expectHarnessClean(page, errors);
});

for (const scenario of [
  {name:"duplicate", selector:".rule-duplicate", action:"duplicate"},
  {name:"move", selector:'.rule-move[data-direction="down"]', action:"move", openMove:true},
]) {
  test(`Request Rule ${scenario.name} ignores repeated activation while pending`, async ({page}) => {
    const errors = trackPageErrors(page);
    await page.goto(fixtureUrl("capabilities/request-rules"));
    const panel = page.locator("extended-openai-management-panel");
    await createRoutingRule(panel, "Repeat target", "repeat target");
    const card = panel.locator(".request-rule-card").filter({hasText:"Baseline rule"});
    if (scenario.openMove) await card.locator(".rule-move-menu > summary").click();
    const button = card.locator(scenario.selector);

    await gateFirstRequestRuleMutation(page);
    await button.evaluate(node => {
      node.dispatchEvent(new MouseEvent("click", {bubbles:true, composed:true}));
      node.dispatchEvent(new MouseEvent("click", {bubbles:true, composed:true}));
    });
    await expect.poll(() => page.evaluate(() => ruleMutationGate.started.length)).toBe(1);
    await expect(button).toBeDisabled();
    await page.evaluate(() => ruleMutationGate.release());
    await expect.poll(() => page.evaluate(action => browserHarness.calls.filter(
      call => call.section === "request_rules" && call.action === action,
    ).length, scenario.action)).toBe(1);
    await expectHarnessClean(page, errors);
  });
}

test("Request Rule enable toggle ignores a repeated change while pending", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = page.locator("extended-openai-management-panel");
  const toggle = panel.locator(".request-rule-card").filter({hasText:"Baseline rule"}).locator(".rule-enabled");
  await gateFirstRequestRuleMutation(page);

  await toggle.evaluate(node => {
    node.checked = false;
    node.dispatchEvent(new Event("change", {bubbles:true, composed:true}));
    node.checked = true;
    node.dispatchEvent(new Event("change", {bubbles:true, composed:true}));
  });
  await expect.poll(() => page.evaluate(() => ruleMutationGate.started.length)).toBe(1);
  await expect(toggle).toBeDisabled();
  await page.evaluate(() => ruleMutationGate.release());
  await expect(toggle).toBeEnabled();
  expect(await page.evaluate(() => browserHarness.calls.filter(
    call => call.section === "request_rules" && call.action === "update",
  ).length)).toBe(1);
  await expectHarnessClean(page, errors);
});

test("Request Rule group manager suppresses a second mutation while its save is pending", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = page.locator("extended-openai-management-panel");
  await panel.locator("#rule-groups-manage").click();
  await panel.locator("#rule-new-group-name").fill("Single-fire group");
  const add = panel.locator("#rule-group-add");
  await gateFirstRequestRuleMutation(page);

  await add.evaluate(node => {
    node.dispatchEvent(new MouseEvent("click", {bubbles:true, composed:true}));
    node.dispatchEvent(new MouseEvent("click", {bubbles:true, composed:true}));
  });
  await expect.poll(() => page.evaluate(() => ruleMutationGate.started.length)).toBe(1);
  await expect(add).toBeDisabled();
  await page.evaluate(() => ruleMutationGate.release());
  await expect.poll(() => page.evaluate(() => browserHarness.getState().requestRules.groups.length)).toBe(1);
  expect(await page.evaluate(() => browserHarness.calls.filter(
    call => call.section === "request_rules" && call.action === "groups",
  ).length)).toBe(1);
  await expectHarnessClean(page, errors);
});

test("Request Rule delete cannot be submitted twice after confirmation", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = page.locator("extended-openai-management-panel");
  await createRoutingRule(panel, "Delete once", "delete once");
  const card = panel.locator(".request-rule-card").filter({hasText:"Delete once"});
  await gateFirstRequestRuleMutation(page);

  await card.locator(".rule-delete").click();
  await acceptConfirmation(panel);
  await expect.poll(() => page.evaluate(() => ruleMutationGate.started.length)).toBe(1);
  const button = card.locator(".rule-delete");
  await expect(button).toBeDisabled();
  await button.evaluate(node => {
    node.dispatchEvent(new MouseEvent("click", {bubbles:true, composed:true}));
    node.dispatchEvent(new MouseEvent("click", {bubbles:true, composed:true}));
  });
  expect(await page.evaluate(() => ruleMutationGate.started.length)).toBe(1);
  await page.evaluate(() => ruleMutationGate.release());
  await expect(card).toHaveCount(0);
  expect(await page.evaluate(() => browserHarness.calls.filter(
    call => call.section === "request_rules" && call.action === "delete",
  ).length)).toBe(1);
  await expectHarnessClean(page, errors);
});
