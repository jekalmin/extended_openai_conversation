import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

async function createRoutingRule(panel, index) {
  await panel.getByRole("button", {name:"Create rule", exact:true}).first().click();
  await panel.locator("#rule-name").fill(`Nightly queue ${index}`);
  await panel.locator("#rule-phrases").fill(`nightly queue ${index}`);
  await panel.locator("#rule-action-type").selectOption("model_routing");
  await panel.locator("#rule-model").fill("gpt-5-mini");
  await panel.locator("#rule-save").click();
}

test("nightly Request Rules queue serializes a burst of distinct mutations", async ({page}, testInfo) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = page.locator("extended-openai-management-panel");

  for (let index = 1; index <= 4; index += 1) await createRoutingRule(panel, index);
  const startRevision = await page.evaluate(() => browserHarness.getState().requestRules.revision);

  await page.evaluate(() => {
    const hass = browserHarness.hass;
    const original = hass.callWS.bind(hass);
    let release;
    const gate = new Promise(resolve => { release = resolve; });
    window.nightlyRuleQueue = {started:[], release};
    hass.callWS = async request => {
      if (request.section === "request_rules" && request.action === "update") {
        nightlyRuleQueue.started.push(structuredClone(request));
        if (nightlyRuleQueue.started.length === 1) await gate;
      }
      return original(request);
    };
  });

  const toggles = panel.locator(".request-rule-card .rule-enabled");
  for (let index = 0; index < 4; index += 1) {
    await toggles.nth(index).uncheck();
  }
  await expect.poll(() => page.evaluate(() => nightlyRuleQueue.started.length)).toBe(1);
  await page.waitForTimeout(50);
  expect(await page.evaluate(() => nightlyRuleQueue.started.length)).toBe(1);

  await page.evaluate(() => nightlyRuleQueue.release());
  await expect.poll(() => page.evaluate(() => nightlyRuleQueue.started.length)).toBe(4);

  const revisions = await page.evaluate(() => nightlyRuleQueue.started.map(item => item.revision));
  expect(revisions).toEqual([
    startRevision,
    startRevision + 1,
    startRevision + 2,
    startRevision + 3,
  ]);
  expect(await page.evaluate(() => browserHarness.getState().requestRules.revision)).toBe(startRevision + 4);
  for (let index = 0; index < 4; index += 1) await expect(toggles.nth(index)).not.toBeChecked();

  await testInfo.attach("request-rule-mutation-queue", {
    body:JSON.stringify({mutations:4, revisions, serialized:true}, null, 2),
    contentType:"application/json",
  });
  console.log(`ENHANCED REQUEST_RULE_MUTATION_QUEUE mutations=4 start_revision=${startRevision}`);
  await expectHarnessClean(page, errors);
});
