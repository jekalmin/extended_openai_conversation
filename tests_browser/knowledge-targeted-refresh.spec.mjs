import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

test("Knowledge availability patches selected agent and route without broad reloads", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("data-memory/knowledge"));
  const panel = page.locator("extended-openai-management-panel");
  const toggle = panel.locator("#knowledge-enabled-toggle");
  await expect(toggle).toBeVisible();

  const before = await page.evaluate(() => ({
    agents: browserHarness.calls.filter((call) => call.action === "agents").length,
    knowledgeList: browserHarness.calls.filter((call) => call.section === "knowledge" && call.action === "list").length,
    configGet: browserHarness.calls.filter((call) => call.section === "configuration" && call.action === "get").length,
  }));

  await panel.evaluate((host) => {
    const original = host._hass.callWS;
    const overviewKey = `${host._agentId}|overview`;
    host._sectionCache.set(overviewKey, {stale:true});
    host._eocSectionCacheTimes.set(overviewKey, Date.now());
    window.knowledgeTargetedCalls = [];
    host._hass.callWS = async (message) => {
      if (message.section === "knowledge" && message.action === "set_enabled") {
        window.knowledgeTargetedCalls.push(message);
        return {
          revision:"knowledge-targeted-revision",
          knowledge_enabled:message.enabled,
          feature_status:{
            state:message.enabled ? "enabled" : "disabled",
            enabled:message.enabled,
            source_count:Number(host._result?.stats?.source_count || 0),
          },
        };
      }
      return original(message);
    };
  });

  const initialChecked = await toggle.isChecked();
  await toggle.setChecked(!initialChecked);

  await expect.poll(() => page.evaluate(() => knowledgeTargetedCalls.length)).toBe(1);
  const state = await panel.evaluate((host) => ({
    routeEnabled:host._result?.feature_status?.enabled,
    agentEnabled:host._selectedAgent()?.knowledge_enabled,
    agentFeatureEnabled:host._selectedAgent()?.feature_status?.knowledge?.enabled,
    configData:host._configData,
    draft:host._draft,
    overviewCached:host._sectionCache.has(`${host._agentId}|overview`),
    mutationCount:host._eocAgentMutations,
  }));
  expect(state.routeEnabled).toBe(!initialChecked);
  expect(state.agentEnabled).toBe(!initialChecked);
  expect(state.agentFeatureEnabled).toBe(!initialChecked);
  expect(state.configData).toBeNull();
  expect(state.draft).toBeNull();
  expect(state.overviewCached).toBe(false);
  expect(state.mutationCount).toBe(0);

  const after = await page.evaluate(() => ({
    agents: browserHarness.calls.filter((call) => call.action === "agents").length,
    knowledgeList: browserHarness.calls.filter((call) => call.section === "knowledge" && call.action === "list").length,
    configGet: browserHarness.calls.filter((call) => call.section === "configuration" && call.action === "get").length,
    targeted:knowledgeTargetedCalls.map((call) => ({action:call.action, enabled:call.enabled})),
  }));
  expect(after.agents).toBe(before.agents);
  expect(after.knowledgeList).toBe(before.knowledgeList);
  expect(after.configGet).toBe(before.configGet);
  expect(after.targeted).toEqual([{action:"set_enabled", enabled:!initialChecked}]);
  await expectHarnessClean(page, errors);
});

test("late Knowledge availability response does not patch a different agent", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("data-memory/knowledge"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator("#knowledge-enabled-toggle")).toBeVisible();

  const result = await panel.evaluate(async (host) => {
    const original = host._hass.callWS;
    let release;
    host._hass.callWS = async (message) => {
      if (message.section === "knowledge" && message.action === "set_enabled") {
        await new Promise((resolve) => { release = resolve; });
        return {
          revision:"late",
          knowledge_enabled:false,
          feature_status:{state:"disabled", enabled:false},
        };
      }
      return original(message);
    };

    const input = host.shadowRoot.querySelector("#knowledge-enabled-toggle");
    input.checked = false;
    input.dispatchEvent(new Event("change", {bubbles:true}));
    while (!release) await new Promise((resolve) => setTimeout(resolve, 0));
    const firstAgent = host._selectedAgent();
    firstAgent.knowledge_enabled = true;
    host._agentId = "__different-agent__";
    release();
    while (host.shadowRoot.querySelector("#knowledge-enabled-toggle")?.disabled) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    host._hass.callWS = original;
    return {firstAgentEnabled:firstAgent.knowledge_enabled};
  });

  expect(result.firstAgentEnabled).toBe(true);
  await expectHarnessClean(page, errors);
});
