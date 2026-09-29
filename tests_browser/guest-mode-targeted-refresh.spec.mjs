import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

test("Guest Mode primary UI paints before capability details settle", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("overview"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".dashboard-grid")).toBeVisible();

  const result = await panel.evaluate(async (host) => {
    const original = host._hass.callWS;
    let releaseDetails;
    host._hass.callWS = async (message) => {
      if (message.section === "guest_mode" && message.action === "details") {
        await new Promise((resolve) => { releaseDetails = resolve; });
      }
      return original(message);
    };

    let navigationResolved = false;
    const pending = host._navigate("capabilities", "guest-mode").then(() => {
      navigationResolved = true;
    });
    while (!releaseDetails) await new Promise((resolve) => setTimeout(resolve, 0));
    const primaryIsVisible = () => Boolean(
      host.shadowRoot.querySelector(".guest-intro")
      && host.shadowRoot.querySelector("#guest-now")
    );
    const detailsAreLoading = () => host._result?.loading?.details === true
      && host.shadowRoot.textContent.includes("Loading Guest capability details");
    const primaryDeadline = Date.now() + 5000;
    while (Date.now() < primaryDeadline
      && (!navigationResolved || !primaryIsVisible() || !detailsAreLoading())) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }

    const primaryVisible = primaryIsVisible();
    const loadingDetails = detailsAreLoading();

    releaseDetails();
    await pending;
    const detailsDeadline = Date.now() + 5000;
    while (Date.now() < detailsDeadline && host._result?.loading?.details) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    host._hass.callWS = original;
    return {
      navigationResolved,
      primaryVisible,
      loadingDetails,
      detailsLoaded: host._result?.loading?.details === false,
      policyPresent: Boolean(host._result?.policy),
    };
  });

  expect(result).toEqual({
    navigationResolved:true,
    primaryVisible:true,
    loadingDetails:true,
    detailsLoaded:true,
    policyPresent:true,
  });
  await expectHarnessClean(page, errors);
});

test("Guest Mode schedule mutations refresh only the selected agent and route", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/guest-mode"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name:"Guest Mode", exact:true})).toBeVisible();

  const result = await panel.evaluate(async (host) => {
    const original = host._hass.callWS;
    const calls = window.browserHarness.calls;
    const initialAgentCalls = calls.filter((call) => call.action === "agents").length;
    const agentId = host._agentId;
    const agent = host._data.agents.find((item) => item.subentry_id === agentId);
    agent.guest_mode = {...(agent.guest_mode || {}), has_home_assistant_exclusions:true};

    let status = {
      state:"inactive",
      currently_active:false,
      scheduled:false,
      indefinite:false,
      active_from:null,
      active_until:null,
    };
    const guestCalls = [];
    host._hass.callWS = async (message) => {
      if (message.section !== "guest_mode") return original(message);
      guestCalls.push(message.action);
      if (message.action === "update") {
        status = {
          state:"active_indefinitely",
          currently_active:true,
          scheduled:false,
          indefinite:true,
          active_from:"2026-09-22T16:00:00+00:00",
          active_until:null,
        };
        return {status:{...status}};
      }
      if (message.action === "disable") {
        status = {
          state:"inactive",
          currently_active:false,
          scheduled:false,
          indefinite:false,
          active_from:null,
          active_until:null,
        };
        return {status:{...status}};
      }
      if (message.action === "get") {
        return {
          ...host._result,
          status:{...status},
        };
      }
      if (message.action === "details") {
        return {
          policy:{guest_active:status.currently_active, marker:status.state},
          knowledge_sources:[],
          functions:[],
          function_groups:[],
          domains:[],
        };
      }
      return original(message);
    };

    await host._updateGuestMode(true);
    const afterUpdate = {
      agent:{...agent.guest_mode},
      routeStatus:{...host._result.status},
      routePolicy:{...host._result.policy},
    };

    host._confirm = async () => true;
    await host._disableGuestMode();
    const afterDisable = {
      agent:{...agent.guest_mode},
      routeStatus:{...host._result.status},
      routePolicy:{...host._result.policy},
    };

    host._hass.callWS = original;
    return {
      initialAgentCalls,
      finalAgentCalls:calls.filter((call) => call.action === "agents").length,
      guestCalls,
      afterUpdate,
      afterDisable,
    };
  });

  expect(result.finalAgentCalls).toBe(result.initialAgentCalls);
  expect(result.guestCalls).toEqual(["update", "details", "disable", "details"]);
  expect(result.afterUpdate.agent).toMatchObject({
    state:"active_indefinitely",
    currently_active:true,
    has_home_assistant_exclusions:true,
  });
  expect(result.afterUpdate.routeStatus.state).toBe("active_indefinitely");
  expect(result.afterUpdate.routePolicy).toMatchObject({guest_active:true, marker:"active_indefinitely"});
  expect(result.afterDisable.agent).toMatchObject({
    state:"inactive",
    currently_active:false,
    has_home_assistant_exclusions:true,
  });
  expect(result.afterDisable.routeStatus.state).toBe("inactive");
  expect(result.afterDisable.routePolicy).toMatchObject({guest_active:false, marker:"inactive"});
  await expectHarnessClean(page, errors);
});

test("late Guest Mode refresh does not replace a newer route", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/guest-mode"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name:"Guest Mode", exact:true})).toBeVisible();

  const result = await panel.evaluate(async (host) => {
    const original = host._hass.callWS;
    let releaseDetails;
    host._hass.callWS = async (message) => {
      if (message.section === "guest_mode" && message.action === "update") {
        return {status:{state:"active_indefinitely", currently_active:true, indefinite:true}};
      }
      if (message.section === "guest_mode" && message.action === "details") {
        await new Promise((resolve) => { releaseDetails = resolve; });
        return {policy:{guest_active:true}, knowledge_sources:[], functions:[], function_groups:[], domains:[]};
      }
      return original(message);
    };

    const pending = host._updateGuestMode(true);
    while (!releaseDetails) await new Promise((resolve) => setTimeout(resolve, 0));
    await host._navigate("guide");
    const guideResult = host._result;
    releaseDetails();
    await pending;
    host._hass.callWS = original;
    return {
      view:host._viewKey(),
      sameResult:host._result === guideResult,
      agentState:host._data.agents.find((item) => item.subentry_id === host._agentId)?.guest_mode?.state,
    };
  });

  expect(result).toMatchObject({
    view:"guide",
    sameResult:true,
    agentState:"active_indefinitely",
  });
  await expectHarnessClean(page, errors);
});

test("saving Guest exclusions refreshes visible derived counts without reloading the route", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/guest-mode"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name:"Guest Mode", exact:true})).toBeVisible();
  await panel.evaluate(host => {
    const original = host._hass.callWS.bind(host._hass);
    let config = structuredClone(host._result.config);
    host._result.policy = {readable_entity_count:7, controllable_entity_count:7, configured_tool_count:0};
    window.guestProjectionCalls = [];
    host._hass.callWS = async message => {
      if (message.section === "guest_mode" && message.action === "save_policy") {
        window.guestProjectionCalls.push("save_policy");
        config = structuredClone(message.config);
        return {revision:`${message.revision}x`, config};
      }
      if (message.section === "guest_mode" && message.action === "details") {
        window.guestProjectionCalls.push("details");
        const excluded = (config.guest_excluded_domains || []).length + (config.guest_excluded_entities || []).length;
        return {policy:{readable_entity_count:7-excluded, controllable_entity_count:7-excluded, configured_tool_count:0}, knowledge_sources:[], functions:[], function_groups:[], domains:["camera"]};
      }
      return original(message);
    };
    host._render();
  });

  const visibleCount = () => panel.locator(".metric-grid").filter({hasText:"Guest-visible entities"});
  await expect(visibleCount()).toContainText("7");
  const exclusions = panel.locator('ha-selector[data-guest-key="guest_excluded_domains"]');
  await exclusions.evaluate(node => { node.value = ["camera"]; node.dispatchEvent(new CustomEvent("value-changed", {detail:{value:["camera"]}, bubbles:true})); });
  await expect(panel.locator(".save-bar #save-page")).toBeEnabled();
  const agentReads = await page.evaluate(() => browserHarness.calls.filter(call => call.action === "agents").length);
  await panel.locator(".save-bar #save-page").click();
  await expect(visibleCount()).toContainText("6");
  expect(await page.evaluate(() => browserHarness.calls.filter(call => call.action === "agents").length)).toBe(agentReads);
  expect(await page.evaluate(() => guestProjectionCalls)).toEqual(["save_policy", "details"]);

  await exclusions.evaluate(node => { node.value = []; node.dispatchEvent(new CustomEvent("value-changed", {detail:{value:[]}, bubbles:true})); });
  await panel.locator(".save-bar #save-page").click();
  await expect(visibleCount()).toContainText("7");
  expect(await page.evaluate(() => guestProjectionCalls)).toEqual(["save_policy", "details", "save_policy", "details"]);

  const entities = panel.locator('ha-selector[data-guest-key="guest_excluded_entities"]');
  await entities.evaluate(node => { node.value = ["light.kitchen"]; node.dispatchEvent(new CustomEvent("value-changed", {detail:{value:["light.kitchen"]}, bubbles:true})); });
  await expect(panel.locator(".save-bar #save-page")).toBeEnabled();
  await panel.locator(".save-bar #save-page").click();
  await expect(visibleCount()).toContainText("6");
  expect(await page.evaluate(() => guestProjectionCalls)).toEqual(["save_policy", "details", "save_policy", "details", "save_policy", "details"]);
  await expectHarnessClean(page, errors);
});
