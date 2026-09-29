import {expect, test} from "@playwright/test";
import {acceptConfirmation, expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

const panelFor = (page) => page.locator("extended-openai-management-panel");

const run = (runId, errorType) => ({
  run_id:runId,
  completed_at:"2026-09-29T12:00:00Z",
  total_tokens:240,
  cached_input_tokens:40,
  request_count:2,
  duration_ms:800,
  successful:false,
  error_type:errorType,
});

async function openUsage(page, customCall) {
  await page.goto(fixtureUrl("overview"));
  const panel = panelFor(page);
  await expect(panel.locator(".dashboard-grid")).toBeVisible();
  await panel.evaluate(async (host, callSource) => {
    const custom = Function(`return (${callSource})`)();
    const original = host._hass.callWS.bind(host._hass);
    window.usageDetailsNightly = {calls:[]};
    host._hass.callWS = async (message) => {
      if (message.section === "usage") {
        window.usageDetailsNightly.calls.push(structuredClone(message));
        const response = await custom(message);
        if (response !== undefined) return response;
      }
      return original(message);
    };
    await host._navigate("usage-maintenance", "usage");
  }, customCall.toString());
  await expect(panel.locator("#usage-window")).toBeVisible();
  return panel;
}

test("nightly run details show failed provider requests, recover from errors, and ignore closed or switched dialogs", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openUsage(page, async (message) => {
    if (message.action === "summary") return {today:{date:"2026-09-29",total_tokens:240},lifetime:{total_tokens:240}};
    if (message.action === "daily") return {days:[{date:"2026-09-29",total_tokens:240}]};
    if (message.action === "retention") return {};
    if (message.action === "runs") return {runs:[
      {run_id:"run-close",successful:false,error_type:"TimeoutError"},
      {run_id:"run-failed",successful:false,error_type:"ProviderTimeout"},
      {run_id:"run-empty",successful:false,error_type:"NoDetails"},
      {run_id:"run-retry",successful:false,error_type:"Retryable"},
    ]};
    if (message.action !== "requests") return undefined;
    if (message.run_id === "run-close") return new Promise((resolve) => { window.releaseRunClose = () => resolve({requests:[{provider:"Stale provider",model:"old-model",successful:true}]}); });
    if (message.run_id === "run-failed") return new Promise((resolve) => { window.releaseRunFailed = () => resolve({requests:[{
      provider:"Nightly provider",model:"gpt-nightly",api_mode:"responses",timestamp:"2026-09-29T12:00:00Z",request_stage:"primary",successful:false,error_type:"ProviderTimeout",total_tokens:70,cached_input_tokens:10,reasoning_tokens:5,duration_ms:6500,tool_calls_requested:1,web_search_used:true,
    }]}); });
    if (message.run_id === "run-empty") return {requests:[]};
    if (message.run_id === "run-retry") {
      window.usageDetailsNightly.retryCount = (window.usageDetailsNightly.retryCount || 0) + 1;
      if (window.usageDetailsNightly.retryCount === 1) throw new Error("Request details temporarily unavailable");
      return {requests:[{provider:"Recovered provider",model:"retry-model",successful:true,total_tokens:12}]};
    }
    return {requests:[]};
  });
  const dialog = panel.locator("#usage-request-dialog");
  await panel.locator('[data-usage-run-id="run-close"]').click();
  await expect(dialog).toHaveJSProperty("open",true);
  await expect(panel.locator("#usage-request-body")).toContainText("Loading");
  await panel.locator(".close-usage-requests").first().click();
  await expect(dialog).toHaveJSProperty("open",false);
  const generationAfterClose = await panel.evaluate((host) => host._usageRequestGeneration);

  await panel.locator('[data-usage-run-id="run-failed"]').click();
  await expect(dialog).toHaveJSProperty("open",true);
  await expect.poll(() => page.evaluate(() => Boolean(window.releaseRunFailed))).toBe(true);
  expect(await panel.evaluate((host) => host._usageRequestGeneration)).toBeGreaterThan(generationAfterClose);
  await page.evaluate(() => window.releaseRunClose());
  await expect(panel.locator("#usage-request-body")).toContainText("Loading");
  await page.evaluate(() => window.releaseRunFailed());
  await expect(panel.locator(".usage-request-card h3")).toContainText("ProviderTimeout");
  await expect(panel.locator(".usage-request-card")).toContainText("Nightly provider · gpt-nightly · responses");
  await expect(panel.locator(".usage-request-card")).toContainText("6.50 s");
  await expect(panel.locator(".usage-request-card")).toContainText("Web search");

  await panel.locator(".close-usage-requests").last().click();
  await panel.locator('[data-usage-run-id="run-empty"]').click();
  await expect(panel.locator("#usage-request-body")).toContainText("No retained provider-request details are available");
  await panel.locator(".close-usage-requests").last().click();

  await panel.locator('[data-usage-run-id="run-retry"]').click();
  await expect(panel.locator("#usage-request-body [role=alert]")).toContainText("Request details temporarily unavailable");
  await panel.locator(".close-usage-requests").last().click();
  await panel.locator('[data-usage-run-id="run-retry"]').click();
  await expect(panel.locator(".usage-request-card")).toContainText("Recovered provider · retry-model");
  expect(await page.evaluate(() => usageDetailsNightly.calls.filter((call) => call.action === "requests" && call.run_id === "run-retry").length)).toBe(2);
  await expectHarnessClean(page, errors);
});

test("nightly Clear Recent Details confirms, retries, deduplicates, and preserves aggregate history", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openUsage(page, async (message) => {
    if (message.action === "summary") return {today:{date:"2026-09-29",total_tokens:240},lifetime:{total_tokens:240},latest:{total_tokens:80}};
    if (message.action === "daily") return {days:[
      {date:"2026-09-28",total_tokens:100,input_tokens:80,output_tokens:20,run_count:1},
      {date:"2026-09-29",total_tokens:140,input_tokens:100,output_tokens:40,run_count:1},
    ]};
    if (message.action === "retention") return {request_days:30,run_days:90};
    if (message.action === "runs") return {runs:[{run_id:"retained-run",successful:false,error_type:"ProviderFailure"}]};
    if (message.action === "clear_details") {
      window.usageDetailsNightly.clearCount = (window.usageDetailsNightly.clearCount || 0) + 1;
      if (window.usageDetailsNightly.clearCount === 1) throw new Error("Details store unavailable");
      return new Promise((resolve) => { window.releaseClearDetails = () => resolve({deleted_runs:1,deleted_requests:2}); });
    }
    return undefined;
  });
  const clear = panel.locator("#clear-details");
  await expect(clear).toBeVisible();
  await panel.locator('.inline-route[data-subsection="retention"]').click();
  await expect(panel.locator('[data-config="usage_request_retention_days"]')).toBeVisible();
  await panel.locator('.top-nav button[data-page="usage-maintenance"]').click();
  await panel.locator('.subsection-nav button[data-subsection="usage"]').click();
  await expect(clear).toBeVisible();
  await panel.locator("#usage-window").selectOption("year");
  await expect(panel.locator("#usage-window")).toHaveValue("year");
  const chartBefore = await panel.locator(".chart-column").evaluateAll((columns) => columns.map((column) => column.getAttribute("aria-label")));
  const aggregateBefore = await panel.evaluate((host) => ({days:structuredClone(host._result.days),retention:structuredClone(host._result.retention),summary:structuredClone(host._result.summary)}));

  await clear.click();
  await expect(panel.locator("#confirm-dialog")).toHaveJSProperty("open",true);
  await panel.locator("#confirm-cancel").click();
  expect(await page.evaluate(() => usageDetailsNightly.calls.filter((call) => call.action === "clear_details").length)).toBe(0);

  await clear.click();
  await acceptConfirmation(panel);
  await expect(panel.locator("#toast")).toContainText("Details store unavailable");
  await expect(clear).toBeEnabled();

  await clear.click();
  await acceptConfirmation(panel);
  await expect.poll(() => page.evaluate(() => Boolean(window.releaseClearDetails))).toBe(true);
  await expect(clear).toBeDisabled();
  await clear.dispatchEvent("click");
  expect(await page.evaluate(() => usageDetailsNightly.calls.filter((call) => call.action === "clear_details").length)).toBe(2);
  await page.evaluate(() => window.releaseClearDetails());
  await expect(clear).toBeEnabled();
  const clearedRunState = await panel.evaluate((host) => host._result.runs);
  expect(clearedRunState).toMatchObject({runs:[],total:0});
  await expect(panel.locator("[data-eoc-usage-runs]")).toContainText("No retained recent runs");
  await expect(panel.locator("#usage-window")).toHaveValue("year");
  expect(await panel.locator(".chart-column").evaluateAll((columns) => columns.map((column) => column.getAttribute("aria-label")))).toEqual(chartBefore);
  const aggregateAfter = await panel.evaluate((host) => ({days:host._result.days,retention:host._result.retention,summary:{today:host._result.summary.today,lifetime:host._result.summary.lifetime}}));
  expect(aggregateAfter.days).toEqual(aggregateBefore.days);
  expect(aggregateAfter.retention).toEqual(aggregateBefore.retention);
  expect(aggregateAfter.summary).toEqual({today:aggregateBefore.summary.today,lifetime:aggregateBefore.summary.lifetime});
  await expectHarnessClean(page, errors);
});

test("nightly non-admin Usage omits detail clearing and retention controls", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("usage-maintenance/usage", "&admin=0"));
  const panel = panelFor(page);
  await expect(panel.getByText("Administrator permission is required for this section.")).toBeVisible();
  await expect(panel.locator("#clear-details")).toHaveCount(0);
  await expect(panel.locator('.inline-route[data-subsection="retention"]')).toHaveCount(0);
  await expectHarnessClean(page, errors);
});
