import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

const panelFor = (page) => page.locator("extended-openai-management-panel");

async function openUsage(page, handler) {
  await page.clock.install({time: new Date("2026-09-29T12:00:00Z")});
  await page.goto(fixtureUrl("overview"));
  const panel = panelFor(page);
  await panel.locator(".dashboard-grid").waitFor();
  await panel.evaluate(async (host, callbackSource) => {
    const callback = Function(`return (${callbackSource})`)();
    const original = host._hass.callWS.bind(host._hass);
    window.usageNightly = {calls:[]};
    host._hass.callWS = async (message) => {
      if (message.section === "usage") {
        window.usageNightly.calls.push(structuredClone(message));
        const result = await callback(message, original);
        if (result !== undefined) return result;
      }
      return original(message);
    };
    await host._navigate("usage-maintenance", "usage");
  }, handler.toString());
  await expect(panel.locator("#usage-window")).toBeVisible();
  return panel;
}

test("nightly Usage history windows load lazily, cache selections, page all history, and keep charts responsive", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openUsage(page, async (message) => {
    if (message.action === "summary") return {today:{date:"2026-09-29",total_tokens:9999},lifetime:{total_tokens:9999}};
    if (message.action === "retention") return {request_days:30,run_days:30};
    if (message.action === "runs") return {runs:[]};
    if (message.action !== "daily") return undefined;
    if (message.start_date === "0000-01-01") {
      const start = new Date("2025-08-26T00:00:00Z");
      const firstIndex = message.start_date === "0000-01-01" ? 0 : 366;
      const count = firstIndex === 0 ? 366 : 34;
      const days = Array.from({length:count}, (_, index) => {
        const date = new Date(start.getTime() + (firstIndex + index) * 86400000).toISOString().slice(0,10);
        return {date,total_tokens:100 + index,cached_input_tokens:25};
      });
      return {days,has_more:firstIndex === 0};
    }
    if (message.start_date === "2026-09-23") return {days:[]};
    if (message.start_date === "2026-07-02") return {days:[
      {date:"2026-07-17",total_tokens:70,input_tokens:50,output_tokens:20,run_count:1},
      {date:"2026-09-29",total_tokens:90,input_tokens:60,output_tokens:30,run_count:1},
    ]};
    if (message.start_date === "2026-01-01") return {days:[
      {date:"2026-02-01",total_tokens:200,cached_input_tokens:20},
      {date:"2026-09-29",total_tokens:900,cached_input_tokens:100},
    ]};
    return {days:[
      {date:message.start_date,total_tokens:30,cached_input_tokens:5},
      {date:message.end_date,total_tokens:300,cached_input_tokens:30},
    ]};
  });

  const windowControl = panel.locator("#usage-window");
  expect(await windowControl.locator("option").allTextContents()).toEqual(["7 days","30 days","90 days","Year to date","All available"]);
  const initialCalls = await page.evaluate(() => usageNightly.calls.filter((call) => call.action === "daily"));
  expect(initialCalls.map(({start_date,end_date}) => [start_date,end_date])).toEqual([["2026-08-31","2026-09-29"]]);
  await expect(panel.locator(".chart-column")).toHaveCount(2);

  await windowControl.selectOption("7");
  await expect(panel.getByText("No daily usage is recorded in this period.")).toBeVisible();
  await expect(panel.locator(".chart-column")).toHaveCount(0);
  await expect(panel.locator(".usage-history-note")).toContainText("No recorded daily aggregates yet");

  await windowControl.selectOption("90");
  await expect(panel.locator(".usage-history-note")).toContainText("first stored daily aggregate is");
  await expect(panel.locator(".chart-column")).toHaveCount(2);
  await windowControl.selectOption("30");
  await windowControl.selectOption("90");
  expect(await page.evaluate(() => usageNightly.calls.filter((call) => call.action === "daily" && call.start_date === "2026-07-02").length)).toBe(1);

  await windowControl.selectOption("year");
  await expect(panel.getByRole("heading", {name:"Tokens by month"})).toBeVisible();
  await expect(panel.locator(".chart-column")).toHaveCount(2);
  await expect(panel.locator(".chart-axis span").first()).toContainText("Feb");

  await windowControl.selectOption("all");
  await expect(panel.locator(".usage-history-note")).toContainText("Aug 26, 2025 to Sep 29, 2026");
  await expect(panel.locator(".chart-column")).toHaveCount(14);
  const allCalls = await page.evaluate(() => usageNightly.calls.filter((call) => call.action === "daily").slice(-2));
  expect(allCalls.map(({start_date,end_date}) => [start_date,end_date])).toEqual([
    ["0000-01-01","2026-09-29"],
    ["2026-08-27","2026-09-29"],
  ]);
  await page.setViewportSize({width:390,height:844});
  await expect(panel.locator(".usage-range-card")).toHaveCSS("display","grid");
  const responsiveChart = await panel.locator(".chart").evaluate((element) => ({clientWidth:element.clientWidth,scrollWidth:element.scrollWidth,columns:element.children.length}));
  expect(responsiveChart.columns).toBe(14);
  expect(responsiveChart.scrollWidth).toBeLessThanOrEqual(responsiveChart.clientWidth);
  expect(await page.evaluate(() => usageNightly.calls.filter((call) => call.action === "daily").length)).toBe(6);
  await expectHarnessClean(page, errors);
});

test("nightly Usage window waits for pending history before accepting another selection", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await openUsage(page, async (message) => {
    if (message.action === "summary") return {today:{date:"2026-09-29",total_tokens:9999},lifetime:{total_tokens:9999}};
    if (message.action === "retention") return {};
    if (message.action === "runs") return {runs:[]};
    if (message.action !== "daily") return undefined;
    if (message.start_date === "2026-09-23") return new Promise((resolve) => { window.releaseUsage7 = () => resolve({days:[{date:"2026-09-23",total_tokens:7}]}); });
    if (message.start_date === "2026-07-02") return new Promise((resolve) => { window.releaseUsage90 = () => resolve({days:[{date:"2026-07-02",total_tokens:90}]}); });
    return {days:[{date:"2026-08-31",total_tokens:30}]};
  });
  const select = panel.locator("#usage-window");
  await select.selectOption("7");
  await expect.poll(() => page.evaluate(() => Boolean(window.releaseUsage7))).toBe(true);
  await expect(select).toBeDisabled();
  await page.evaluate(() => window.releaseUsage7());
  await expect(select).toBeEnabled();
  await expect(panel.locator(".chart-column")).toHaveAttribute("aria-label", /7 total/);

  await select.selectOption("90");
  await expect.poll(() => page.evaluate(() => Boolean(window.releaseUsage90))).toBe(true);
  await page.evaluate(() => window.releaseUsage90());
  await expect(select).toHaveValue("90");
  await expect(panel.locator(".chart-column")).toHaveAttribute("aria-label", /90 total/);
  expect(await panel.evaluate((host) => host._usageHistoryWindow)).toBe("90");
  await expectHarnessClean(page, errors);
});
