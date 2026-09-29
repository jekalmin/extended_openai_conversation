import {expect, test} from "@playwright/test";
import {readFile} from "node:fs/promises";
import {acceptConfirmation, expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

const panelFor = (page) => page.locator("extended-openai-management-panel");
const RULE_PACK_FORMAT = "extended_openai_request_rule_pack";

function rulePack(names) {
  return {
    format:RULE_PACK_FORMAT,
    version:1,
    groups:[],
    rules:names.map((name, order) => ({
      id:`portable-${order}`,
      name,
      enabled:true,
      phrases:[`phrase ${order}`],
      match_type:"equals",
      action_type:"model_routing",
      action:{model:"gpt-5-mini",reasoning_effort:"",scope:"request",reset:false,continue_to_ai:true,success_response:"Updated"},
      matching_behavior:"custom",
      matching:{word_forms:true,wording_alternatives:true,fuzzy:false,fuzzy_threshold:90},
      order,
      conditions:[],
      group_id:null,
      continue_matching:order === 0,
      ai_input_mode:"original",
      ai_input_capture:null,
    })),
  };
}

async function choosePack(panel, pack, name = "nightly-pack.json") {
  await panel.locator("#rule-pack-file").setInputFiles({
    name,
    mimeType:"application/json",
    buffer:Buffer.from(typeof pack === "string" ? pack : JSON.stringify(pack)),
  });
  await panel.locator("#rule-pack-review-button").click();
}

test("nightly Rule Pack UI rejects malformed and oversized files, preserves review cancellation, retries failures, and round-trips priority", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = panelFor(page);
  const sharing = panel.locator("#rule-sharing");
  if (!(await sharing.evaluate((element) => element.open))) await sharing.locator("summary").click();
  await expect(panel.locator("#rule-pack-file")).toBeVisible();

  await choosePack(panel, "{not-json");
  await expect(panel.locator("#rule-pack-message")).toContainText("JSON");
  await choosePack(panel, {format:"unsupported",version:99,groups:[],rules:[]}, "unsupported.json");
  await expect(panel.locator("#rule-pack-message")).toContainText("Invalid rule pack");

  const reviewCallsBeforeLargeFile = await page.evaluate(() => window.browserHarness.calls.filter((call) => call.action === "rule_pack_review").length);
  await choosePack(panel, JSON.stringify({format:RULE_PACK_FORMAT,version:1,groups:[],rules:[],padding:"x".repeat(2*1024*1024)}), "large-pack.json");
  await expect(panel.locator("#rule-pack-message")).toContainText("2 MB");
  expect(await page.evaluate(() => window.browserHarness.calls.filter((call) => call.action === "rule_pack_review").length)).toBe(reviewCallsBeforeLargeFile);

  const pack = rulePack(["Nightly shared second", "Nightly shared first"]);
  await choosePack(panel, pack);
  await expect(panel.getByRole("heading", {name:"Review Rule Pack"})).toBeVisible();
  await expect(panel.locator(".request-rule-card").filter({hasText:"Nightly shared second"})).toHaveCount(0);
  await panel.locator("#rule-pack-cancel").click();
  expect(await page.evaluate(() => window.browserHarness.calls.filter((call) => call.action === "rule_pack_import").length)).toBe(0);
  await expect(panel.locator(".rule-pack-review")).toHaveCount(0);

  await choosePack(panel, pack);
  await page.evaluate(() => {
    const element = window.browserHarness.panel;
    const original = element._call.bind(element);
    let failed = false;
    element._call = async (section, action, data) => {
      if (!failed && section === "request_rules" && action === "rule_pack_import") {
        failed = true;
        throw new Error("Nightly import failure");
      }
      return original(section, action, data);
    };
  });
  await panel.locator("#rule-pack-confirm").click();
  await expect(panel.locator("#rule-pack-message")).toContainText("Nightly import failure");
  await expect(panel.locator("#rule-pack-confirm")).toBeEnabled();
  await panel.locator("#rule-pack-confirm").click();
  await expect(panel.locator("#rule-pack-message")).toContainText("2 disabled rules imported");

  const titles = await panel.locator(".request-rule-card h2").allTextContents();
  expect(titles.slice(-2)).toEqual(["Nightly shared second", "Nightly shared first"]);
  await expect(panel.locator(".request-rule-card").filter({hasText:"Nightly shared second"}).locator(".rule-enabled")).not.toBeChecked();
  const imported = await page.evaluate(() => window.browserHarness.getState().requestRules.rules.slice(-2));
  expect(imported.map((rule) => rule.order)).toEqual([1,2]);
  expect(imported.map((rule) => rule.continue_matching)).toEqual([true,false]);

  await panel.locator("#rule-pack-selection").selectOption("selected");
  const exportsBeforeCancel = await page.evaluate(() => window.browserHarness.calls.filter((call) => call.action === "rule_pack_export").length);
  await panel.locator("#rule-pack-export").click();
  await expect(panel.locator("#rule-pack-select-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#rule-pack-select-cancel").click();
  expect(await page.evaluate(() => window.browserHarness.calls.filter((call) => call.action === "rule_pack_export").length)).toBe(exportsBeforeCancel);

  await panel.locator("#rule-pack-export").click();
  await panel.locator("#rule-pack-rule-list label").filter({hasText:"Nightly shared first"}).locator("input").check();
  const downloadPromise = page.waitForEvent("download");
  await panel.locator("#rule-pack-select-done").click();
  const download = await downloadPromise;
  const exported = JSON.parse(await readFile(await download.path(), "utf8"));
  expect(exported.format).toBe(RULE_PACK_FORMAT);
  expect(exported.rules.map((rule) => rule.name)).toEqual(["Nightly shared first"]);
  await expect(panel.locator("#rule-pack-message")).toContainText("1 rules exported");
  await expectHarnessClean(page, errors);
});

test("nightly Live Request confirmation, repeated activation, results, errors, stale navigation, and preview independence", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = panelFor(page);
  const live = panel.locator("#eoc-rule-live-test");
  await live.locator("summary").click();
  const state = await page.evaluate(() => {
    const panel = window.browserHarness.panel;
    const original = panel._call.bind(panel);
    let releaseLocal;
    let releaseLate;
    const localPending = new Promise((resolve) => { releaseLocal = resolve; });
    const latePending = new Promise((resolve) => { releaseLate = resolve; });
    const calls = window.nightlyLiveCalls = [];
    panel._call = async (section, action, data) => {
      calls.push({section,action,text:data?.text});
      if (section !== "request_rules") return original(section,action,data);
      if (action === "test_match") return {matched:false};
      if (action !== "test") return original(section,action,data);
      if (data.text === "local success") return localPending;
      if (data.text === "late stale") return latePending;
      if (data.text === "AI success") return {response:"Provider replied",handled_locally:false,conversation_id:"nightly-conversation",matched_rule:null};
      if (data.text === "provider failure") throw new Error("Provider request failed");
      return {response:"Done",handled_locally:true,conversation_id:"nightly-local",matched_rule:{name:"Nightly local action"}};
    };
    window.releaseNightlyLocal = () => releaseLocal({response:"Local action completed",handled_locally:true,conversation_id:"nightly-local",matched_rule:{name:"Nightly local action"}});
    window.releaseNightlyLate = () => releaseLate({response:"Late result",handled_locally:true,conversation_id:"nightly-late",matched_rule:{name:"Nightly late action"}});
    return true;
  });
  expect(state).toBe(true);

  await panel.locator("#eoc-rule-live-text").fill("cancel this");
  await panel.locator("#eoc-rule-live-run").click();
  await expect(panel.locator("#confirm-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#confirm-cancel").click();
  expect(await page.evaluate(() => window.nightlyLiveCalls.filter((call) => call.action === "test").length)).toBe(0);

  await panel.locator("#rule-match-test-text").fill("safe preview first");
  await panel.locator("#rule-match-test").click();
  await expect(panel.locator("#rule-match-test-result")).toContainText("No Request Rule matched");
  const safeResult = await panel.locator("#rule-match-test-result").evaluate((element) => element.textContent);

  await panel.locator("#eoc-rule-live-text").fill("local success");
  await panel.locator("#eoc-rule-live-run").click();
  await acceptConfirmation(panel);
  await expect(panel.locator("#eoc-rule-live-run")).toBeDisabled();
  await panel.locator("#eoc-rule-live-run").dispatchEvent("click");
  expect(await page.evaluate(() => window.nightlyLiveCalls.filter((call) => call.action === "test").length)).toBe(1);
  await page.evaluate(() => window.releaseNightlyLocal());
  await expect(panel.locator("#eoc-rule-live-result")).toContainText("Local action completed");
  await expect(panel.locator("#eoc-rule-live-result")).toContainText("Handled locally");
  await expect(panel.locator("#rule-match-test-result")).toHaveJSProperty("textContent", safeResult);

  await panel.locator("#eoc-rule-live-text").fill("AI success");
  await panel.locator("#eoc-rule-live-run").click();
  await acceptConfirmation(panel);
  await expect(panel.locator("#eoc-rule-live-result")).toContainText("Provider replied");
  await expect(panel.locator("#eoc-rule-live-result")).toContainText("AI provider");

  await panel.locator("#eoc-rule-live-text").fill("provider failure");
  await panel.locator("#eoc-rule-live-run").click();
  await acceptConfirmation(panel);
  await expect(panel.locator("#eoc-rule-live-result")).toContainText("Provider request failed");
  await expect(panel.locator("#eoc-rule-live-run")).toBeEnabled();

  await panel.locator("#eoc-rule-live-text").fill("late stale");
  await panel.locator("#eoc-rule-live-run").click();
  await acceptConfirmation(panel);
  await expect.poll(() => page.evaluate(() => window.nightlyLiveCalls.filter((call) => call.text === "late stale").length)).toBe(1);
  await panel.locator('.top-nav button[data-page="overview"]').click();
  await expect(panel.locator('.top-nav button[data-page="overview"]')).toHaveAttribute("aria-current", "page");
  await page.evaluate(() => window.releaseNightlyLate());
  expect(await page.evaluate(() => window.nightlyLiveCalls.filter((call) => call.action === "test").length)).toBe(4);
  await expectHarnessClean(page, errors);
});
