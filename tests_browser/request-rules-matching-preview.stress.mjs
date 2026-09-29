import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

const panelFor = (page) => page.locator("extended-openai-management-panel");

test("nightly Request Rules defaults and wording edits persist, recover from failure, and control rule matching modes", async ({page}, testInfo) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  let panel = panelFor(page);
  await panel.locator(".rule-settings details").evaluate((details) => { details.open = true; });
  await panel.locator(".rule-wording details").evaluate((details) => { details.open = true; });

  await panel.locator("#rules-default-word-forms").uncheck();
  await panel.locator("#rules-default-wording").uncheck();
  await panel.locator("#rules-default-fuzzy").check();
  await expect(panel.locator("#rules-default-threshold")).toBeEnabled();
  await panel.locator("#rules-default-threshold").fill("100");
  await panel.locator("#rules-default-threshold").fill("70");
  await panel.locator("#rules-default-fuzzy").uncheck();
  await expect(panel.locator("#rules-default-threshold")).toBeDisabled();
  await panel.locator("#rules-default-fuzzy").check();
  await expect(panel.locator("#rules-default-threshold")).toHaveValue("70");
  await panel.locator("#wording-add").click();
  const wording = panel.locator(".wording-group").last();
  await wording.locator(".wording-canonical").fill("café");
  await wording.locator(".wording-alternatives").fill("coffee, café crème");

  await page.evaluate(() => {
    const panel = window.browserHarness.panel;
    const original = panel._call.bind(panel);
    let failed = false;
    panel._call = async (section, action, data) => {
      if (!failed && section === "request_rules" && action === "settings") {
        failed = true;
        throw new Error("Nightly injected settings failure");
      }
      return original(section, action, data);
    };
  });
  await panel.locator("#save-page").click();
  await expect(panel.locator("#toast")).toContainText("Nightly injected settings failure");
  await expect(panel.locator("#save-page")).toBeEnabled();
  await expect(wording.locator(".wording-canonical")).toHaveValue("café");
  await panel.locator("#save-page").click();
  await expect(panel.locator(".save-bar")).toHaveCount(0);

  await panel.locator(".request-rule-card").filter({hasText:"Baseline rule"}).locator(".rule-edit").click();
  await panel.locator("#rule-advanced summary").click();
  await expect(panel.locator("#rule-matching-behavior")).toHaveValue("defaults");
  await expect(panel.locator("#rule-matching-controls")).toBeHidden();
  await panel.locator(".rule-close").first().click();

  await page.goto(fixtureUrl("capabilities/request-rules"));
  panel = panelFor(page);
  await panel.locator(".rule-settings details").evaluate((details) => { details.open = true; });
  await expect(panel.locator("#rules-default-word-forms")).not.toBeChecked();
  await expect(panel.locator("#rules-default-wording")).not.toBeChecked();
  await expect(panel.locator("#rules-default-fuzzy")).toBeChecked();
  await expect(panel.locator("#rules-default-threshold")).toHaveValue("70");
  await expect(panel.locator(".wording-canonical").last()).toHaveValue("café");
  await expect(panel.locator(".wording-alternatives").last()).toHaveValue("coffee, café crème");

  await testInfo.attach("request-rule-matching-settings", {
    body:JSON.stringify({defaults:{word_forms:false,wording_alternatives:false,fuzzy:true,fuzzy_threshold:70},unicodeWordingPersisted:true,failedSaveRetried:true,freshLoadVerified:true}, null, 2),
    contentType:"application/json",
  });
  await expectHarnessClean(page, errors);
});

test("nightly Sentence Pattern helpers replace selections, insert at the caret, and edit empty fields", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = panelFor(page);
  await panel.getByRole("button", {name:"Create rule", exact:true}).first().click();
  await panel.locator("#rule-match").selectOption("sentence_pattern");
  const textarea = panel.locator("#rule-phrases");

  const helper = (kind) => panel.locator(`.pattern-helper[data-pattern-helper="${kind}"]`);
  for (const [kind, expected] of [["optional", "[optional words]"], ["choice", "(one|two)"], ["variable", "{name}"], ["range", "{level=0..100}"]]) {
    await textarea.fill("");
    await helper(kind).click();
    await expect(textarea).toHaveValue(expected);
  }

  await textarea.fill("please turn lights on");
  await textarea.evaluate((node) => { node.setSelectionRange(7, 11); node.dispatchEvent(new Event("select", {bubbles:true})); });
  await helper("optional").click();
  await expect(textarea).toHaveValue("please [turn] lights on");

  await textarea.fill("please one lights on");
  await textarea.evaluate((node) => node.setSelectionRange(7, 10));
  await helper("choice").click();
  await expect(textarea).toHaveValue("please (one|alternative) lights on");

  await textarea.fill("ask room");
  await textarea.evaluate((node) => node.setSelectionRange(4, 8));
  await helper("variable").click();
  await expect(textarea).toHaveValue("ask {room}");

  await textarea.fill("please set");
  await textarea.evaluate((node) => node.setSelectionRange(6, 6));
  await helper("optional").click();
  await expect(textarea).toHaveValue("please[optional words] set");

  await textarea.fill("set level");
  await textarea.evaluate((node) => node.setSelectionRange(4, 9));
  await helper("range").click();
  await expect(textarea).toHaveValue("set {level=0..100}");
  await expect(textarea).toHaveJSProperty("selectionStart", 5);
  await expect(textarea).toHaveJSProperty("selectionEnd", 10);
  await panel.locator(".rule-close").first().click();
  await expectHarnessClean(page, errors);
});

test("nightly Safe Preview presents match, captures, skipped conditions, no-match, and never starts a live request", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = panelFor(page);
  await expect(panel.locator("#rule-match-test-text")).toBeVisible();
  const responses = {
    "ask weather":{matched:true,rule:{name:"Preview capture",action_type:"model_routing",match_type:"sentence_pattern"},matched_phrase:"ask {question}",captured_values:{question:"weather"},would_do:{model:"gpt-5-mini",scope:"request"}},
    door:{matched:false,skipped_conditions:[{name:"Home is occupied"}]},
    unmatched:{matched:false},
  };
  await page.evaluate((responses) => {
    const panel = window.browserHarness.panel;
    const original = panel._call.bind(panel);
    const calls = window.browserHarness.previewCalls = [];
    panel._call = async (section, action, data) => {
      if (section === "request_rules" && action === "test_match") {
        calls.push({section, action, data});
        return responses[data.text];
      }
      calls.push({section, action, data});
      return original(section, action, data);
    };
  }, responses);

  for (const [text, expected] of [["ask weather", "Captured values"], ["door", "Only when conditions were false"], ["unmatched", "No Request Rule matched"]]) {
    await panel.locator("#rule-match-test-text").fill(text);
    await panel.locator("#rule-match-test").click();
    await expect(panel.locator("#rule-match-test-result")).toContainText(expected);
  }
  expect(await page.evaluate(() => window.browserHarness.previewCalls.filter((call) => call.action === "test").length)).toBe(0);
  expect(await page.evaluate(() => window.browserHarness.calls.filter((call) => call.section === "request_rules" && call.action === "test").length)).toBe(0);
  await expect(panel.locator("#eoc-rule-live-test")).not.toHaveJSProperty("open", true);
  await expectHarnessClean(page, errors);
});

const realBackendUrl = process.env.REAL_HA_BACKEND_URL;
const realFixtureUrl = (route) => `/tests_browser/real-ha-fixture.html?route=${encodeURIComponent(route)}&backend=${encodeURIComponent(realBackendUrl)}`;

test("nightly saved wording defaults change the real Safe Preview matcher result", async ({page}) => {
  test.skip(!realBackendUrl, "requires the dedicated genuine Home Assistant backend bridge");
  const errors = trackPageErrors(page);
  const unique = `Nightly matcher ${Date.now()}`;
  await page.goto(realFixtureUrl("capabilities/request-rules"));
  let panel = panelFor(page);
  await expect(panel.getByRole("heading", {name:"Request Rules", exact:true})).toBeVisible();
  const originalDefaults = await panel.evaluate((element) => structuredClone(element._result.defaults));
  const originalWordingCount = await panel.locator(".wording-group").count();
  let ruleCreated = false;
  let wordingCreated = false;

  const saveAndPreview = async ({wordForms = true, wording = true, text, expected}) => {
    await panel.locator(".rule-settings details").evaluate((details) => { details.open = true; });
    await panel.locator("#rules-default-word-forms").setChecked(wordForms);
    await panel.locator("#rules-default-wording").setChecked(wording);
    await panel.locator("#save-page").click();
    await expect(panel.locator(".save-bar")).toHaveCount(0);
    await panel.locator("#rule-match-test-text").fill(text);
    await panel.locator("#rule-match-test").click();
    await expect(panel.locator("#rule-match-test-result")).toContainText(expected);
  };

  try {
    await panel.locator(".rule-wording details").evaluate((details) => { details.open = true; });
    await panel.locator("#wording-add").click();
    const row = panel.locator(".wording-group").last();
    await row.locator(".wording-canonical").fill("activate");
    await row.locator(".wording-alternatives").fill("power on");
    wordingCreated = true;
    await panel.locator("#rules-default-word-forms").check();
    await panel.locator("#rules-default-wording").check();
    await panel.locator("#save-page").click();
    await expect(panel.locator(".save-bar")).toHaveCount(0);

    await panel.locator("#wording-add").click();
    let invalidRow = panel.locator(".wording-group").last();
    await invalidRow.locator(".wording-canonical").fill("   ");
    await invalidRow.locator(".wording-alternatives").fill("blank alternative");
    await panel.locator("#save-page").click();
    await expect(panel.locator("#toast")).toContainText("non-empty");
    await expect(panel.locator(".save-bar")).toBeVisible();
    await invalidRow.locator(".wording-remove").click();
    await panel.locator("#save-page").click();
    await expect(panel.locator(".save-bar")).toHaveCount(0);

    await panel.locator("#wording-add").click();
    invalidRow = panel.locator(".wording-group").last();
    await invalidRow.locator(".wording-canonical").fill("activate");
    await invalidRow.locator(".wording-alternatives").fill("switch on");
    await panel.locator("#save-page").click();
    await expect(panel.locator("#toast")).toContainText("duplicate phrase");
    await expect(panel.locator(".save-bar")).toBeVisible();
    await invalidRow.locator(".wording-remove").click();
    await panel.locator("#save-page").click();
    await expect(panel.locator(".save-bar")).toHaveCount(0);

    await panel.getByRole("button", {name:"Create rule", exact:true}).first().click();
    await panel.locator("#rule-name").fill(unique);
    await panel.locator("#rule-phrases").fill("activate nightly lamps");
    await panel.locator("#rule-match").selectOption("contains");
    await panel.locator("#rule-action-type").selectOption("model_routing");
    await panel.locator("#rule-model").fill("gpt-5-mini");
    await panel.locator("#rule-save").click();
    ruleCreated = true;

    await panel.locator("#rule-match-test-text").fill("activate nightly lamp");
    await panel.locator("#rule-match-test").click();
    await expect(panel.locator("#rule-match-test-result")).toContainText(unique);

    await saveAndPreview({wordForms:false, wording:true, text:"activate nightly lamp", expected:"No Request Rule matched"});
    await saveAndPreview({wordForms:true, wording:true, text:"power on nightly lamp", expected:unique});

    await page.goto(realFixtureUrl("capabilities/request-rules"));
    panel = panelFor(page);
    await expect(panel.locator(".request-rule-card").filter({hasText:unique})).toBeVisible();
    await saveAndPreview({wordForms:true, wording:false, text:"power on nightly lamp", expected:"No Request Rule matched"});
  } finally {
    try {
      if (ruleCreated) {
        const card = panel.locator(".request-rule-card").filter({hasText:unique});
        if (await card.count()) {
          await card.locator(".rule-delete").click();
          await panel.locator("#confirm-accept").click();
        }
      }
      if (wordingCreated) {
        await panel.locator(".rule-wording details").evaluate((details) => { details.open = true; });
        const rows = panel.locator(".wording-group");
        if (await rows.count() > originalWordingCount) await rows.last().locator(".wording-remove").click();
      }
      await panel.locator(".rule-settings details").evaluate((details) => { details.open = true; });
      await panel.locator("#rules-default-word-forms").setChecked(Boolean(originalDefaults.word_forms));
      await panel.locator("#rules-default-wording").setChecked(Boolean(originalDefaults.wording_alternatives));
      if (await panel.locator("#save-page").count()) await panel.locator("#save-page").click();
    } catch (cleanupError) {
      console.error(`Request Rule matcher cleanup failed: ${cleanupError.message}`);
      throw cleanupError;
    }
  }
  await expectHarnessClean(page, errors);
});
