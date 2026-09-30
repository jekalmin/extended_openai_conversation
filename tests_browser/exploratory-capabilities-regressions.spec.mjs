import {expect, test} from "@playwright/test";
import {acceptConfirmation, expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

const panelFor = page => page.locator("extended-openai-management-panel");

test("Knowledge reconciles disabled, source added, enabled and source availability in place", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("data-memory/knowledge"));
  const panel = panelFor(page);
  const toggle = panel.locator("#knowledge-enabled-toggle");
  const status = panel.locator("#knowledge-status .status-value");
  await toggle.uncheck();
  await expect(status).toHaveText("Off");
  await panel.locator("#add-source").click();
  await panel.locator("#knowledge-title").fill("Available reference");
  await panel.locator("#knowledge-content").fill("Reference information.");
  await panel.locator("#knowledge-save").click();
  await expect(panel.locator(".knowledge-source-availability-badge")).toHaveText("Available");
  await expect(status).toHaveText("Off");
  const before = await panel.evaluate(host => {
    window.knowledgeCard = host.shadowRoot.querySelector("[data-source-id]");
    window.knowledgeList = host.shadowRoot.querySelector(".knowledge-list");
    return browserHarness.calls.filter(call => call.action === "agents" || call.action === "list").length;
  });
  await toggle.check();
  await expect(status).toHaveText("Available");
  expect(await panel.evaluate(host => window.knowledgeCard === host.shadowRoot.querySelector("[data-source-id]") && window.knowledgeList === host.shadowRoot.querySelector(".knowledge-list"))).toBe(true);
  expect(await page.evaluate(() => browserHarness.calls.filter(call => call.action === "agents" || call.action === "list").length)).toBe(before);
  await panel.locator(".source-edit-button").click();
  await panel.locator("#knowledge-source-enabled").uncheck();
  await panel.locator("#knowledge-save").click();
  await expect(status).toHaveText("Needs sources");
  await panel.locator(".source-edit-button").click();
  await panel.locator("#knowledge-source-enabled").check();
  await panel.locator("#knowledge-save").click();
  await expect(status).toHaveText("Available");
  await expectHarnessClean(page, errors);
});

test("Request Rules distinguish an emptied group from a failed text search", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = panelFor(page);
  await panel.locator("#rule-groups-manage").click();
  await panel.locator("#rule-new-group-name").fill("Temporary group");
  await panel.locator("#rule-group-add").click();
  const group = await panel.evaluate(host => host._result.groups[0].id);
  await panel.locator("#rule-groups-done").click();
  await panel.locator(".rule-edit").click();
  await panel.locator("#rule-group").selectOption(group);
  await panel.locator("#rule-save").click();
  await expect(panel.locator("#rule-dialog")).not.toHaveJSProperty("open", true);
  await panel.locator(".rule-edit").click();
  await panel.locator("#rule-group").selectOption("");
  await panel.locator("#rule-save").click();
  await expect(panel.locator("#rule-dialog")).not.toHaveJSProperty("open", true);
  await panel.locator("#rule-group-filter").selectOption(group);
  const empty = panel.locator("[data-eoc-rule-search-empty]");
  await expect(empty).toHaveText("No rules in this group");
  await panel.locator("#rule-search").fill("uniquenonexistentterm");
  await expect(empty).toContainText("No rules match your search");
  await panel.locator("#rule-group-filter").selectOption("all");
  await expect(empty).toContainText("No rules match your search");
  await panel.locator("#rule-search").fill("");
  await expect(empty).toBeHidden();
  await expect(panel.locator(".request-rule-card")).toBeVisible();
  await panel.locator(".rule-delete").click();
  await acceptConfirmation(panel);
  await expect(panel.locator(".request-rule-card")).toHaveCount(0);
  await panel.locator("#rule-group-filter").selectOption(group);
  await expect(empty).toHaveText("No rules in this group");
  await expectHarnessClean(page, errors);
});

test("Function Tools search shows no results and clearing or editing restores cards", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/functions"));
  const panel = panelFor(page);
  const search = panel.getByRole("searchbox", {name:"Search functions and groups"});
  const empty = panel.locator("#function-search-empty");
  await search.fill("uniquenonexistentterm");
  await expect(empty).toContainText("No functions or groups match your search");
  await expect(panel.locator(".tool-card:visible")).toHaveCount(0);
  await panel.getByRole("button", {name:"Clear search", exact:true}).click();
  await expect(search).toHaveValue("");
  await expect(search).toBeFocused();
  await expect(empty).toBeHidden();
  await expect(panel.getByRole("heading", {name:"Baseline group", exact:true})).toBeVisible();
  await search.fill("uniquenonexistentterm");
  await search.fill("baseline");
  await expect(empty).toBeHidden();
  await expect(panel.locator(".tool-card")).toBeVisible();
  await expectHarnessClean(page, errors);
});

test("Speech preview clears a successful result when regex validation fails", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("assistant/speech"));
  const panel = panelFor(page);
  await expect(panel.locator("#preview-speech")).toBeVisible();
  await panel.evaluate(host => {
    const original = host._hass.callWS;
    host._hass.callWS = async message => {
      if (message.section === "configuration" && message.action === "speech_preview") {
        if (message.config.speech_regex_replacements.some(rule => rule.pattern === "[")) throw new Error("Invalid regular expression: unterminated character set");
        return {speech_text:"Current transformed speech"};
      }
      return original(message);
    };
  });
  await panel.locator("#config-speech_processing_enabled").check();
  await panel.locator("#speech-sample").fill("Sample response");
  await panel.locator("#preview-speech").click();
  await expect(panel.locator("#speech-output")).toHaveValue("Current transformed speech");
  await panel.locator("#add-regex").click();
  await panel.locator(".regex-pattern").fill("[");
  await panel.locator("#preview-speech").click();
  await expect(panel.locator("#toast")).toContainText("Invalid regular expression");
  await expect(panel.locator("#speech-output")).toHaveValue("");
  await expectHarnessClean(page, errors);
});

test("native Request Rule add-condition button has an accessible name and still opens its selector", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  // Model HA's nested Lit render boundary, including an icon-only add button.
  await page.evaluate(() => {
    class NativeButton extends HTMLElement {
      constructor() {
        super();
        this.attachShadow({mode:"open"}).innerHTML = '<button type="button">+</button>';
        this.shadowRoot.querySelector("button").addEventListener("click", () => { this.dataset.opened = "true"; });
      }
      get updateComplete() { return Promise.resolve(); }
    }
    class Conditions extends HTMLElement {
      constructor() { super(); this.attachShadow({mode:"open"}).innerHTML = '<div class="buttons"><ha-button></ha-button></div>'; }
      get updateComplete() { return Promise.resolve(); }
    }
    class ConditionSelector extends HTMLElement {
      constructor() { super(); this.attachShadow({mode:"open"}).innerHTML = '<ha-automation-condition></ha-automation-condition>'; }
      get updateComplete() { return Promise.resolve(); }
    }
    class Selector extends HTMLElement {
      constructor() { super(); this.attachShadow({mode:"open"}); }
      connectedCallback() {
        if (this.selector?.condition) {
          this.shadowRoot.innerHTML = '<ha-selector-condition></ha-selector-condition>';
          setTimeout(() => customElements.define("ha-selector-condition", ConditionSelector), 0);
        }
      }
      get updateComplete() { return Promise.resolve(); }
    }
    customElements.define("ha-button", NativeButton);
    customElements.define("ha-automation-condition", Conditions);
    customElements.define("ha-selector", Selector);
  });
  const panel = panelFor(page);
  await panel.getByRole("button", {name:"Create rule", exact:true}).first().click();
  await panel.getByRole("button", {name:"Add conditions — optional", exact:true}).click();
  const add = panel.locator("#rule-condition-host").getByRole("button", {name:"Add condition", exact:true});
  await expect(add).toBeVisible();
  await add.click();
  await expect(panel.locator("#rule-condition-host ha-button")).toHaveAttribute("data-opened", "true");
  await expectHarnessClean(page, errors);
});
