import {expect, test} from "@playwright/test";
import {fixtureUrl, trackPageErrors, expectHarnessClean} from "./browser-helpers.mjs";

test("Assistant parent introduction stays between subsection navigation and the active card", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("assistant/basics"));
  const panel = page.locator("extended-openai-management-panel");
  const intro = panel.locator("#eoc-assistant-intro-host .page-intro");
  await expect(intro.getByRole("heading", {name:"Assistant settings"})).toBeVisible();
  const firstIntro = await intro.evaluate(node => { window.__assistantIntro = node; return node.textContent; });
  expect(firstIntro).toContain("responds");
  const callsBefore = await page.evaluate(() => browserHarness.calls.filter(call => call.section === "configuration" && call.action === "get").length);
  for (const [subsection, cardHeading] of [["conversation", "Conversation"], ["voice", "Voice & identity"], ["basics", "General"]]) {
    await panel.locator(`.subsection-nav button[data-subsection="${subsection}"]`).click();
    await expect(panel.locator(".config-section-heading").getByText(cardHeading, {exact:true})).toBeVisible();
    await expect(panel.locator(".page-intro")).toHaveCount(1);
    expect(await panel.evaluate(host => {
      const root = host.shadowRoot;
      const introNode = root.querySelector("#eoc-assistant-intro-host .page-intro");
      return introNode === window.__assistantIntro
        && Boolean(root.querySelector(".subsection-nav").compareDocumentPosition(introNode) & Node.DOCUMENT_POSITION_FOLLOWING)
        && Boolean(introNode.compareDocumentPosition(root.querySelector("[data-eoc-main] .config-section-heading")) & Node.DOCUMENT_POSITION_FOLLOWING);
    })).toBe(true);
  }
  expect(await page.evaluate(() => browserHarness.calls.filter(call => call.section === "configuration" && call.action === "get").length)).toBe(callsBefore);
  await expectHarnessClean(page, errors);
});

test("Web search and retention use contextual page introductions without repeating navigation labels", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = page.locator("extended-openai-management-panel");
  await page.goto(fixtureUrl("capabilities/web-skills"));
  await expect(panel.locator("[data-eoc-main] > .page-intro").getByRole("heading", {name:"Web search & Skills"})).toBeVisible();
  await expect(panel.locator("#config-capabilities .config-section-heading")).toHaveCount(0);
  await page.goto(fixtureUrl("usage-maintenance/retention"));
  await expect(panel.locator("[data-eoc-main] > .page-intro").getByRole("heading", {name:"Usage data retention"})).toBeVisible();
  await expect(panel.locator("#config-retention .card-heading").getByRole("heading", {name:"Retention periods"})).toBeVisible();
  await expectHarnessClean(page, errors);
});

test("subsection navigation keeps its description accessible without a desktop tagline", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Request Rules", exact: true})).toBeVisible();
  await expect(panel.locator(".section-selector p")).toHaveCount(0);
  await expect(panel.locator("#local-section")).toHaveAttribute("aria-description", /local commands/);
  await expect(panel.locator(".section-selector")).toBeHidden();
  await page.setViewportSize({width: 390, height: 780});
  await expect(panel.locator("#local-section")).toBeVisible();
  await panel.locator("#local-section").selectOption("functions");
  await expect(page).toHaveURL(/\/capabilities\/functions$/);
  await expect(panel.getByRole("heading", {name: "Function Tools & Groups"})).toBeVisible();
  await expectHarnessClean(page, errors);
});

test("guide links belong to section introductions and open the matching topic", async ({page}) => {
  const errors = trackPageErrors(page);
  const cases = [
    ["data-memory/memory-settings", "memory", "Memory settings"],
    ["data-memory/memories", "memory", "Memories"],
    ["data-memory/knowledge", "knowledge", "Knowledge Library"],
    ["capabilities/functions", "functions", "Function Tools & Groups"],
  ];
  const panel = page.locator("extended-openai-management-panel");
  for (const [route, topic, heading] of cases) {
    await page.goto(fixtureUrl(route));
    await expect(panel.getByRole("heading", {name: heading, exact: true})).toBeVisible();
    const link = panel.locator(`.page-intro .guide-topic-link[data-guide-topic="${topic}"], .section-heading .guide-topic-link[data-guide-topic="${topic}"]`);
    await expect(link).toBeVisible();
    await expect(link).toHaveText("Learn more");
    await link.click();
    await expect(page).toHaveURL(/\/extended-openai\/guide$/);
    await expect(panel.locator(`#guide-${topic}`)).toHaveJSProperty("open", true);
  }
  await expectHarnessClean(page, errors);
});

test("Memory and Knowledge keep status inside their owning cards", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = page.locator("extended-openai-management-panel");
  await page.goto(fixtureUrl("data-memory/memories"));
  await expect(panel.locator("[data-persistent-memories] [data-memory-feature-status]")).toBeVisible();
  await expect(panel.locator(".feature-status-card")).toHaveCount(0);
  await panel.locator('.memory-kind[data-kind="temporary"]').click();
  await expect(panel.locator("[data-temporary-memories] .embedded-feature-status")).toBeVisible();
  const shortTerm = panel.locator("[data-temporary-feature-status]");
  await expect(shortTerm.locator(".status-value")).toHaveText("Off");
  await expect(shortTerm).toContainText("does not create new");
  await expect(panel.locator("[data-temporary-memories] > .help")).toContainText("expiry time");
  const reads = await page.evaluate(() => browserHarness.calls.length);
  await panel.evaluate(host => { host._selectedAgent().temporary_memory = "balanced"; host._render(); });
  await expect(shortTerm.locator(".status-value")).toHaveText("Balanced");
  await expect(shortTerm).toContainText("clearly relevant");
  await panel.evaluate(host => { host._selectedAgent().temporary_memory = "eager"; host._render(); });
  await expect(shortTerm.locator(".status-value")).toHaveText("Eager");
  await expect(shortTerm.locator(".status-value")).not.toHaveText("Expires automatically");
  expect(await page.evaluate(() => browserHarness.calls.length)).toBe(reads);
  await expect(panel.locator(".feature-status-card")).toHaveCount(0);
  await page.goto(fixtureUrl("data-memory/knowledge"));
  await expect(panel.locator("[data-knowledge-collection] #knowledge-status")).toBeVisible();
  await expect(panel.locator("[data-knowledge-collection] #knowledge-enabled-toggle")).toHaveCount(1);
  await expect(panel.locator(".feature-status-card")).toHaveCount(0);
  await expectHarnessClean(page, errors);
});

test("Request Rules keeps both create paths and a task-ordered layout", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator("#rule-search")).toBeVisible();
  await expect(panel.locator(".rule-toolbar .count")).toBeHidden();
  await expect(panel.locator("#rule-add")).toBeVisible();
  const order = await panel.locator(".rule-collection,.rule-settings,.rule-wording,.rule-test-tools,.rule-sharing").evaluateAll(
    (nodes) => nodes.map((node) => [...node.classList].find((name) => ["rule-collection", "rule-settings", "rule-wording", "rule-test-tools", "rule-sharing"].includes(name))),
  );
  expect(order).toEqual(["rule-collection", "rule-settings", "rule-wording", "rule-test-tools", "rule-sharing"]);
  await expect(panel.locator(".page-intro .guide-topic-link[data-guide-topic='request-rules']")).toBeVisible();
  await expect(panel.locator(".page-intro .rule-routing-help")).toHaveCount(0);
  await expect(panel.locator(".rule-settings").getByRole("heading", {name:"Matching defaults"})).toBeVisible();
  await panel.locator("#rule-search").fill("no match");
  await expect(panel.locator(".rule-toolbar .count")).toHaveText("Showing 0 of 1 rules");
  await expect(panel.locator(".rule-toolbar .count")).toBeVisible();
  await expect(panel.locator("[data-eoc-rule-search-empty]")).toBeVisible();
  await expect(panel.locator("#rule-search")).toBeFocused();
  await panel.locator("#rule-search").fill("");
  await expect(panel.locator('[data-rule-key="rule-1"]')).toBeVisible();
  await panel.locator(".rule-settings summary").first().click();
  await expect(panel.locator("#rules-default-word-forms")).toBeVisible();
  await panel.locator(".rule-wording summary").click();
  await expect(panel.locator("#wording-add")).toBeVisible();

  await page.evaluate(() => {
    const state = browserHarness.getState();
    state.requestRules.rules = [];
    localStorage.setItem("extended-openai-browser-harness-state-v3", JSON.stringify(state));
  });
  await page.goto(fixtureUrl("capabilities/request-rules"));
  await expect(panel.getByRole("heading", {name: "Create your first Request Rule"})).toBeVisible();
  await expect(panel.locator("#rule-search")).toBeHidden();
  await expect(panel.getByRole("button", {name: "Create rule", exact: true})).toHaveCount(2);
  await expect(panel.locator("#rules-title")).toHaveText("Rules (0)");
  await panel.locator("#rule-empty-add").click();
  await expect(panel.locator("#rule-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#rule-name").fill("First browser rule");
  await panel.locator("#rule-phrases").fill("hello browser");
  await panel.locator("#rule-action-type").selectOption("model_routing");
  await panel.locator("#rule-model").fill("gpt-5-mini");
  await panel.locator("#rule-save").click();
  await expect(panel.locator("#rule-dialog")).toHaveJSProperty("open", false);
  await expect(panel.locator("#rule-search")).toBeVisible();
  await expect(panel.locator("#rules-title")).toHaveText("Rules (1)");
  await expect(panel.locator(".rule-toolbar .count")).toBeHidden();
  await expect(panel.locator("#rule-add")).toBeVisible();
  await expectHarnessClean(page, errors);
});

test("rule testing distinguishes safe preview from confirmed live execution", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Test rules"})).toBeVisible();
  await expect(panel.locator("#rules-title")).toHaveText("Rules (1)");
  await expect(panel.locator("#rule-match-tester .notice")).toHaveCount(0);
  await expect(panel.locator("#rule-match-test-text")).toHaveAttribute("maxlength", "2048");
  await panel.locator("#rule-match-test-text").fill("baseline route");
  await panel.locator("#rule-match-test").click();
  await expect.poll(() => page.evaluate(() => browserHarness.calls.filter(call => call.section === "request_rules" && call.action === "test_match").length)).toBeGreaterThan(0);
  expect(await page.evaluate(() => browserHarness.calls.filter(call => call.section === "request_rules" && call.action === "test").length)).toBe(0);
  const live = panel.locator("#eoc-rule-live-test");
  await expect(live).not.toHaveJSProperty("open", true);
  await live.locator("summary").click();
  await expect(live.locator(".eoc-live-label")).toHaveText("Live · real effects possible");
  await live.locator("#eoc-rule-live-text").fill("baseline route");
  await expect(live.locator("#eoc-rule-live-text")).toHaveValue("baseline route");
  await live.locator("#eoc-rule-live-run").click();
  await expect(panel.locator("#confirm-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#confirm-cancel").click();
  expect(await page.evaluate(() => browserHarness.calls.filter(call => call.section === "request_rules" && call.action === "test").length)).toBe(0);
  await expectHarnessClean(page, errors);
});

test("Conversation history keeps scope selection and retained search under clear headings", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("data-memory/conversations"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Conversation history"})).toBeVisible();
  await expect(panel.locator('.scope-bar[aria-label="Conversation scope"]')).toBeVisible();
  await expect(panel.getByRole("heading", {name: "Retained conversations"})).toBeVisible();
  await panel.locator("#archive-query").fill("kitchen");
  await panel.locator("#archive-search").click();
  await expect.poll(() => page.evaluate(() => browserHarness.calls.filter(call => call.section === "conversations" && call.action === "search").length)).toBe(1);
  await expectHarnessClean(page, errors);
});
