import {expect, test} from "@playwright/test";
import {acceptConfirmation, browserToolYaml, expectHarnessClean, trackPageErrors} from "./browser-helpers.mjs";
import {expectContractCalls} from "./real-ha-contract.mjs";

const backendUrl = process.env.REAL_HA_BACKEND_URL;
test.skip(!backendUrl, "requires the dedicated genuine Home Assistant backend bridge");
const realFixtureUrl = (route) => `/tests_browser/real-ha-fixture.html?route=${encodeURIComponent(route)}&backend=${encodeURIComponent(backendUrl)}`;

test("real browser saves General Settings through the genuine HA backend", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(realFixtureUrl("assistant/basics"));

  let panel = page.locator("extended-openai-management-panel");
  const title = panel.locator('[data-config="__title"]');
  await expect(title).toBeVisible();
  await expect(panel.locator('[data-config="chat_model"]')).toBeVisible();

  await title.fill("Browser Real HA Saved");
  await expect(panel.getByText("Unsaved changes", {exact: true})).toBeVisible();
  await panel.getByRole("button", {name: "Save changes", exact: true}).click();
  await expect(panel.getByText("Unsaved changes", {exact: true})).toHaveCount(0);

  await page.goto(realFixtureUrl("assistant/basics"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator('[data-config="__title"]')).toHaveValue("Browser Real HA Saved");
  await expect(panel.locator("#agent option:checked")).toHaveText("Browser Real HA Saved");

  const actions = await page.evaluate(() => window.browserHarness.calls
    .filter((call) => call.section === "configuration")
    .map((call) => call.action));
  expect(actions).toContain("get");
  await expectContractCalls(page, "configuration");
  await expectHarnessClean(page, pageErrors);
});

test("real browser creates, edits, reloads, and deletes a Memory through HA", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(realFixtureUrl("data-memory/memories"));

  let panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Memories", exact: true})).toBeVisible();
  await panel.locator("#add-memory").click();
  await expect(panel.locator("#memory-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#memory-content").fill("Real HA browser memory");
  await panel.locator("#memory-category").fill("browser-acceptance");
  await panel.locator("#memory-save").click();
  await expect(panel.getByText("Real HA browser memory", {exact: true})).toBeVisible();

  await page.goto(realFixtureUrl("data-memory/memories"));
  panel = page.locator("extended-openai-management-panel");
  let card = panel.locator(".list-card").filter({hasText: "Real HA browser memory"});
  await expect(card).toBeVisible();
  await card.locator(".memory-edit-button").click();
  await panel.locator("#memory-content").fill("Real HA browser memory edited");
  await panel.locator("#memory-category").fill("browser-acceptance-edited");
  await panel.locator("#memory-save").click();

  await page.goto(realFixtureUrl("data-memory/memories"));
  panel = page.locator("extended-openai-management-panel");
  card = panel.locator(".list-card").filter({hasText: "Real HA browser memory edited"});
  await expect(card).toContainText("browser-acceptance-edited");
  await card.locator(".delete-memory").click();
  await acceptConfirmation(panel);
  await expect(panel.getByText("Real HA browser memory edited", {exact: true})).toHaveCount(0);

  await page.goto(realFixtureUrl("data-memory/memories"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByText("Real HA browser memory edited", {exact: true})).toHaveCount(0);
  await expectContractCalls(page, "memory");
  await expectHarnessClean(page, pageErrors);
});

test("real browser creates, groups, edits, reloads, and deletes a Request Rule through HA", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(realFixtureUrl("capabilities/request-rules"));

  let panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Request Rules", exact: true})).toBeVisible();

  const groups = panel.locator(".rule-groups");
  await panel.locator("#rule-groups-manage").click();
  await groups.locator("#rule-new-group-name").fill("Real HA browser rule group");
  await groups.locator("#rule-group-add").click();
  let groupRow = groups.locator(".rule-group-row").last();
  await expect(groupRow.locator(".rule-group-name")).toHaveValue("Real HA browser rule group");
  await groups.locator("#rule-groups-done").click();

  await panel.getByRole("button", {name: "Create rule", exact: true}).first().click();
  await expect(panel.locator("#rule-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#rule-name").fill("Real HA browser rule");
  await panel.locator("#rule-phrases").fill("real browser route");
  await panel.locator("#rule-match").selectOption("contains");
  await panel.locator("#rule-action-type").selectOption("model_routing");
  await panel.locator("#rule-model").fill("gpt-5-mini");
  await panel.locator("#rule-reasoning").selectOption("medium");
  await panel.locator("#rule-scope").selectOption("conversation");
  await panel.locator("#rule-group").selectOption({label: "Real HA browser rule group"});
  const onlyWhen = [{condition:"template",value_template:"{{ true }}"}];
  await panel.locator("#rule-condition-host ha-selector").evaluate((selector, value) => {
    selector.value=value;
    selector.dispatchEvent(new CustomEvent("value-changed",{detail:{value},bubbles:true,composed:true}));
  }, onlyWhen);
  await panel.locator("#rule-save").click();
  await expect(panel.getByRole("heading", {name: "Real HA browser rule", exact: true})).toBeVisible();

  await page.goto(realFixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  await panel.locator("#rule-groups-manage").click();
  groupRow = panel.locator(".rule-group-row").last();
  await expect(groupRow.locator(".rule-group-name")).toHaveValue("Real HA browser rule group");
  await panel.locator("#rule-groups-done").click();
  let card = panel.locator(".request-rule-card").filter({hasText: "Real HA browser rule"});
  await expect(card).toBeVisible();
  await expect(card).toContainText("Real HA browser rule group");
  await card.locator(".rule-edit").click();
  expect(await panel.locator("#rule-condition-host ha-selector").evaluate((selector) => selector.value)).toEqual(onlyWhen);
  await panel.locator("#rule-condition-host ha-selector").evaluate((selector) => {
    selector.value=[];
    selector.dispatchEvent(new CustomEvent("value-changed", {detail: {value: []}, bubbles: true, composed: true}));
  });
  await panel.locator("#rule-name").fill("Real HA browser rule edited");
  await panel.locator("#rule-model").fill("gpt-5-nano");
  await panel.locator("#rule-save").click();

  await page.goto(realFixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  card = panel.locator(".request-rule-card").filter({hasText: "Real HA browser rule edited"});
  await expect(card).toContainText("gpt-5-nano");
  await panel.locator("#rule-match-test-text").fill("real browser route");
  await panel.locator("#rule-match-test").click();
  await expect(panel.locator("#rule-match-test-result")).toContainText("Real HA browser rule edited");
  await card.locator(".rule-duplicate").click();
  await expect(panel.locator(".request-rule-card")).toHaveCount(2);
  await panel.locator(".request-rule-card").last().locator('[data-direction="up"]').click();
  await expect(panel.locator(".request-rule-card")).toHaveCount(2);
  await panel.locator(".wording-editor summary").click();
  await panel.locator("#wording-add").click();
  await panel.locator(".wording-group").last().locator(".wording-canonical").fill("browser phrase");
  await panel.locator(".wording-group").last().locator(".wording-alternatives").fill("browser alternative");
  await panel.locator("#save-page").click();
  await expect(panel.locator(".save-bar")).toHaveCount(0);
  await page.goto(realFixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".wording-canonical").last()).toHaveValue("browser phrase");
  await panel.locator(".request-rule-card").last().locator(".rule-delete").click();
  await acceptConfirmation(panel);
  await expect(panel.locator(".request-rule-card")).toHaveCount(1);
  card = panel.locator(".request-rule-card");
  await card.locator(".rule-delete").click();
  await acceptConfirmation(panel);
  await expect(panel.getByRole("heading", {name: "Real HA browser rule edited", exact: true})).toHaveCount(0);

  const groupManager = panel.locator(".rule-groups");
  await panel.locator("#rule-groups-manage").click();
  groupRow = groupManager.locator(".rule-group-row").last();
  await expect(groupRow.locator(".rule-group-name")).toHaveValue("Real HA browser rule group");
  await groupRow.locator(".rule-group-name").fill("Real HA browser rule group edited");
  await groupRow.locator(".rule-group-rename").click();
  groupRow = groupManager.locator(".rule-group-row").last();
  await expect(groupRow.locator(".rule-group-name")).toHaveValue("Real HA browser rule group edited");
  await groupRow.locator(".rule-group-delete").click();
  await acceptConfirmation(panel);
  await expect(groupManager.locator(".rule-group-row")).toHaveCount(0);
  await groupManager.locator("#rule-groups-done").click();

  await page.goto(realFixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Real HA browser rule edited", exact: true})).toHaveCount(0);
  await panel.locator("#rule-groups-manage").click();
  await expect(panel.locator(".rule-group-row")).toHaveCount(0);
  await panel.locator("#rule-groups-done").click();
  await expectContractCalls(page, "request_rules");
  await expectHarnessClean(page, pageErrors);
});

test("real browser exports, reviews, and imports a Request Rule Pack through HA", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(realFixtureUrl("capabilities/request-rules"));
  let panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Request Rules", exact: true})).toBeVisible();
  await panel.getByRole("button", {name: "Create rule", exact: true}).first().click();
  await panel.locator("#rule-name").fill("Wire contract pack rule");
  await panel.locator("#rule-phrases").fill("wire contract pack phrase");
  await panel.locator("#rule-match").selectOption("contains");
  await panel.locator("#rule-action-type").selectOption("model_routing");
  await panel.locator("#rule-model").fill("gpt-5-mini");
  await panel.locator("#rule-reasoning").selectOption("medium");
  await panel.locator("#rule-scope").selectOption("request");
  await panel.locator("#rule-save").click();
  await expect(panel.locator(".request-rule-card")).toHaveCount(1);

  await panel.locator("#rule-sharing summary").click();
  const downloadPromise = page.waitForEvent("download");
  await panel.locator("#rule-pack-export").click();
  const download = await downloadPromise;
  const packPath = await download.path();
  expect(packPath).toBeTruthy();
  await expect(panel.locator("#rule-pack-message")).toContainText("1 rules exported");
  await panel.locator("#rule-pack-file").setInputFiles(packPath);
  await panel.locator("#rule-pack-review-button").click();
  await expect(panel.locator(".rule-pack-review")).toContainText("1 rules found");
  await panel.locator("#rule-pack-confirm").click();
  await expect(panel.locator("#rule-pack-message")).toContainText("disabled rules imported");
  await expect(panel.locator(".request-rule-card")).toHaveCount(2);

  await page.goto(realFixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".request-rule-card")).toHaveCount(2);
  for (let remaining = 2; remaining > 0; remaining--) {
    await panel.locator(".request-rule-card").last().locator(".rule-delete").click();
    await acceptConfirmation(panel);
    await expect(panel.locator(".request-rule-card")).toHaveCount(remaining - 1);
  }
  await expectContractCalls(page, "rule_pack");
  await expectHarnessClean(page, pageErrors);
});

test("real browser manages a Function Tool and dependent Group through genuine HA", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(realFixtureUrl("capabilities/functions"));

  let panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Function Tools & Groups", exact: true})).toBeVisible();

  await panel.locator("#add-tool").click();
  await expect(panel.locator("#tool-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#tool-yaml").fill(browserToolYaml("Real HA browser tool"));
  await panel.locator("#tool-save").click();
  let tool = panel.locator(".tool-card").filter({hasText: "browser_tool"});
  await expect(tool).toContainText("Real HA browser tool");

  await panel.locator("#add-group").click();
  await expect(panel.locator("#group-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#group-name").fill("Real HA browser group");
  await panel.locator("#group-id").fill("real-ha-browser-group");
  await panel.locator("#group-description").fill("Real HA browser group description");
  await panel.locator('#group-functions input[value="browser_tool"]').check();
  await panel.locator("#group-save").click();
  let group = panel.locator('.function-group-card[data-group-id="real-ha-browser-group"]');
  await expect(group).toContainText("Real HA browser group");

  await page.goto(realFixtureUrl("capabilities/functions"));
  panel = page.locator("extended-openai-management-panel");
  tool = panel.locator(".tool-card").filter({hasText: "browser_tool"});
  group = panel.locator('.function-group-card[data-group-id="real-ha-browser-group"]');
  await expect(tool).toContainText("Real HA browser tool");
  await expect(group).toContainText("Real HA browser group description");
  await group.locator("summary").click();
  await expect(group.locator(".tool-card").filter({hasText: "browser_tool"})).toBeVisible();

  await tool.locator(".edit-tool").click();
  await panel.locator("#tool-yaml").fill(browserToolYaml("Real HA browser tool edited"));
  await panel.locator("#tool-save").click();
  await expect(panel.locator("#tool-dialog")).toHaveJSProperty("open", false);
  await expect(tool).toContainText("Real HA browser tool edited");
  await group.locator(".edit-group").click();
  await panel.locator("#group-name").fill("Real HA browser group edited");
  await panel.locator("#group-description").fill("Real HA browser group description edited");
  await panel.locator("#group-save").click();

  await page.goto(realFixtureUrl("capabilities/functions"));
  panel = page.locator("extended-openai-management-panel");
  tool = panel.locator(".tool-card").filter({hasText: "browser_tool"});
  group = panel.locator('.function-group-card[data-group-id="real-ha-browser-group"]');
  await expect(tool).toContainText("Real HA browser tool edited");
  await expect(group).toContainText("Real HA browser group edited");
  await expect(group).toContainText("Real HA browser group description edited");

  const showGroupMembers = async () => {
    const details = group.locator("details").first();
    if (!await details.evaluate((element) => element.open)) await details.locator("summary").click();
  };
  await showGroupMembers();
  await tool.locator(".tool-enabled-control").click();
  await expect(tool.locator(".tool-enabled")).not.toBeChecked();
  await expect(panel.locator("#toast")).toContainText("Function disabled");
  await expect(tool.locator(".tool-enabled")).toBeEnabled();
  await showGroupMembers();
  await tool.locator(".tool-enabled-control").click();
  await expect(tool.locator(".tool-enabled")).toBeChecked();
  await expect(panel.locator("#toast")).toContainText("Function enabled");
  await expect(group.locator(".group-enabled")).toBeEnabled();
  await group.locator(".group-enabled-control").click();
  await expect(group.locator(".group-enabled")).not.toBeChecked();
  await expect(panel.locator("#toast")).toContainText("Function group disabled");
  await expect(group.locator(".group-enabled")).toBeEnabled();
  await group.locator(".group-enabled-control").click();
  await expect(group.locator(".group-enabled")).toBeChecked();
  await expect(panel.locator("#toast")).toContainText("Function group enabled");

  await page.goto(realFixtureUrl("capabilities/functions"));
  panel = page.locator("extended-openai-management-panel");
  tool = panel.locator(".tool-card").filter({hasText: "browser_tool"});
  group = panel.locator('.function-group-card[data-group-id="real-ha-browser-group"]');
  await expect(tool.locator(".tool-enabled")).toBeChecked();
  await expect(group.locator(".group-enabled")).toBeChecked();

  await group.locator(".delete-group").click();
  await acceptConfirmation(panel);
  await expect(panel.locator('.function-group-card[data-group-id="real-ha-browser-group"]')).toHaveCount(0);
  await tool.locator(".delete-tool").click();
  await acceptConfirmation(panel);
  await expect(panel.locator(".tool-card").filter({hasText: "browser_tool"})).toHaveCount(0);

  await page.goto(realFixtureUrl("capabilities/functions"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator('.function-group-card[data-group-id="real-ha-browser-group"]')).toHaveCount(0);
  await expect(panel.locator(".tool-card").filter({hasText: "browser_tool"})).toHaveCount(0);
  await expectContractCalls(page, "functions");
  await expectHarnessClean(page, pageErrors);
});

test("real browser creates, edits, and deletes Knowledge through genuine HA", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(realFixtureUrl("data-memory/knowledge"));
  let panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Sources", exact: true})).toBeVisible();
  await panel.locator("#add-source").click();
  await panel.locator("#knowledge-title").fill("Browser contract source");
  await panel.locator("#knowledge-content").fill("Knowledge payload crossed HA WebSocket validation");
  await panel.locator("#knowledge-save").click();
  await expect(panel.locator(".list-card").filter({hasText: "Browser contract source"})).toBeVisible();
  await panel.locator(".knowledge-availability-setting .switch-control").click();
  await expect(panel.locator("#knowledge-enabled-toggle")).toBeChecked();
  await expect(panel.locator("#toast")).toContainText("Knowledge enabled");

  await page.goto(realFixtureUrl("data-memory/knowledge"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator("#knowledge-enabled-toggle")).toBeChecked();
  let source = panel.locator(".list-card").filter({hasText: "Browser contract source"});
  await source.locator(".source-edit-button").click();
  await panel.locator("#knowledge-content").fill("Knowledge changed after authoritative reload");
  await panel.locator("#knowledge-save").click();
  await expect(panel.locator("#knowledge-dialog")).not.toBeVisible();

  await page.goto(realFixtureUrl("data-memory/knowledge"));
  panel = page.locator("extended-openai-management-panel");
  source = panel.locator(".list-card").filter({hasText: "Browser contract source"});
  await source.locator(".source-edit-button").click();
  await expect(panel.locator("#knowledge-content")).toHaveValue("Knowledge changed after authoritative reload");
  await panel.locator("#knowledge-dialog").getByRole("button", {name: "Close"}).click();
  await source.locator(".delete-source").click();
  await acceptConfirmation(panel);
  await expect(panel.locator(".list-card").filter({hasText: "Browser contract source"})).toHaveCount(0);
  await panel.locator(".knowledge-availability-setting .switch-control").click();
  await expect(panel.locator("#knowledge-enabled-toggle")).not.toBeChecked();
  await expect(panel.locator("#toast")).toContainText("Knowledge disabled");

  await page.goto(realFixtureUrl("data-memory/knowledge"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".list-card").filter({hasText: "Browser contract source"})).toHaveCount(0);
  await expectContractCalls(page, "knowledge");
  await expectHarnessClean(page, pageErrors);
});

test("real browser saves Guest and Quiet Hours policy payloads through HA", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(realFixtureUrl("capabilities/guest-mode"));
  let panel = page.locator("extended-openai-management-panel");
  const reviewLegacy = panel.locator("#guest-review-converted");
  await reviewLegacy.click();
  const knowledgePolicy = panel.locator('[data-guest-mode="guest_knowledge_policy"]');
  await expect(knowledgePolicy).toBeVisible();
  const guestValue = await knowledgePolicy.inputValue() === "on" ? "off" : "on";
  await knowledgePolicy.selectOption(guestValue);
  await panel.locator("#save-page").click();
  await expect(panel.locator(".save-bar")).toHaveCount(0);
  await page.goto(realFixtureUrl("capabilities/guest-mode"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator('[data-guest-mode="guest_knowledge_policy"]')).toHaveValue(guestValue);

  await page.goto(realFixtureUrl("capabilities/quiet-hours"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Quiet Hours", exact: true})).toBeVisible();
  const wakeValue = await panel.locator("#qh-wake").inputValue() === "unchanged" ? "off" : "unchanged";
  await panel.locator("#qh-wake").selectOption(wakeValue);
  await panel.locator("#save-page").click();
  await expect(panel.locator(".save-bar")).toHaveCount(0);
  await page.goto(realFixtureUrl("capabilities/quiet-hours"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator("#qh-wake")).toHaveValue(wakeValue);
  await expectContractCalls(page, "guest_quiet");
  await expectHarnessClean(page, pageErrors);
});

test("real browser full backup restores cross-feature state through genuine HA", async ({page}) => {
  const pageErrors = trackPageErrors(page);

  await page.goto(realFixtureUrl("assistant/basics"));
  let panel = page.locator("extended-openai-management-panel");
  const originalModel = await panel.locator('[data-config="chat_model"]').inputValue();
  await panel.locator('[data-config="__title"]').fill("Real HA backup source");
  await expect(panel.locator('[data-config="__title"]')).toHaveValue("Real HA backup source");
  await expect(panel.locator('[data-config="chat_model"]')).toHaveValue(originalModel);
  await panel.getByRole("button", {name: "Save changes", exact: true}).click();
  await expect(panel.getByText("Unsaved changes", {exact: true})).toHaveCount(0);

  await page.goto(realFixtureUrl("data-memory/memories"));
  panel = page.locator("extended-openai-management-panel");
  await panel.locator("#add-memory").click();
  await panel.locator("#memory-content").fill("Real HA backup memory");
  await panel.locator("#memory-category").fill("backup-acceptance");
  await panel.locator("#memory-save").click();
  await expect(panel.getByText("Real HA backup memory", {exact: true})).toBeVisible();

  await page.goto(realFixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  await panel.getByRole("button", {name: "Create rule", exact: true}).first().click();
  await panel.locator("#rule-name").fill("Real HA backup rule");
  await panel.locator("#rule-phrases").fill("real ha backup route");
  await panel.locator("#rule-match").selectOption("contains");
  await panel.locator("#rule-action-type").selectOption("model_routing");
  await panel.locator("#rule-model").fill("gpt-5-mini");
  await panel.locator("#rule-reasoning").selectOption("medium");
  await panel.locator("#rule-scope").selectOption("conversation");
  await panel.locator("#rule-save").click();
  await expect(panel.getByRole("heading", {name: "Real HA backup rule", exact: true})).toBeVisible();

  await page.goto(realFixtureUrl("usage-maintenance/backup-restore"));
  panel = page.locator("extended-openai-management-panel");
  await panel.locator("#transfer-export-mode").selectOption("full");
  const downloadPromise = page.waitForEvent("download");
  await panel.locator("#create-backup-transfer").click();
  const download = await downloadPromise;
  const backupPath = await download.path();
  expect(backupPath).toBeTruthy();

  await page.goto(realFixtureUrl("assistant/basics"));
  panel = page.locator("extended-openai-management-panel");
  await panel.locator('[data-config="__title"]').fill("Real HA backup mutated");
  await panel.getByRole("button", {name: "Save changes", exact: true}).click();
  await expect(panel.getByText("Unsaved changes", {exact: true})).toHaveCount(0);
  await page.goto(realFixtureUrl("assistant/basics"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator('[data-config="__title"]')).toHaveValue("Real HA backup mutated");

  await page.goto(realFixtureUrl("data-memory/memories"));
  panel = page.locator("extended-openai-management-panel");
  let card = panel.locator(".list-card").filter({hasText: "Real HA backup memory"});
  await card.locator(".delete-memory").click();
  await acceptConfirmation(panel);

  await page.goto(realFixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  card = panel.locator(".request-rule-card").filter({hasText: "Real HA backup rule"});
  await card.locator(".rule-delete").click();
  await acceptConfirmation(panel);

  await page.goto(realFixtureUrl("usage-maintenance/backup-restore"));
  panel = page.locator("extended-openai-management-panel");
  await panel.locator("#backup-file-transfer").setInputFiles(backupPath);
  await expect(panel.locator("#restore-dialog")).toHaveJSProperty("open", true);
  await expect(panel.locator("#restore-transfer-apply")).toBeEnabled();
  await panel.locator("#restore-transfer-apply").click();
  await acceptConfirmation(panel);
  await expect(panel.locator("#restore-dialog")).toHaveJSProperty("open", false);

  await page.goto(realFixtureUrl("assistant/basics"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator('[data-config="__title"]')).toHaveValue("Real HA backup source");

  await page.goto(realFixtureUrl("data-memory/memories"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByText("Real HA backup memory", {exact: true})).toBeVisible();

  await page.goto(realFixtureUrl("capabilities/request-rules"));
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Real HA backup rule", exact: true})).toBeVisible();
  await expectContractCalls(page, "backup");
  await expectHarnessClean(page, pageErrors);
});
