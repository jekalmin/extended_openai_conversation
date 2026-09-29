import {expect, test} from "@playwright/test";
import {acceptConfirmation, expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

const toolYaml = (name, description = name) => `spec:
  name: ${name}
  description: ${description}
  parameters:
    type: object
    properties: {}
function:
  type: script
  sequence: []
`;

async function addTool(panel, name) {
  await panel.locator("#add-tool").click();
  await expect(panel.locator("#tool-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#tool-yaml").fill(toolYaml(name, `${name} nightly probe`));
  await panel.locator("#tool-save").click();
  await expect(panel.locator(`[data-tool-key="${name}"]`)).toBeVisible();
}

async function createGroup(panel, {name, id, description = "Nightly Function Group", members = []}) {
  await panel.locator("#add-group").click();
  await expect(panel.locator("#group-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#group-name").fill(name);
  await panel.locator("#group-id").fill(id);
  await panel.locator("#group-description").fill(description);
  for (const member of members) {
    await panel.locator(`#group-functions input[value="${member}"]`).check();
  }
  await panel.locator("#group-save").click();
  await expect(panel.locator(`.function-group-card[data-group-id="${id}"]`)).toBeVisible();
}

async function gateFirstToolMutation(page, actions) {
  await page.evaluate((watched) => {
    const hass = browserHarness.hass;
    const original = hass.callWS.bind(hass);
    let release;
    const gate = new Promise(resolve => { release = resolve; });
    window.functionMutationGate = {started: [], release};
    hass.callWS = async message => {
      if (message.section === "tools" && watched.includes(message.action)) {
        window.functionMutationGate.started.push(structuredClone(message));
        if (window.functionMutationGate.started.length === 1) await gate;
      }
      return original(message);
    };
  }, actions);
}

test("nightly mixed Function and Group mutations preserve semantic state and fresh revisions", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/functions"));
  const panel = page.locator("extended-openai-management-panel");
  await addTool(panel, "queue_tool");

  const startRevision = await panel.evaluate(host => host._configData.revision);
  await gateFirstToolMutation(page, ["set_enabled", "save_group"]);

  const queueCard = panel.locator('[data-tool-key="queue_tool"]');
  await queueCard.locator(".tool-enabled").uncheck();
  await expect.poll(() => page.evaluate(() => functionMutationGate.started.length)).toBe(1);

  const groupToggle = panel.locator('.group-enabled[data-group-id="baseline-group"]');
  await groupToggle.uncheck();

  const assignment = queueCard.locator(".function-group-assignment");
  await assignment.selectOption("baseline-group");

  await page.waitForTimeout(50);
  expect(await page.evaluate(() => functionMutationGate.started.length)).toBe(1);

  await page.evaluate(() => functionMutationGate.release());
  await expect.poll(() => page.evaluate(() => functionMutationGate.started.length)).toBe(3);

  const calls = await page.evaluate(() => functionMutationGate.started.map(
    ({action, revision, group, name, enabled}) => ({
      action, revision, group:group ? structuredClone(group) : null, name, enabled,
    }),
  ));
  expect(calls.map(item => item.action)).toEqual(["set_enabled", "save_group", "save_group"]);
  expect(calls.map(item => item.revision)).toEqual([
    startRevision,
    `${startRevision}x`,
    `${startRevision}xx`,
  ]);
  expect(calls[1].group).toMatchObject({id:"baseline-group", enabled:false, functions:["baseline_tool"]});
  expect(calls[2].group).toMatchObject({
    id:"baseline-group",
    enabled:false,
    functions:["baseline_tool", "queue_tool"],
  });

  const group = panel.locator('.function-group-card[data-group-id="baseline-group"]');
  await expect(group).toHaveClass(/is-disabled/);
  await group.locator("summary").click();
  await expect(group.locator('[data-tool-key="queue_tool"]')).toBeVisible();
  await expect(group.locator('[data-tool-key="queue_tool"] .tool-enabled')).not.toBeChecked();

  expect(await panel.evaluate(host => ({
    backendTail:host._eocFunctionMutationTail,
    groupTail:host._eocFunctionGroupUiTail,
  }))).toEqual({backendTail:null, groupTail:null});
  await expectHarnessClean(page, errors);
});

test("nightly Function Group validation, failure recovery, rename and delete stay single-fire", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/functions"));
  const panel = page.locator("extended-openai-management-panel");

  await panel.locator("#add-group").click();
  const save = panel.locator("#group-save");
  await save.click();
  await expect(panel.locator("#group-error")).toContainText("Group name is required");

  await panel.locator("#group-name").fill("Nightly invalid");
  await panel.locator("#group-id").fill("1-invalid");
  await panel.locator("#group-description").fill("Validation probe");
  await save.click();
  await expect(panel.locator("#group-error")).toContainText("Group ID must start with a lowercase letter");

  await panel.locator("#group-id").fill("nightly-group");
  await panel.locator("#group-description").fill("");
  await save.click();
  await expect(panel.locator("#group-error")).toContainText("Add a concise description");

  await panel.locator("#group-description").fill("Recovered group");
  await panel.evaluate(host => {
    const original = host._call.bind(host);
    window.groupSaveFailures = 0;
    host._call = async (section, action, payload) => {
      if (section === "tools" && action === "save_group" && window.groupSaveFailures === 0) {
        window.groupSaveFailures += 1;
        throw new Error("Injected Function Group save failure");
      }
      return original(section, action, payload);
    };
  });
  await save.click();
  await expect(panel.locator("#group-dialog")).toHaveJSProperty("open", true);
  await expect(panel.locator("#group-error")).toContainText("Injected Function Group save failure");
  await expect(save).toBeEnabled();

  await save.click();
  await expect(panel.locator("#group-dialog")).not.toHaveJSProperty("open", true);
  await expect(panel.locator('.function-group-card[data-group-id="nightly-group"]')).toBeVisible();

  await panel.locator('.edit-group[data-group-id="nightly-group"]').click();
  await panel.locator("#group-name").fill("Nightly renamed");
  await panel.locator("#group-id").fill("nightly-renamed");
  await panel.locator("#group-description").fill("Renamed group");
  await panel.locator("#group-save").click();
  await expect(panel.locator('.function-group-card[data-group-id="nightly-renamed"] h3')).toHaveText("Nightly renamed");
  await expect(panel.locator('.function-group-card[data-group-id="nightly-group"]')).toHaveCount(0);

  await page.evaluate(() => {
    const hass = browserHarness.hass;
    const original = hass.callWS.bind(hass);
    let release;
    window.groupDeleteGate = {attempts:0, release:() => release?.()};
    hass.callWS = async message => {
      if (message.section === "tools" && message.action === "delete_group") {
        window.groupDeleteGate.attempts += 1;
        await new Promise(resolve => { release = resolve; });
      }
      return original(message);
    };
  });

  const group = panel.locator('.function-group-card[data-group-id="nightly-renamed"]');
  const remove = group.locator(".delete-group");
  await remove.click();
  await acceptConfirmation(panel);
  await expect.poll(() => page.evaluate(() => groupDeleteGate.attempts)).toBe(1);
  await expect(remove).toBeDisabled();
  await expect(remove).toHaveText("Deleting…");
  await remove.evaluate(button => {
    button.dispatchEvent(new MouseEvent("click", {bubbles:true, composed:true}));
    button.dispatchEvent(new MouseEvent("click", {bubbles:true, composed:true}));
  });
  expect(await page.evaluate(() => groupDeleteGate.attempts)).toBe(1);
  await page.evaluate(() => groupDeleteGate.release());
  await expect(group).toHaveCount(0);
  expect(await page.evaluate(() => groupDeleteGate.attempts)).toBe(1);
  await expectHarnessClean(page, errors);
});

test("nightly HA LLM catalogue handles scale, duplicate opening, filtering, add failure and retry", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/functions"));
  const panel = page.locator("extended-openai-management-panel");

  await panel.evaluate(host => {
    const hass = browserHarness.hass;
    const original = hass.callWS.bind(hass);
    window.haCatalogReads = 0;
    window.haAddAttempts = 0;
    window.haAddPayloads = [];
    const tools = Array.from({length:120}, (_, index) => ({
      name:`HA Tool ${index}`,
      source:index % 2 ? "Beta" : "Alpha",
      description:`Capability ${index}`,
      already_added:index === 0,
      reference:{
        type:"ha_llm",
        source_id:index % 2 ? "beta" : "alpha",
        api_id:"assist",
        tool_name:`ha_tool_${index}`,
      },
    }));
    hass.callWS = async message => {
      if (message.section === "tools" && message.action === "ha_catalog") {
        window.haCatalogReads += 1;
        await new Promise(resolve => setTimeout(resolve, 30));
        return {tools:structuredClone(tools), saved:{}, unavailable_sources:["Offline source"]};
      }
      if (message.section === "tools" && message.action === "ha_add") {
        window.haAddAttempts += 1;
        window.haAddPayloads.push(structuredClone(message));
        if (window.haAddAttempts === 1) throw new Error("Injected HA add failure");
        const additions = message.tools.map((reference, index) => ({
          spec:{name:`added_ha_${index}`, description:"Added HA tool", parameters:{type:"object",properties:{}}},
          function:structuredClone(reference),
          enabled:true,
        }));
        const groups = structuredClone(host._draft.function_groups || []);
        const target = groups.find(group => group.id === message.group_id);
        if (target) target.functions = [...new Set([...(target.functions || []), ...additions.map(tool => tool.spec.name)])];
        return {
          functions:[...structuredClone(host._draft.functions || []), ...additions],
          function_groups:groups,
          references:{},
          revision:`${message.revision}x`,
          ha_saved:Object.fromEntries(additions.map(tool => [tool.spec.name, {
            description:"Added HA tool", available:true, source:"Beta",
          }])),
        };
      }
      return original(message);
    };
  });

  const addButton = panel.locator("#add-ha-tools");
  await addButton.evaluate(button => {
    button.dispatchEvent(new MouseEvent("click", {bubbles:true, composed:true}));
    button.dispatchEvent(new MouseEvent("click", {bubbles:true, composed:true}));
  });
  const dialog = panel.locator('dialog[data-ha-llm-tools-dialog]');
  await expect(dialog).toHaveCount(1);
  await expect(dialog).toHaveJSProperty("open", true);
  await expect.poll(() => page.evaluate(() => haCatalogReads)).toBe(1);
  await expect(dialog.locator("[data-status]")).toContainText("Some sources are unavailable");
  await expect(dialog.locator("[data-tools] input")).toHaveCount(120);
  await expect(dialog.locator('[data-tools] input[data-index="0"]')).toBeDisabled();

  await dialog.locator("[data-search]").fill("HA Tool 11");
  await dialog.locator("[data-sources]").selectOption(["Beta"]);
  const visibleRows = dialog.locator("[data-tools] label:visible");
  await expect(visibleRows).not.toHaveCount(0);
  await dialog.locator("[data-all]").click();
  const selectedCount = await dialog.locator("[data-tools] input:checked").count();
  expect(selectedCount).toBeGreaterThan(0);
  await dialog.locator("[data-group]").selectOption("baseline-group");

  const addSelected = dialog.locator("[data-add]");
  await addSelected.click();
  await expect(dialog.locator("[data-status]")).toContainText("Injected HA add failure");
  await expect(addSelected).toBeEnabled();
  await addSelected.evaluate(button => {
    button.dispatchEvent(new MouseEvent("click", {bubbles:true, composed:true}));
    button.dispatchEvent(new MouseEvent("click", {bubbles:true, composed:true}));
  });
  await expect.poll(() => page.evaluate(() => haAddAttempts)).toBe(2);
  await expect(dialog).toHaveCount(0);

  const payloads = await page.evaluate(() => haAddPayloads);
  expect(payloads).toHaveLength(2);
  expect(payloads[1].group_id).toBe("baseline-group");
  expect(payloads[1].tools.length).toBe(selectedCount);
  expect(payloads[1].revision).toBeTruthy();
  const targetGroup = panel.locator('.function-group-card[data-group-id="baseline-group"]');
  await targetGroup.locator("summary").click();
  await expect(targetGroup.locator('[data-tool-key^="added_ha_"]').first()).toBeVisible();
  await expectHarnessClean(page, errors);
});

test("nightly HA LLM refresh is single-flight and latest catalogue state wins", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/functions"));
  const panel = page.locator("extended-openai-management-panel");

  await panel.evaluate(host => {
    const tool = host._draft.functions[0];
    tool.function = {type:"ha_llm", tool_name:"baseline_ha", source_id:"fixture", api_id:"assist"};
    host._haCatalog = {saved:{baseline_tool:{description:"Before refresh",available:true,source:"fixture"}}};
    host._haCatalogAgent = host._agentId;
    host._haCatalogLoadedAt = Date.now();
    host._render();

    const hass = browserHarness.hass;
    const original = hass.callWS.bind(hass);
    let release;
    window.refreshProbe = {reads:0, release:() => release?.()};
    hass.callWS = message => {
      if (message.section === "tools" && message.action === "ha_catalog") {
        window.refreshProbe.reads += 1;
        return new Promise(resolve => {
          release = () => resolve({
            tools:[],
            saved:{baseline_tool:{description:"After refresh",available:false,source:"fixture"}},
          });
        });
      }
      return original(message);
    };
  });

  const refresh = panel.locator("#refresh-ha-tools");
  await expect(refresh).toBeVisible();
  await expect.poll(() => panel.evaluate(host => host.shadowRoot.querySelector(".tools-surface")?.__eocHaBound === true)).toBe(true);
  await refresh.evaluate(button => {
    button.dispatchEvent(new MouseEvent("click", {bubbles:true, composed:true}));
    button.dispatchEvent(new MouseEvent("click", {bubbles:true, composed:true}));
  });
  await expect.poll(() => page.evaluate(() => refreshProbe.reads)).toBe(1);
  await expect(refresh).toBeDisabled();
  await page.evaluate(() => refreshProbe.release());
  await expect(panel.locator('[data-tool-key="baseline_tool"]')).toContainText("After refresh");
  await expect(panel.locator('[data-tool-key="baseline_tool"]')).toContainText("Unavailable");
  expect(await page.evaluate(() => refreshProbe.reads)).toBe(1);
  await expectHarnessClean(page, errors);
});
