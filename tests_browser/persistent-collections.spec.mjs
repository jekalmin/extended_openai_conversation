import {test, expect} from "@playwright/test";
import {acceptConfirmation, expectHarnessClean, trackPageErrors} from "./browser-helpers.mjs";
import {openCollection, beginCollectionMeasure, finishCollectionMeasure} from "./collection-helpers.mjs";

const tool = (panel, name) => panel.locator(`[data-tool-key="${name}"]`);
const rule = (panel, id) => panel.locator(`[data-rule-key="${id}"]`);
async function remember(page, selector) {
  await page.evaluate(selector => {
    const root = browserHarness.panel.shadowRoot;
    window.retainedNode = root.querySelector(selector);
    window.retainedSearch = root.querySelector("#tool-search,#rule-search");
    window.retainedMain = root.querySelector("main");
  }, selector);
}
async function expectRetained(page, selector) {
  expect(await page.evaluate(selector => {
    const root = browserHarness.panel.shadowRoot;
    return retainedNode === root.querySelector(selector) && retainedSearch === root.querySelector("#tool-search,#rule-search") && retainedMain === root.querySelector("main");
  }, selector)).toBe(true);
}
async function calls(page, section, action) {
  return page.evaluate(([section, action]) => browserHarness.calls.filter(c => c.section === section && c.action === action), [section, action]);
}

for (const surface of ["functions", "request-rules"]) {
  test(`${surface}: toggle changes only its card and repeated updates stay single-fire`, async ({page}) => {
    const diagnostics = trackPageErrors(page);
    const panel = await openCollection(page, `capabilities/${surface}`);
    const isTools = surface === "functions";
    const selector = isTools ? '[data-tool-key="tool_1"]' : '[data-rule-key="rule-2"]';
    const input = isTools ? tool(panel, "tool_0").locator(".tool-enabled") : rule(panel, "rule-1").locator(".rule-enabled");
    await remember(page, selector);
    for (let i = 0; i < 4; i++) {
      await beginCollectionMeasure(page);
      await input.setChecked(i % 2 !== 0);
      await expect(input).toBeEnabled();
      await expect(input).toBeChecked({checked: i % 2 !== 0});
      const result = await finishCollectionMeasure(page);
      expect(result.mainChildReplacements).toBe(0);
      expect(result.retainedCards).toBe(39);
      if (isTools) expect(result.retainedGroups).toBe(5);
      expect(result.elementChanges).toBeLessThan(100);
      await expectRetained(page, selector);
    }
    const mutations = await calls(page, isTools ? "tools" : "request_rules", isTools ? "set_enabled" : "update");
    expect(mutations).toHaveLength(4);
    if (!isTools) expect(mutations.map(c => c.revision)).toEqual([3, 4, 5, 6]);
    await expectHarnessClean(page, diagnostics);
  });

  test(`${surface}: search preserves input identity, focus and visibility without rendering`, async ({page}) => {
    const panel = await openCollection(page, `capabilities/${surface}`);
    const isTools = surface === "functions";
    const selector = isTools ? '[data-tool-key="tool_1"]' : '[data-rule-key="rule-2"]';
    const search = panel.locator(isTools ? "#tool-search" : "#rule-search");
    await remember(page, selector);
    await beginCollectionMeasure(page);
    await search.fill("not found anywhere");
    await expect(panel.locator(`${isTools ? ".tool-card" : ".request-rule-card"}:visible`)).toHaveCount(0);
    await expect(search).toBeFocused();
    await search.fill("fixture");
    await expect(panel.locator(`${isTools ? ".tool-card" : ".request-rule-card"}:visible`)).toHaveCount(40);
    await search.fill("");
    await expect(search).toBeFocused();
    const result = await finishCollectionMeasure(page);
    expect(result.renders).toBe(0);
    expect(result.retainedCards).toBe(40);
    await expectRetained(page, selector);
  });
}

test("Functions: group membership, metadata, deletion and counts retain unrelated and moved cards", async ({page}) => {
  const panel = await openCollection(page, "capabilities/functions");
  await remember(page, '[data-tool-key="tool_0"]');
  const select = tool(panel, "tool_0").locator(".function-group-assignment");
  await select.selectOption("group-1");
  await expect(panel.locator('[data-group-id="group-1"].function-group-card [data-tool-key="tool_0"]')).toHaveCount(1);
  await expect(panel.locator('[data-group-id="group-0"].function-group-card .function-count')).toHaveText("9 functions");
  await expect(panel.locator('[data-group-id="group-1"].function-group-card .function-count')).toHaveText("11 functions");
  await expectRetained(page, '[data-tool-key="tool_0"]');
  await panel.locator('.group-enabled[data-group-id="group-2"]').uncheck();
  await expect(panel.locator('[data-group-id="group-2"].function-group-card')).toHaveClass(/is-disabled/);
  await expectRetained(page, '[data-tool-key="tool_0"]');
  await panel.locator('.delete-group[data-group-id="group-1"]').click();
  await acceptConfirmation(panel);
  await expect(panel.locator('.always-card [data-tool-key="tool_0"]')).toHaveCount(1);
  await expectRetained(page, '[data-tool-key="tool_0"]');
  expect(await calls(page, "tools", "save_group")).toHaveLength(2);
  expect(await calls(page, "tools", "delete_group")).toHaveLength(1);
});

test("Functions: edit YAML and delete preserve unrelated cards, search and current-index dispatch", async ({page}) => {
  const panel = await openCollection(page, "capabilities/functions");
  await remember(page, '[data-tool-key="tool_1"]');
  await panel.locator("#tool-search").fill("fixture");
  await tool(panel, "tool_0").locator(".edit-tool").click();
  await expect(panel.locator("#tool-yaml")).toHaveValue(/Fixture function 0/);
  await panel.locator("#tool-yaml").fill("spec:\n  name: tool_0\n  description: Changed fixture tool\nfunction:\n  type: script\n  sequence: []\n");
  await panel.locator("#tool-save").click();
  await expect(tool(panel, "tool_0")).toContainText("Changed fixture tool");
  await expectRetained(page, '[data-tool-key="tool_1"]');
  await expect(panel.locator("#tool-search")).toHaveValue("fixture");
  await tool(panel, "tool_0").locator(".delete-tool").click();
  await acceptConfirmation(panel);
  await expect(tool(panel, "tool_0")).toHaveCount(0);
  await expectRetained(page, '[data-tool-key="tool_1"]');
  await tool(panel, "tool_1").locator(".tool-enabled").uncheck();
  await expect(tool(panel, "tool_1")).toHaveClass(/is-disabled/);
  expect((await calls(page, "tools", "set_enabled"))[0].name).toBe("tool_1");
});

test("Request Rules: edit/save, delete and create retain unrelated cards and filtered count", async ({page}) => {
  const panel = await openCollection(page, "capabilities/request-rules");
  await remember(page, '[data-rule-key="rule-2"]');
  await panel.locator("#rule-search").fill("fixture rule 0");
  await rule(panel, "rule-1").locator(".rule-edit").click();
  await panel.locator("#rule-name").fill("Changed rule");
  await panel.locator("#rule-save").click();
  await expect(rule(panel, "rule-1")).toContainText("Changed rule");
  await expect(rule(panel, "rule-1")).toBeHidden();
  await expect(panel.locator(".rule-toolbar .count")).toHaveText("Showing 0 of 40 rules");
  await expectRetained(page, '[data-rule-key="rule-2"]');
  await panel.locator("#rule-search").fill("");
  await rule(panel, "rule-1").locator(".rule-edit").click();
  await expect(panel.locator("#rule-name")).toHaveValue("Changed rule");
  await panel.locator(".rule-close").last().click();
  await rule(panel, "rule-1").locator(".rule-delete").click();
  await acceptConfirmation(panel);
  await expect(rule(panel, "rule-1")).toHaveCount(0);
  await expectRetained(page, '[data-rule-key="rule-2"]');
  await panel.locator("#rule-add").click();
  await panel.locator("#rule-name").fill("New rule");
  await panel.locator("#rule-phrases").fill("new phrase");
  await panel.locator("#rule-action-type").selectOption("model_routing");
  await panel.locator("#rule-model").fill("gpt-5-mini");
  await panel.locator("#rule-save").click();
  await expect(rule(panel, "rule-41")).toContainText("New rule");
  await expect(panel.locator("#rules-title")).toHaveText("Rules (40)");
  await expect(panel.locator(".rule-toolbar .count")).toBeHidden();
  await expectRetained(page, '[data-rule-key="rule-2"]');
  expect((await calls(page, "request_rules", "update"))[0].revision).toBe(3);
  expect((await calls(page, "request_rules", "delete"))[0].revision).toBe(4);
  expect((await calls(page, "request_rules", "create"))[0].revision).toBe(5);
});

test("Request Rules: reorder reads the current revision and keeps distant cards", async ({page}) => {
  const panel = await openCollection(page, "capabilities/request-rules");
  await remember(page, '[data-rule-key="rule-20"]');
  for (let i = 0; i < 3; i++) {
    await rule(panel, "rule-1").locator(".rule-move-menu > summary").click();
    await rule(panel, "rule-1").locator('.rule-move[data-direction="down"]').click();
    await expect(panel.locator(".request-rule-card").nth(i + 1)).toHaveAttribute("data-rule-key", "rule-1");
    await expectRetained(page, '[data-rule-key="rule-20"]');
  }
  expect((await calls(page, "request_rules", "move")).map(c => c.revision)).toEqual([3, 4, 5]);
});

for (const surface of ["functions", "request-rules"]) {
  test(`${surface}: failed toggle keeps authoritative data and can be retried once`, async ({page}) => {
    const panel = await openCollection(page, `capabilities/${surface}`);
    const isTools = surface === "functions";
    await page.evaluate(isTools => {
      const hass = browserHarness.hass;
      const original = hass.callWS.bind(hass);
      let failed = false;
      hass.callWS = message => {
        if (!failed && message.section === (isTools ? "tools" : "request_rules") && message.action === (isTools ? "set_enabled" : "update")) {
          failed = true;
          return Promise.reject(new Error("stale revision"));
        }
        return original(message);
      };
    }, isTools);
    const input = isTools ? tool(panel, "tool_0").locator(".tool-enabled") : rule(panel, "rule-1").locator(".rule-enabled");
    await input.click();
    await expect(input).toBeChecked();
    await expect(input).toBeEnabled();
    await input.uncheck();
    await expect(input).not.toBeChecked();
    await expect(input).toBeEnabled();
    expect(await calls(page, isTools ? "tools" : "request_rules", isTools ? "set_enabled" : "update")).toHaveLength(1);
  });
}

test("Functions: HA availability refresh changes only the affected tool", async ({page}) => {
  const panel = await openCollection(page, "capabilities/functions");
  await page.evaluate(() => {
    const p = browserHarness.panel;
    const hass = browserHarness.hass, original = hass.callWS.bind(hass);
    hass.callWS = message => message.section === "tools" && message.action === "ha_catalog"
      ? Promise.resolve({saved: {tool_0: {description: "HA fixture capability", available: false}}}) : original(message);
    const config = p._draft.functions[0];
    config.function = {type: "ha_llm", tool_name: "fixture_capability", source_id: "test", api_id: "assist"};
    p._render();
  });
  await remember(page, '[data-tool-key="tool_1"]');
  await panel.locator("#refresh-ha-tools").click();
  await expect(tool(panel, "tool_0")).toContainText("Unavailable");
  await expectRetained(page, '[data-tool-key="tool_1"]');
});

test("Request Rules: moved cards retain identity and boundary buttons remain disabled after completion", async ({page}) => {
  const panel = await openCollection(page, "capabilities/request-rules", 2);
  await remember(page, '[data-rule-key="rule-1"]');
  const down = rule(panel, "rule-1").locator('.rule-move[data-direction="down"]');
  const up = rule(panel, "rule-1").locator('.rule-move[data-direction="up"]');
  await expect(up).toBeDisabled();
  await rule(panel, "rule-1").locator(".rule-move-menu > summary").click();
  await down.click();
  await expect(panel.locator(".request-rule-card").last()).toHaveAttribute("data-rule-key", "rule-1");
  await expect(down).toBeDisabled();
  await expect(up).toBeEnabled();
  await expectRetained(page, '[data-rule-key="rule-1"]');
  await rule(panel, "rule-1").locator(".rule-move-menu > summary").click();
  await up.click();
  await expect(panel.locator(".request-rule-card").first()).toHaveAttribute("data-rule-key", "rule-1");
  await expect(up).toBeDisabled();
  await expect(down).toBeEnabled();
  await expectRetained(page, '[data-rule-key="rule-1"]');
  expect(await calls(page, "request_rules", "move")).toHaveLength(2);
});

test("Functions: create and rename a group preserve tool and other group identity", async ({page}) => {
  const panel = await openCollection(page, "capabilities/functions");
  await remember(page, '[data-tool-key="tool_0"]');
  await page.evaluate(() => { window.retainedGroup = browserHarness.panel.shadowRoot.querySelector('.function-group-card[data-group-id="group-0"]'); });
  await panel.locator("#add-group").click();
  await panel.locator("#group-name").fill("New group");
  await panel.locator("#group-description").fill("Fixture group description");
  await panel.locator("#group-save").click();
  await expect(panel.locator('.function-group-card[data-group-id="new-group"]')).toHaveCount(1);
  await expect(panel.locator("[data-function-totals]")).toContainText("5 groups");
  await expectRetained(page, '[data-tool-key="tool_0"]');
  await panel.locator('.edit-group[data-group-id="new-group"]').click();
  await panel.locator("#group-name").fill("Renamed group");
  await panel.locator("#group-save").click();
  await expect(panel.locator('.function-group-card[data-group-id="new-group"] h3')).toHaveText("Renamed group");
  await expect(tool(panel, "tool_0").locator('.function-group-assignment option[value="new-group"]')).toHaveText("Renamed group");
  await expectRetained(page, '[data-tool-key="tool_0"]');
  expect(await page.evaluate(() => retainedGroup === browserHarness.panel.shadowRoot.querySelector('.function-group-card[data-group-id="group-0"]'))).toBe(true);
});

test("Functions: adding HA tools refreshes saved metadata without rebinding existing cards", async ({page}) => {
  const panel = await openCollection(page, "capabilities/functions");
  await remember(page, '[data-tool-key="tool_0"]');
  await page.evaluate(() => {
    const hass = browserHarness.hass, original = hass.callWS.bind(hass);
    const reference = {type: "ha_llm", source_id: "fixture", api_id: "assist", tool_name: "new_ha_tool"};
    window.catalogReads = 0;
    hass.callWS = message => {
      if (message.section === "tools" && message.action === "ha_catalog") {
        catalogReads++;
        return Promise.resolve({tools: [{name: "new_ha_tool", source: "fixture", description: "Fixture live tool", reference}], saved: {}});
      }
      if (message.section === "tools" && message.action === "ha_add") {
        const config = browserHarness.panel._draft;
        return Promise.resolve({functions: [...structuredClone(config.functions), {spec: {name: "new_ha_tool"}, function: reference}], function_groups: structuredClone(config.function_groups), ha_saved: {new_ha_tool: {description: "Saved tool metadata", available: true, source: "fixture"}}});
      }
      return original(message);
    };
  });
  await panel.locator("#add-ha-tools").click();
  const dialog = panel.locator('dialog[aria-label="Add Home Assistant LLM Tools"]');
  await dialog.locator("[data-tools] input").check();
  await dialog.locator("[data-add]").click();
  await expect(dialog).toHaveCount(0);
  await expect(tool(panel, "new_ha_tool")).toContainText("Saved tool metadata");
  await expectRetained(page, '[data-tool-key="tool_0"]');
  expect(await page.evaluate(() => catalogReads)).toBe(1);
});


test("Request Rules: duplicate preserves existing cards and fires once after repeated updates", async ({page}) => {
  const diagnostics = trackPageErrors(page);
  const panel = await openCollection(page, "capabilities/request-rules");
  await remember(page, '[data-rule-key="rule-2"]');
  await panel.locator("#rule-search").fill("fixture rule 0");
  for (let i = 0; i < 2; i++) {
    await beginCollectionMeasure(page);
    await rule(panel, "rule-1").locator(".rule-duplicate").click();
    await expect(rule(panel, `rule-${41 + i}`)).toBeVisible();
    await expect(panel.locator(".rule-toolbar .count")).toHaveText(`Showing ${2 + i} of ${41 + i} rules`);
    await expectRetained(page, '[data-rule-key="rule-2"]');
    const result = await finishCollectionMeasure(page);
    expect(result.mainChildReplacements).toBe(0);
    expect(result.retainedCards).toBe(40 + i);
  }
  expect((await calls(page, "request_rules", "duplicate")).map(c => c.revision)).toEqual([3, 4]);
  await expectHarnessClean(page, diagnostics);
});
