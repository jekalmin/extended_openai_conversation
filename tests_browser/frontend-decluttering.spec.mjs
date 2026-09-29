import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

const host = page => page.locator("extended-openai-management-panel");

async function seed(page, route, update) {
  await page.goto(fixtureUrl("guide"));
  await expect(host(page).locator(".page-shell")).toBeVisible();
  await page.evaluate(update);
  await page.goto(fixtureUrl(route));
  return host(page);
}

test("rule movement is compact, keyboard-accessible, dismissible, and disabled when filtered", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await seed(page, "capabilities/request-rules", () => {
    const state = browserHarness.getState();
    const first = state.requestRules.rules[0];
    first.match_type = "equals";
    first.phrases = ["first phrase", "second phrase"];
    state.requestRules.rules.push({...structuredClone(first), id:"rule-2", name:"Second rule", order:1});
    localStorage.setItem("extended-openai-browser-harness-state-v3",JSON.stringify(state));
  });
  const first = panel.locator('[data-rule-key="rule-1"]');
  const menu = first.locator(".rule-move-menu");
  const summary = menu.locator("summary");
  await expect(menu).toHaveJSProperty("open", false);
  await expect(first.locator(".actions > button")).toHaveCount(3);
  await expect(first.locator(".phrase-chips b")).toHaveCount(0);
  await expect(first.locator(".meta").last()).toContainText("Equals");
  await summary.focus();
  await page.keyboard.press("Enter");
  await expect(menu).toHaveJSProperty("open", true);
  await expect(menu.getByRole("button", {name:"Move up",exact:true})).toBeDisabled();
  await page.keyboard.press("Tab");
  await expect(menu.getByRole("button", {name:"Move down",exact:true})).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(panel.locator(".request-rule-card").last()).toHaveAttribute("data-rule-key", "rule-1");
  await expect(menu).toHaveJSProperty("open", false);
  await expect(summary).toBeFocused();
  await summary.press("Enter");
  await page.keyboard.press("Escape");
  await expect(menu).toHaveJSProperty("open", false);
  await expect(summary).toBeFocused();
  await summary.click();
  await panel.locator("#rule-search").click();
  await expect(menu).toHaveJSProperty("open", false);
  await panel.locator("#rule-search").fill("first phrase");
  await expect(panel.locator(".rule-filter-help")).toBeVisible();
  await summary.click();
  for (const direction of ["up", "down", "top", "bottom"]) {
    await expect(menu.locator(`[data-direction="${direction}"]`)).toBeDisabled();
  }
  expect(await page.evaluate(() => browserHarness.calls.filter(call => call.section === "request_rules" && call.action === "move").map(call => call.direction))).toEqual(["down"]);
  await expectHarnessClean(page, errors);
});

test("rule conditions preserve values, reveal validation errors, and reset visibility between editors", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = host(page);
  await panel.locator(".rule-edit").click();
  await expect(panel.locator("#rule-add-conditions")).toBeVisible();
  await expect(panel.locator("#rule-conditions-body")).toBeHidden();
  await panel.evaluate(p => {
    const original = p._call.bind(p);
    let once = true;
    p._call = (section, action, ...args) => {
      if (once && section === "request_rules" && action === "update") {
        once = false;
        return Promise.reject(Object.assign(new Error("Invalid condition: fix the template"), {field:"conditions"}));
      }
      return original(section, action, ...args);
    };
  });
  await panel.locator("#rule-save").click();
  await expect(panel.locator("#rule-error")).toContainText("Invalid condition");
  await expect(panel.locator("#rule-conditions-body")).toBeVisible();
  const conditions = [{condition:"and", conditions:[{condition:"template",value_template:"{{ true }}"}]}];
  await panel.locator("#rule-condition-host ha-selector").evaluate((selector, value) => {
    selector.value=value;
    selector.dispatchEvent(new CustomEvent("value-changed",{detail:{value},bubbles:true,composed:true}));
  }, conditions);
  await panel.locator("#rule-save").click();
  await expect(panel.locator("#rule-dialog")).not.toHaveJSProperty("open", true);
  await panel.locator(".rule-edit").click();
  await expect(panel.locator("#rule-conditions-body")).toBeVisible();
  await expect(panel.locator("#rule-condition-host ha-selector")).toHaveJSProperty("value", conditions);
  await panel.locator(".rule-close").first().click();
  await panel.locator("#rule-add").click();
  await expect(panel.locator("#rule-conditions-body")).toBeHidden();
  await expect(panel.locator("#rule-condition-host ha-selector")).toHaveJSProperty("value", []);
  await panel.locator("#rule-add-conditions").focus();
  await page.keyboard.press("Enter");
  await expect(panel.locator("#rule-conditions-body")).toBeVisible();
  await expect(panel.locator("#rule-add-conditions")).toBeHidden();
  await expectHarnessClean(page, errors);
});

test("pattern and substitution reference stays reachable without hiding response precedence", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/request-rules"));
  const panel = host(page);
  await panel.locator("#rule-add").click();
  await panel.locator("#rule-match").selectOption("sentence_pattern");
  await expect(panel.locator("#sentence-pattern-help > .help")).toContainText("[please]");
  const pattern = panel.locator("#sentence-pattern-help details");
  await expect(pattern).toHaveJSProperty("open", false);
  await pattern.locator("summary").focus();
  await page.keyboard.press("Enter");
  await expect(pattern.locator("p")).toContainText("named expansions and permutations are not supported");
  await expect(panel.locator("#rule-local-responses")).toContainText("Set conversation response overrides it");
  const response = panel.locator("#rule-response-reference");
  await expect(response).toHaveJSProperty("open", false);
  await response.locator("summary").click();
  await expect(response.locator("p")).toContainText("{battery.level}");
  await expect(response.locator("p")).toContainText("native script variables are only available inside actions");
  await expectHarnessClean(page, errors);
});

test("Function group identifiers remain searchable and editable without duplicate labels", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/functions"));
  const panel = host(page);
  await expect(panel.locator(".always-card .availability-badge")).toHaveCount(0);
  const group = panel.locator('[data-group-id="baseline-group"].function-group-card');
  await expect(group.locator(".function-group-heading code")).toHaveCount(0);
  await panel.locator("#tool-search").fill("baseline-group");
  await expect(group).toBeVisible();
  await group.locator(".edit-group").click();
  await expect(panel.locator("#group-id")).toHaveValue("baseline-group");
  await expectHarnessClean(page, errors);
});

for (const admin of [true,false]) {
  test(`Knowledge access ${admin ? "has one effective status and explanation" : "preserves administrator-only route access"}`, async ({page}) => {
    const errors = trackPageErrors(page);
    await page.goto(fixtureUrl("data-memory/knowledge", admin ? "" : "&admin=0"));
    const panel = host(page);
    if (!admin) {
      // Knowledge management is an administrator-only route. Presentation
      // simplification must not grant access to its data or controls.
      await expect(panel.locator("main")).toContainText("Administrator permission is required");
      await expect(panel.locator("#knowledge-enabled-toggle")).toHaveCount(0);
      await expectHarnessClean(page, errors);
      return;
    }
    await expect(panel.locator("#knowledge-status")).toHaveCount(1);
    await expect(panel.locator("#knowledge-status")).toBeVisible();
    if (admin) {
      const toggle = panel.locator("#knowledge-enabled-toggle");
      await toggle.uncheck();
      await expect(panel.locator("#knowledge-status")).toContainText("Off");
      await expect(panel.locator("#knowledge-status")).toContainText("Stored sources stay in the library");
      expect(await panel.locator(".knowledge-availability-setting").innerText()).toMatch(/cannot use them/);
      await toggle.check();
      await expect(panel.locator("#knowledge-status")).toContainText("Needs sources");
    }
    await expectHarnessClean(page, errors);
  });
}

test("Memory settings heading retains one working cross-link and no global tagline", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("data-memory/memory-settings"));
  const panel = host(page);
  const manage = panel.getByRole("button",{name:"Manage stored memories",exact:true});
  await expect(manage).toHaveCount(1);
  await expect(panel.locator(".page-intro")).toContainText("Manage stored memories");
  await expect(panel).not.toContainText("Looking for stored memories?");
  await expect(panel.locator("header")).not.toContainText("Configure your assistant, capabilities, retained data, and maintenance");
  await manage.click();
  await expect(panel.locator("#add-memory")).toBeVisible();
  await expectHarnessClean(page, errors);
});

test("Guest permissions share the appropriate cards without changing activation", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/guest-mode"));
  const panel = host(page);
  const activation = panel.locator(".content-card").filter({has:page.getByRole("heading",{name:"Guest Mode activation",exact:true})});
  await expect(activation.locator("#guest-controls-enabled")).toHaveCount(1);
  await activation.getByText("Assistant activation permission",{exact:true}).click();
  await activation.locator("#guest-controls-enabled").check();
  await expect(activation).toContainText("No interval configured");
  await expect(panel.locator("#guest-disable")).toHaveCount(0);
  const capabilities = panel.locator(".content-card").filter({has:page.getByRole("heading",{name:"Guest capabilities",exact:true})});
  await expect(capabilities.locator("#guest-web-search")).toBeVisible();
  await capabilities.locator("#guest-web-search").check();
  await expect(capabilities).toContainText("Personal memory and owner conversation archives are always unavailable");
  await expect(panel).not.toContainText("Guest archive retention");
  await expect(panel.getByRole("heading",{name:"Hosted capabilities",exact:true})).toHaveCount(0);
  expect(await panel.evaluate(p => ({permission:p._guestDraft.guest_mode_enabled,web:p._guestDraft.guest_web_search}))).toEqual({permission:true,web:true});
  expect(await page.evaluate(() => browserHarness.calls.filter(call => call.section==="guest_mode" && ["update","disable"].includes(call.action)))).toEqual([]);
  await expectHarnessClean(page, errors);
});

test("Voice policies retain native picker identity and saved inactive choices across reveal and search", async ({page}) => {
  const errors = trackPageErrors(page);
  const panel = await seed(page,"assistant/voice",() => {
    const state = browserHarness.getState();
    Object.assign(state.configuration.config,{voice_scope_policy:"unretained",voice_unmapped_policy:"default_user",voice_default_user_id:"test-user",voice_device_mappings:{}});
    localStorage.setItem("extended-openai-browser-harness-state-v3",JSON.stringify(state));
  });
  const inactive = panel.locator("[data-voice-inactive-settings]");
  const picker = panel.locator("#config-voice_default_user_picker");
  await expect(panel.locator("#voice-current-summary")).toContainText("no retained personal data");
  await expect(inactive).toHaveJSProperty("open",false);
  await expect(picker).toBeHidden();
  await panel.evaluate(p => {window.savedVoicePicker=p.shadowRoot.querySelector("#config-voice_default_user_picker");});
  await panel.locator("#config-voice_scope_policy").selectOption("device_mapping");
  await expect(panel.locator("[data-voice-fallback-card]")).toBeVisible();
  await expect(picker).toHaveJSProperty("disabled",false);
  await expect(panel.locator("#voice-current-summary")).toContainText("Test User");
  await panel.locator("#config-voice_scope_policy").selectOption("unretained");
  await expect(picker).toHaveJSProperty("disabled",true);
  await expect(picker).toBeHidden();
  await panel.locator("#settings-search").fill("Default voice user");
  await panel.locator('.settings-result[data-target="config-voice_default_user_id"]').click();
  await expect(inactive).toHaveJSProperty("open",true);
  await expect(picker).toHaveJSProperty("disabled",true);
  expect(await picker.evaluate(node => node===window.savedVoicePicker)).toBe(true);
  await expect(panel.locator("#config-voice_default_user_id")).toHaveValue("test-user");
  await panel.locator("#config-voice_scope_policy").selectOption("default_user");
  await expect(picker).toHaveJSProperty("disabled",false);
  expect(await panel.evaluate(p=>({policy:p._draft.voice_scope_policy,fallback:p._draft.voice_unmapped_policy,user:p._draft.voice_default_user_id}))).toEqual({policy:"default_user",fallback:"default_user",user:"test-user"});
  await expectHarnessClean(page, errors);
});

test("Quiet Hours uses compact healthy satellites and reveals problems and draft overrides", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/quiet-hours"));
  const panel=host(page);
  await expect(panel.locator("#qh-enabled")).toBeVisible();
  await panel.evaluate(p => {
    p._hass.states={"media_player.kitchen":{state:"idle",attributes:{volume_level:0.5,friendly_name:"Kitchen speaker"}},"media_player.manual":{state:"idle",attributes:{volume_level:0.2}}};
    p._result.satellites=[
      {satellite_entity_id:"assist_satellite.kitchen",name:"Kitchen",media_player_entity_id:"media_player.kitchen",media_player_source:"auto",media_player_candidates:["media_player.kitchen","media_player.manual"],wake_sound_candidates:[]},
      {satellite_entity_id:"assist_satellite.problem",name:"Unavailable speaker",media_player_entity_id:"media_player.missing",media_player_source:"auto",wake_sound_candidates:[]},
      {satellite_entity_id:"assist_satellite.ambiguous",name:"Ambiguous wake switch",media_player_entity_id:"media_player.kitchen",media_player_source:"auto",wake_sound_candidates:["switch.a","switch.b"]},
    ];
    p._eocMainMarkup=null;p._render();
  });
  const kitchen=panel.locator('[data-qh-satellite="assist_satellite.kitchen"]');
  await expect(kitchen.locator("details")).toHaveJSProperty("open",false);
  await expect(kitchen).toContainText("Kitchen speaker");
  await expect(kitchen).toContainText("media_player.kitchen");
  await expect(kitchen).toContainText("optional");
  await expect(panel.locator('[data-qh-satellite="assist_satellite.problem"] details')).toHaveJSProperty("open",true);
  await expect(panel.locator('[data-qh-satellite="assist_satellite.ambiguous"] details')).toHaveJSProperty("open",true);
  await kitchen.locator("summary").click();
  await kitchen.locator('[data-kind="media_player_entity_id"]').evaluate(node => {
    node.value="media_player.manual";
    node.dispatchEvent(new CustomEvent("value-changed",{detail:{value:"media_player.manual"},bubbles:true,composed:true}));
  });
  await expect(kitchen.locator("[data-qh-media-summary]")).toContainText("Manual · media_player.manual");
  expect(await panel.evaluate(p => p._quietHoursDraft.overrides)).toEqual({"assist_satellite.kitchen":{media_player_entity_id:"media_player.manual"}});
  await panel.evaluate(p => {
    const original=p._call.bind(p);
    p._call=async (section,action,...args) => {
      const result=await original(section,action,...args);
      if (section!=="quiet_hours" || action!=="update") return result;
      const selected=result.config.overrides?.["assist_satellite.kitchen"]?.media_player_entity_id;
      return {...result,satellites:p._result.satellites.map(item => item.satellite_entity_id==="assist_satellite.kitchen"
        ? {...item,media_player_entity_id:selected || "media_player.kitchen",media_player_source:selected ? "manual" : "auto"} : item)};
    };
    window.savedQuietPicker=p.shadowRoot.querySelector('[data-qh-satellite="assist_satellite.kitchen"] .qh-override');
  });
  await panel.locator("#save-page").click();
  await expect(panel.locator(".save-bar")).toHaveCount(0);
  const media=kitchen.locator('[data-kind="media_player_entity_id"]');
  await media.evaluate(node => {
    node.value="";
    node.dispatchEvent(new CustomEvent("value-changed",{detail:{value:""},bubbles:true,composed:true}));
  });
  await expect(kitchen.locator("[data-qh-media-summary]")).toContainText("resolved after saving");
  await panel.locator("#save-page").click();
  await expect(panel.locator(".save-bar")).toHaveCount(0);
  await expect(kitchen.locator("[data-qh-media-summary]")).toContainText("Automatic · Kitchen speaker (media_player.kitchen)");
  await expect(kitchen.locator("[data-qh-speaker-status]")).toHaveText("Speaker ready");
  await expect(media).toHaveJSProperty("placeholder","Automatic · media_player.kitchen");
  await expect(kitchen.locator("details")).toHaveJSProperty("open",true);
  expect(await media.evaluate(node => node===window.savedQuietPicker)).toBe(true);
  await expectHarnessClean(page,errors);
});
