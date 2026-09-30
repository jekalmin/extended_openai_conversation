import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

const panelFor = (page) => page.locator("extended-openai-management-panel");
const satellite = (id = "assist_satellite.kitchen", suffix = "kitchen") => ({
  satellite_entity_id:id, name:suffix, media_player_entity_id:`media_player.${suffix}`, wake_sound_entity_id:`switch.${suffix}_wake`,
  media_player_source:"auto", wake_sound_source:"auto", media_player_candidates:[`media_player.${suffix}`, `media_player.${suffix}_alt`],
  wake_sound_candidates:[`switch.${suffix}_wake`, `switch.${suffix}_chime`],
});

async function installSatelliteFixture(panel, satellites = [satellite()]) {
  await panel.evaluate((host, items) => {
    host._result.satellites = items;
    host._hass.states ||= {};
    for (const item of items) {
      host._hass.states[item.media_player_entity_id] = {state:"on", attributes:{volume_level:0.8, friendly_name:item.name}};
      host._hass.states[item.wake_sound_entity_id] = {state:"on", attributes:{friendly_name:`${item.name} wake`}};
    }
    host._render();
  }, satellites);
}

async function selectEntity(panel, selector, value) {
  await panel.locator(selector).evaluate((picker, selected) => {
    picker.value = selected;
    picker.dispatchEvent(new CustomEvent("value-changed", {detail:{value:selected}, bubbles:true, composed:true}));
  }, value);
}

test("nightly Quiet Hours schedule controls round-trip through save, route reopen, and fresh load", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/quiet-hours"));
  const panel = panelFor(page);
  await expect(panel.locator("#qh-enabled")).toBeVisible();

  for (const [percent, policy, start, end] of [[0, "off", "00:00", "00:00"], [47, "on", "23:59", "00:01"], [100, "unchanged", "23:30", "06:15"]]) {
    await panel.locator("#qh-enabled").setChecked(percent !== 0);
    await panel.locator("#qh-start").fill(start);
    await panel.locator("#qh-end").fill(end);
    await panel.locator("#qh-volume").fill(String(percent));
    await expect(panel.locator("#qh-volume-value")).toHaveText(`${percent}%`);
    await panel.locator("#qh-wake").selectOption(policy);
    await panel.locator("#save-page").click();
    await expect(panel.locator(".save-bar")).toHaveCount(0);
    const config = await panel.evaluate(host => host._result.config);
    expect(config).toMatchObject({enabled:percent !== 0, start, end, max_volume:percent / 100, wake_sound:policy});
    await panel.evaluate(host => host._navigate("overview", ""));
    await panel.evaluate(host => host._navigate("capabilities", "quiet-hours"));
    await expect(panel.locator("#qh-volume")).toHaveValue(String(percent));
    await expect(panel.locator("#qh-wake")).toHaveValue(policy);
  }

  await page.goto(fixtureUrl("capabilities/quiet-hours"));
  const reloaded = panelFor(page);
  await expect(reloaded.locator("#qh-start")).toHaveValue("23:30");
  await expect(reloaded.locator("#qh-end")).toHaveValue("06:15");
  await expect(reloaded.locator("#qh-volume-value")).toHaveText("100%");
  await expect(reloaded.locator("#qh-enabled")).toBeChecked();
  await expect(reloaded.locator("#qh-wake")).toHaveValue("unchanged");
  await expectHarnessClean(page, errors);
});

test("nightly Quiet Hours satellite overrides clear to pending automatic resolution and preserve other drafts", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/quiet-hours"));
  const panel = panelFor(page);
  await installSatelliteFixture(panel, [satellite(), satellite("assist_satellite.office", "office")]);
  await panel.evaluate(host => {
    host._result.config.overrides = {"assist_satellite.kitchen":{media_player_entity_id:"media_player.kitchen_alt"}};
    host._quietHoursDraft = structuredClone(host._result.config);
    host._render();
  });
  await selectEntity(panel, '[data-qh-satellite="assist_satellite.kitchen"] [data-kind="media_player_entity_id"]', "media_player.kitchen_alt");
  await selectEntity(panel, '[data-qh-satellite="assist_satellite.kitchen"] [data-kind="wake_sound_entity_id"]', "switch.kitchen_chime");
  await selectEntity(panel, '[data-qh-satellite="assist_satellite.office"] [data-kind="media_player_entity_id"]', "media_player.office_alt");
  await selectEntity(panel, '[data-qh-satellite="assist_satellite.kitchen"] [data-kind="media_player_entity_id"]', "");
  await expect(panel.locator('[data-qh-satellite="assist_satellite.kitchen"] [data-qh-media-summary]')).toContainText("resolved after saving");
  await expect(panel.locator('[data-qh-satellite="assist_satellite.kitchen"] [data-qh-wake-summary]')).toContainText("Manual");
  expect(await panel.evaluate(host => host._quietHoursDraft.overrides)).toEqual({
    "assist_satellite.kitchen":{wake_sound_entity_id:"switch.kitchen_chime"},
    "assist_satellite.office":{media_player_entity_id:"media_player.office_alt"},
  });

  await panel.locator("#qh-wake").selectOption("unchanged");
  await panel.locator("#save-page").click();
  await expect(panel.locator(".save-bar")).toHaveCount(0);
  expect(await panel.evaluate(host => host._result.config.overrides)).toEqual({
    "assist_satellite.kitchen":{wake_sound_entity_id:"switch.kitchen_chime"},
    "assist_satellite.office":{media_player_entity_id:"media_player.office_alt"},
  });
  await page.goto(fixtureUrl("capabilities/quiet-hours"));
  await installSatelliteFixture(panelFor(page), [satellite(), satellite("assist_satellite.office", "office")]);
  await expect(panelFor(page).locator('[data-qh-satellite="assist_satellite.office"] [data-qh-media-summary]')).toContainText("media_player.office_alt");
  await expectHarnessClean(page, errors);
});

test("nightly Quiet Hours surfaces unavailable and changing satellite candidates while editing", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/quiet-hours"));
  const panel = panelFor(page);
  await installSatelliteFixture(panel, [satellite()]);
  await panel.evaluate(host => {
    const player = host._result.satellites[0].media_player_entity_id;
    host._hass.states[player].state = "unavailable";
    host._result.satellites[0].media_player_candidates = ["media_player.new_candidate"];
    host._render();
  });
  await expect(panel.locator('[data-qh-satellite="assist_satellite.kitchen"] [data-qh-speaker-status]')).toHaveText("Speaker unavailable");
  await selectEntity(panel, '[data-qh-satellite="assist_satellite.kitchen"] [data-kind="media_player_entity_id"]', "media_player.new_candidate");
  await expect.poll(() => panel.locator('[data-qh-satellite="assist_satellite.kitchen"] [data-kind="media_player_entity_id"]').evaluate(p => p.includeEntities)).toEqual(["media_player.new_candidate"]);
  await expectHarnessClean(page, errors);
});

test("nightly Quiet Hours retries a failed save and protects agent selection while saving", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/quiet-hours"));
  const panel = panelFor(page);
  await panel.evaluate(host => {
    const original = host._hass.callWS.bind(host._hass);
    window.quietSaveCalls = 0;
    window.releaseQuietSave = null;
    host._hass.callWS = async message => {
      if (message.section !== "quiet_hours" || message.action !== "update") return original(message);
      window.quietSaveCalls++;
      if (window.quietSaveCalls === 1) throw new Error("Quiet Hours save unavailable");
      if (window.quietSaveCalls === 3) await new Promise(resolve => { window.releaseQuietSave = resolve; });
      return original(message);
    };
  });
  await panel.locator("#qh-enabled").check();
  await panel.locator("#save-page").click();
  await expect(panel.locator("#toast")).toContainText("Unable to save changes: Quiet Hours save unavailable");
  await expect(panel.locator("#qh-enabled")).toBeChecked();
  await panel.locator("#save-page").click();
  await expect(panel.locator(".save-bar")).toHaveCount(0);

  await panel.locator("#qh-start").fill("21:00");
  await panel.locator("#save-page").click();
  await expect.poll(() => page.evaluate(() => typeof window.releaseQuietSave)).toBe("function");
  await expect(panel.locator("#agent")).toBeDisabled();
  await page.evaluate(() => window.releaseQuietSave());
  await expect(panel.locator("#agent")).toBeEnabled();
  await expect(panel.locator("#qh-start")).toHaveValue("21:00");
  await expectHarnessClean(page, errors);
});
