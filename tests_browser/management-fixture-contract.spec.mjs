import {expect, test} from "@playwright/test";
import {fixtureUrl} from "./browser-helpers.mjs";
import {createDataCollectionBackend} from "./data-collection-backend.mjs";

test("collection backend rejects unknown Knowledge actions", async () => {
  const backend = createDataCollectionBackend(1);
  await expect(backend.call({section: "knowledge", action: "renmae"})).rejects.toThrow(/Unhandled Knowledge fixture action: renmae/);
});

test("unknown management actions and malformed saves fail at the fixture boundary", async ({page}) => {
  await page.goto(fixtureUrl("guide"));
  await expect.poll(() => page.evaluate(() => Boolean(window.browserHarness?.hass))).toBe(true);
  const errors = await page.evaluate(async () => {
    const call = (section, action, extra = {}) => window.browserHarness.hass.callWS({
      type: "extended_openai_conversation_responses/management", section, action,
      entry_id: "entry-1", subentry_id: "agent-1", ...extra,
    }).then(() => "accepted", (error) => error.message);
    return [
      await call("configuration", "saev", {config: {}, revision: "fixture-7"}),
      await call("new_management_section", "get"),
      await call("configuration", "save", {config: "invalid", revision: "fixture-7"}),
      await window.browserHarness.hass.callWS({type: "get_states"}).then(() => "HA accepted"),
      await window.browserHarness.hass.callWS({type: "extended_openai_conversation_responses/renamed_endpoint"}).then(() => "accepted", error => error.message),
    ];
  });
  expect(errors[0]).toMatch(/Unsupported EOAI management fixture request configuration\/saev/);
  expect(errors[1]).toMatch(/Unsupported EOAI management fixture request new_management_section\/get/);
  expect(errors[2]).toMatch(/Malformed EOAI management configuration\/save/);
  expect(errors[3]).toBe("HA accepted");
  expect(errors[4]).toMatch(/Unsupported EOAI fixture WebSocket type/);
});
