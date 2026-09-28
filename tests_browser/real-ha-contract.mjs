import {readFileSync} from "node:fs";
import {expect} from "@playwright/test";

const contract = JSON.parse(readFileSync(new URL("../tests_stress/management_ws_contract.json", import.meta.url), "utf8"));

function containsShape(actual, expected) {
  if (expected === null || typeof expected !== "object") return Object.is(actual, expected);
  if (Array.isArray(expected)) {
    return Array.isArray(actual) && expected.length === actual.length
      && expected.every((value, index) => containsShape(actual[index], value));
  }
  return actual !== null && typeof actual === "object"
    && Object.entries(expected).every(([key, value]) => Object.hasOwn(actual, key)
      && containsShape(actual[key], value));
}

// The bridge records the browser's exact callWS payload before HA validates it.
// A passing UI assertion alone does not prove that every intended action crossed
// the WebSocket command schema, especially when a failed click leaves old UI.
export async function expectContractCalls(page, journey) {
  const {calls: observed, outcomes} = await page.evaluate(() => ({
    calls: window.browserHarness.calls,
    outcomes: window.browserHarness.outcomes,
  }));
  const expected = contract.actions.filter((action) => action.journey === journey);
  expect(expected.length, `No reviewed WebSocket contract entries for ${journey}`).toBeGreaterThan(0);
  const enhanced = process.env.RUN_ENHANCED_MANAGEMENT_CONTRACT === "1";
  if (enhanced) {
    const envelope = new Set(["type", "section", "action", "entry_id", "subentry_id"]);
    for (const [index, call] of observed.entries()) {
      const reviewed = contract.actions.filter((item) =>
        (item.type ? call.type === item.type : call.section === item.section)
        && call.action === item.action);
      if (!reviewed.length || outcomes[index]?.success !== true) continue;
      const fields = Object.keys(call).filter((key) => !envelope.has(key)).sort();
      expect(reviewed.some((item) => {
        const required = item.keys.filter((key) => !envelope.has(key));
        const allowed = new Set([...required, ...(item.optional_keys || [])]);
        return required.every((key) => fields.includes(key)) && fields.every((key) => allowed.has(key));
      }),
        `${journey}: unreviewed fields on ${call.section}/${call.action}: ${JSON.stringify(fields)}`).toBe(true);
    }
  }
  for (const action of expected) {
    const matching = observed.filter((call, index) => (!enhanced || outcomes[index]?.success === true)
      && (action.type ? call.type === action.type : call.section === action.section)
      && call.action === action.action
      && action.keys.every((key) => Object.hasOwn(call, key))
      && containsShape(call, action.contains || action.equals || {}));
    const candidates = observed.flatMap((call, index) =>
      call.section === action.section && call.action === action.action
        ? [{keys: Object.keys(call), success: outcomes[index]?.success}] : []);
    expect(matching.length, `${journey}: successful ${action.section}/${action.action} with ${action.keys.join(", ")}; observed ${JSON.stringify(candidates)}`).toBeGreaterThanOrEqual(action.min_calls || 1);
  }
}
