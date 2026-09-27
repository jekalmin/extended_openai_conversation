import {expect} from "@playwright/test";

export const fixtureUrl = (route, extra = "") => `/tests_browser/fixture.html?route=${route}${extra}${process.env.SHIPPED_BUNDLE === "1" ? "&bundle=1" : ""}`;

export function trackPageErrors(page) {
  const diagnostics = [];
  diagnostics.consoleErrors = [];
  diagnostics.requestFailures = [];
  diagnostics.badResponses = [];

  page.on("pageerror", (error) => diagnostics.push(error.message));
  page.on("console", (message) => {
    if (message.type() !== "error") return;
    const location = message.location();
    const suffix = location?.url
      ? ` (${location.url}${location.lineNumber != null ? `:${location.lineNumber}` : ""})`
      : "";
    diagnostics.consoleErrors.push(`${message.text()}${suffix}`);
  });
  page.on("requestfailed", (request) => {
    diagnostics.requestFailures.push(
      `${request.method()} ${request.url()}: ${request.failure()?.errorText || "request failed"}`,
    );
  });
  page.on("response", (response) => {
    if (response.status() < 400) return;
    diagnostics.badResponses.push(
      `${response.status()} ${response.request().method()} ${response.url()}`,
    );
  });
  return diagnostics;
}

export async function expectHarnessClean(page, diagnostics) {
  const harness = await page.evaluate(() => ({errors: window.browserHarness?.windowErrors || [], rejections: window.browserHarness?.rejections || []}));
  expect(diagnostics).toHaveLength(0);
  expect(diagnostics.consoleErrors || []).toEqual([]);
  expect(diagnostics.requestFailures || []).toEqual([]);
  expect(diagnostics.badResponses || []).toEqual([]);
  expect(harness).toEqual({errors: [], rejections: []});
}
export async function acceptConfirmation(panel) {
  await expect(panel.locator("#confirm-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#confirm-accept").click();
}
export const browserToolYaml = (description = "Browser journey tool") => `spec:\n  name: browser_tool\n  description: ${description}\n  parameters:\n    type: object\n    properties: {}\nfunction:\n  type: native\n  name: get_user_from_user_id\n`;
