import {expect, test} from "@playwright/test";
import {acceptConfirmation, expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

test("Conversation history reads turns and persists end-active and delete mutations", async ({page}) => {
  const errors = trackPageErrors(page);
  const url = fixtureUrl("data-memory/conversations", "&seed_conversations=1");
  await page.goto(url);
  let panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Kitchen project"})).toBeVisible();
  await expect(panel.getByRole("heading", {name: "Kitchen speaker"})).toBeVisible();
  await panel.locator("#archive-query").fill("kitchen plan");
  await panel.locator("#archive-search").click();
  await expect(panel.getByRole("heading", {name: "Kitchen project"})).toBeVisible();
  await panel.getByRole("button", {name: "Clear search"}).click();

  await panel.locator('.view-session[data-id="session-1"]').click();
  await expect(panel.locator("#session-dialog")).toHaveJSProperty("open", true);
  await expect(panel.locator("#session-body")).toContainText("Review the kitchen plan.");
  await panel.locator("#session-dialog .close-session").first().click();

  await panel.locator('.end-active[data-key="active-1"]').click();
  await expect(panel.locator("#confirm-dialog")).toHaveJSProperty("open", true);
  await panel.locator("#confirm-cancel").click();
  await expect(panel.getByRole("heading", {name: "Kitchen speaker"})).toBeVisible();
  await panel.locator('.end-active[data-key="active-1"]').click();
  await acceptConfirmation(panel);
  await expect(panel.getByRole("heading", {name: "Kitchen speaker"})).toHaveCount(0);
  const firstActions = await page.evaluate(() => window.browserHarness.calls.filter((call) => call.section === "conversations").map((call) => call.action));
  expect(firstActions).toEqual(expect.arrayContaining(["list", "active", "search", "get", "end_active"]));

  await page.goto(url);
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Kitchen speaker"})).toHaveCount(0);
  await expect(panel.getByRole("heading", {name: "Kitchen project"})).toBeVisible();
  await panel.locator('.delete-session[data-id="session-1"]').click();
  await acceptConfirmation(panel);
  await expect(panel.getByRole("heading", {name: "Kitchen project"})).toHaveCount(0);
  const actions = await page.evaluate(() => window.browserHarness.calls.filter((call) => call.section === "conversations").map((call) => call.action));
  expect(actions).toEqual(expect.arrayContaining(["list", "active", "delete"]));

  await page.goto(url);
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByText("No retained conversations in this scope.", {exact: true})).toBeVisible();
  await expectHarnessClean(page, errors);
});
