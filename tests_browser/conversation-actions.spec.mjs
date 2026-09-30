import {expect, test} from "@playwright/test";
import {acceptConfirmation, expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

test("Archive search and repeated page navigation settle their controls", async ({page}) => {
  const errors = trackPageErrors(page);
  const url = fixtureUrl("data-memory/conversations", "&seed_conversations=1");
  await page.goto(url);
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name:"Kitchen project"})).toBeVisible();
  await page.evaluate(() => {
    const state = browserHarness.getState();
    const original = state.conversations.sessions[0];
    state.conversations.sessions = Array.from({length:125}, (_, i) => ({...structuredClone(original), session_id:`session-${i+1}`, title:`Archive ${i+1}`}));
    localStorage.setItem("extended-openai-browser-harness-state-v3", JSON.stringify(state));
  });
  await page.goto(url);
  const pager = panel.locator(".eoc-history-pager");
  const next = pager.getByRole("button", {name:"Next", exact:true});
  const previous = pager.getByRole("button", {name:"Previous", exact:true});
  await expect(pager).toContainText("1–50 of 125 conversations");
  await expect(previous).toBeDisabled();
  await expect(next).toBeEnabled();
  await panel.locator("#archive-query").fill("no-matching-conversation-987");
  await panel.locator("#archive-search").click();
  await expect(panel.getByText("No conversations match this search.", {exact:true})).toBeVisible();
  await pager.getByRole("button", {name:"Clear search"}).click();
  await expect(pager).toContainText("1–50 of 125 conversations");
  for (const [action, range, prevEnabled, nextEnabled] of [[next,"51–100",true,true], [next,"101–125",true,false], [previous,"51–100",true,true], [previous,"1–50",false,true]]) {
    await action.click();
    await expect(pager).toContainText(`${range} of 125 conversations`);
    await expect(previous)[prevEnabled ? "toBeEnabled" : "toBeDisabled"]();
    await expect(next)[nextEnabled ? "toBeEnabled" : "toBeDisabled"]();
  }
  await expectHarnessClean(page, errors);
});

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
