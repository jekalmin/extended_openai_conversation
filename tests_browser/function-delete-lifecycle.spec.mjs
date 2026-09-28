import {expect, test} from "@playwright/test";
import {acceptConfirmation, expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

const toolYaml = (name) => `spec:\n  name: ${name}\n  description: ${name} lifecycle probe\n  parameters:\n    type: object\n    properties: {}\nfunction:\n  type: native\n  name: get_user_from_user_id\n`;

async function addTool(panel, name) {
  await panel.locator("#add-tool").click();
  await panel.locator("#tool-yaml").fill(toolYaml(name));
  await panel.locator("#tool-save").click();
  await expect(panel.locator(".tool-card").filter({hasText:name})).toBeVisible();
}

test("delete releases Function mutation lifecycle for another delete and Overview", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/functions"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name:"Function Tools & Groups", exact:true})).toBeVisible();
  await addTool(panel, "delete_lifecycle_a");
  await addTool(panel, "delete_lifecycle_b");

  await panel.evaluate((host) => {
    const original = host._hass.callWS.bind(host._hass);
    const waiting = [];
    window.functionDeleteControl = {
      attempts: 0,
      release() { waiting.shift()?.(); },
    };
    host._hass.callWS = async (message) => {
      if (message.section === "tools" && message.action === "delete") {
        window.functionDeleteControl.attempts += 1;
        await new Promise((resolve) => waiting.push(resolve));
      }
      return original(message);
    };
  });

  let card = panel.locator(".tool-card").filter({hasText:"delete_lifecycle_a"});
  const firstDelete = card.locator(".delete-tool");
  await firstDelete.click();
  await acceptConfirmation(panel);
  await expect(firstDelete).toBeDisabled();
  await expect(firstDelete).toHaveText("Deleting…");
  await firstDelete.evaluate((button) => { button.click(); button.click(); });
  expect(await page.evaluate(() => window.functionDeleteControl.attempts)).toBe(1);
  await page.evaluate(() => window.functionDeleteControl.release());
  await expect(card).toHaveCount(0);

  await panel.evaluate((host) => host._navigate("overview"));
  await expect(panel.locator(".dashboard-grid")).toBeVisible();
  await panel.evaluate((host) => host._navigate("capabilities", "functions"));
  await expect(panel.getByRole("heading", {name:"Function Tools & Groups", exact:true})).toBeVisible();

  card = panel.locator(".tool-card").filter({hasText:"delete_lifecycle_b"});
  await card.locator(".delete-tool").click();
  await acceptConfirmation(panel);
  await expect(card.locator(".delete-tool")).toHaveText("Deleting…");
  await page.evaluate(() => window.functionDeleteControl.release());
  await expect(card).toHaveCount(0);
  await expect(panel.locator(".tool-card").filter({hasText:/delete_lifecycle_[ab]/})).toHaveCount(0);
  const diagnostics = await panel.evaluate((host) => ({
    tailReleased: host._eocFunctionMutationTail === null,
    deletes: host._eocFunctionMutationDiagnostics.filter((item) => item.action === "delete"),
    requests: host._eocRequestDiagnostics.map(({section, action, status}) => ({section, action, status})),
  }));
  expect(diagnostics.tailReleased).toBe(true);
  expect(diagnostics.deletes).toHaveLength(2);
  expect(diagnostics.deletes.every((item) => item.status === "fulfilled" && item.ui)).toBe(true);
  expect(diagnostics.requests.some((item) => item.section === "overview" && item.status === "fulfilled")).toBe(true);
  expect(await page.evaluate(() => window.functionDeleteControl.attempts)).toBe(2);
  await expectHarnessClean(page, pageErrors);
});

test("failed delete restores its button for a deliberate retry", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/functions"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name:"Function Tools & Groups", exact:true})).toBeVisible();
  await addTool(panel, "delete_failure_probe");
  await panel.evaluate((host) => {
    const original = host._hass.callWS.bind(host._hass);
    let fail = true;
    host._hass.callWS = async (message) => {
      if (fail && message.section === "tools" && message.action === "delete") {
        fail = false;
        throw new Error("Injected delete failure");
      }
      return original(message);
    };
  });

  const card = panel.locator(".tool-card").filter({hasText:"delete_failure_probe"});
  const button = card.locator(".delete-tool");
  await button.click();
  await acceptConfirmation(panel);
  await expect(panel.locator("#toast")).toContainText("Injected delete failure");
  await expect(button).toBeEnabled();
  await expect(button).toHaveText("Delete");
  await button.click();
  await acceptConfirmation(panel);
  await expect(card).toHaveCount(0);
  await expectHarnessClean(page, pageErrors);
});
