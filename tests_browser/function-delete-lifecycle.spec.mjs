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

test("queued Function Tool mutation resolves its implicit revision when it starts", async ({page}) => {
  const pageErrors = trackPageErrors(page);
  await page.goto(fixtureUrl("capabilities/functions"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name:"Function Tools & Groups", exact:true})).toBeVisible();
  await addTool(panel, "queued_delete_a");
  await addTool(panel, "queued_toggle_b");

  await panel.evaluate((host) => {
    const original = host._hass.callWS.bind(host._hass);
    let releaseDelete;
    window.queuedRevisionProbe = {deleteStarted:false, release:() => releaseDelete?.(), calls:[]};
    host._hass.callWS = async (message) => {
      if (message.section === "tools" && ["delete", "set_enabled"].includes(message.action)) {
        window.queuedRevisionProbe.calls.push({...message});
      }
      if (message.section === "tools" && message.action === "delete") {
        window.queuedRevisionProbe.deleteStarted = true;
        await new Promise(resolve => { releaseDelete = resolve; });
      }
      return original(message);
    };
  });

  const firstCard = panel.locator('[data-tool-key="queued_delete_a"]');
  const firstName = await firstCard.getAttribute("data-tool-key");
  const secondCard = panel.locator('[data-tool-key="queued_toggle_b"]');
  const secondName = await secondCard.getAttribute("data-tool-key");
  const firstDelete = firstCard.locator(".delete-tool");
  await firstDelete.click();
  await acceptConfirmation(panel);
  await expect.poll(() => page.evaluate(() => queuedRevisionProbe.deleteStarted)).toBe(true);

  const secondToggle = secondCard.locator(".tool-enabled");
  await secondToggle.uncheck();
  await expect(secondToggle).toBeDisabled();
  const queued = await page.evaluate(() => queuedRevisionProbe.calls);
  expect(queued).toHaveLength(1);
  expect(queued[0]).toMatchObject({action:"delete", name:firstName});

  await page.evaluate(() => queuedRevisionProbe.release());
  await expect(firstCard).toHaveCount(0);
  await expect(secondToggle).toBeEnabled();
  await expect(secondToggle).not.toBeChecked();
  const result = await page.evaluate(() => ({
    calls:queuedRevisionProbe.calls,
    currentRevision:browserHarness.panel._configData.revision,
    tailReleased:browserHarness.panel._eocFunctionMutationTail === null,
    mutations:browserHarness.panel._eocAgentMutations,
  }));
  expect(result.calls).toHaveLength(2);
  expect(result.calls[1]).toMatchObject({action:"set_enabled", name:secondName, enabled:false});
  expect(result.calls[0].revision).not.toBe(result.calls[1].revision);
  expect(result.calls[1].revision).toBe(`${result.calls[0].revision}x`);
  expect(result.currentRevision).toBe(`${result.calls[1].revision}x`);
  expect(result.tailReleased).toBe(true);
  expect(result.mutations).toBe(0);

  await panel.evaluate(async host => {
    let observed;
    const original = host._hass.callWS.bind(host._hass);
    host._hass.callWS = async message => {
      if (message.section === "tools" && message.action === "set_enabled") observed = message.revision;
      return original(message);
    };
    await host._call("tools", "set_enabled", {name:"queued_toggle_b", enabled:true, revision:"caller-supplied-revision"}).catch(() => {});
    host._hass.callWS = original;
    window.explicitRevisionProbe = observed;
  });
  expect(await page.evaluate(() => explicitRevisionProbe)).toBe("caller-supplied-revision");
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
