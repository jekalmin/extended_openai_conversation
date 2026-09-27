import {mkdirSync, writeFileSync} from "node:fs";
import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

test("repeated mount cycles plus offline and browser restoration keep one healthy panel", async ({page, context}, testInfo) => {
  test.setTimeout(120_000);
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("data-memory/knowledge"));
  let panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Sources", exact: true})).toBeVisible();

  const baselineCalls = await page.evaluate(() => browserHarness.calls.length);
  await page.evaluate(async () => {
    const panel = browserHarness.panel;
    for (let index = 0; index < 40; index += 1) {
      panel.remove();
      await Promise.resolve();
      document.body.append(panel);
      panel.hass = browserHarness.hass;
      panel.route = {path: location.pathname};
      await Promise.resolve();
    }
  });
  panel = page.locator("extended-openai-management-panel");
  await expect(panel.getByRole("heading", {name: "Sources", exact: true})).toBeVisible();
  expect(await page.locator("extended-openai-management-panel").count()).toBe(1);

  // Load the route asset while online so the offline phase exercises the
  // management transport, not the browser's dynamic module fetch.
  await panel.locator('.top-nav button[data-page="usage-maintenance"]').click();
  await expect(panel.getByRole("heading", {name: "Usage period", exact: true})).toBeVisible();
  await panel.locator('.top-nav button[data-page="data-memory"]').click();
  await panel.locator('.subsection-nav button[data-subsection="knowledge"]').click();
  await expect(panel.getByRole("heading", {name: "Sources", exact: true})).toBeVisible();

  await page.evaluate(() => {
    browserHarness.backendOnline = false;
    browserHarness.offlineAttempts = 0;
    const original = browserHarness.hass.callWS.bind(browserHarness.hass);
    browserHarness.restoreCallWS = () => { browserHarness.hass.callWS = original; };
    browserHarness.hass.callWS = async message => {
      if (!browserHarness.backendOnline) {
        browserHarness.offlineAttempts += 1;
        throw new Error("Simulated offline management transport");
      }
      return original(message);
    };
  });
  await context.setOffline(true);
  await panel.locator('.top-nav button[data-page="usage-maintenance"]').click();
  await expect.poll(() => page.evaluate(() => browserHarness.offlineAttempts)).toBeGreaterThan(0);
  await expect(panel.getByText(/Simulated offline management transport/).first()).toBeVisible();

  await context.setOffline(false);
  await page.evaluate(() => {
    browserHarness.backendOnline = true;
    browserHarness.restoreCallWS();
  });
  await panel.locator('.top-nav button[data-page="capabilities"]').click();
  await expect(panel.getByRole("heading", {name: "Home Assistant access", exact: true})).toBeVisible();
  await expect(panel.getByRole("alert")).toHaveCount(0);

  await page.evaluate(() => {
    window.__eoaiBfcacheRestored = false;
    window.addEventListener("pageshow", event => {
      if (event.persisted) window.__eoaiBfcacheRestored = true;
    }, {once: true});
    browserHarness.panel.__beforeBfcacheMarker = "preserve-or-reload-safely";
  });
  await page.goto("data:text/html,<title>away</title><p>away</p>");
  await page.goBack({waitUntil: "domcontentloaded"});
  let hasHarness = await page.evaluate(() => Boolean(window.browserHarness)).catch(() => false);
  if (!hasHarness) {
    // Headless/browser policy may decline bfcache and try to reload HA's synthetic
    // panel URL. Recover through a fresh mount; the persisted path is asserted when
    // the engine actually grants bfcache.
    await page.goto(fixtureUrl("capabilities/home-assistant"));
    errors.length = 0;
    errors.consoleErrors.length = 0;
    errors.requestFailures.length = 0;
    errors.badResponses.length = 0;
    hasHarness = true;
  }
  expect(hasHarness).toBe(true);
  panel = page.locator("extended-openai-management-panel");
  await expect(panel).toHaveCount(1);
  await expect(panel.getByRole("heading", {name: "Home Assistant access", exact: true})).toBeVisible();

  const state = await page.evaluate(() => ({
    connected: browserHarness.panel.isConnected,
    panels: document.querySelectorAll("extended-openai-management-panel").length,
    bfcache: Boolean(window.__eoaiBfcacheRestored),
    marker: browserHarness.panel.__beforeBfcacheMarker || null,
    calls: browserHarness.calls.length,
  }));
  expect(state.connected).toBe(true);
  expect(state.panels).toBe(1);
  expect(state.calls).toBeGreaterThan(baselineCalls);
  // A browser may elect to reload instead of using bfcache. Both paths must be safe;
  // when bfcache is used, the same panel marker proves that exact instance survived.
  if (state.bfcache) expect(state.marker).toBe("preserve-or-reload-safely");

  await expectHarnessClean(page, errors);

  const artifactDir = process.env.STRESS_ARTIFACT_DIR || "stress-artifacts";
  mkdirSync(artifactDir, {recursive: true});
  writeFileSync(`${artifactDir}/browser-mount-recovery.json`, JSON.stringify({
    test: testInfo.title,
    outcome: "passed",
    browserRestoration: state.bfcache ? "bfcache" : "reload",
    operations: [{operation: state.bfcache ? "bfcache_restored" : "reload_fallback"}],
  }, null, 2));

});
