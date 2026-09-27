import {expect, test} from "@playwright/test";
import {readFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
import {NAVIGATION} from "../custom_components/extended_openai_conversation_responses/frontend/frontend-navigation.js";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

const inventoryPath = fileURLToPath(new URL("../tests_stress/frontend_route_inventory.json", import.meta.url));
const inventory = JSON.parse(readFileSync(inventoryPath, "utf8"));
const routes = NAVIGATION.flatMap(page => page.sections.length
  ? page.sections.map(section => `${page.id}/${section.id}`) : [page.id]);

test("every shipped route has an explicit nightly browser acceptance level", () => {
  expect(inventory.schema_version).toBe(2);
  expect(Object.keys(inventory.routes).sort()).toEqual([...routes].sort());
  expect(Object.values(inventory.routes).every(level =>
    ["full-crud", "read-write", "read-only", "render-navigation"].includes(level))).toBe(true);
  expect(Object.keys(inventory.journeys).sort()).toEqual([...routes].sort());
});

for (const [viewport, width] of [["mobile", 390], ["tablet", 820], ["desktop", 1600]]) {
  test(`all management routes navigate, settle and refresh at ${viewport} width`, async ({page}) => {
    const errors = trackPageErrors(page);
    await page.setViewportSize({width, height: 850});
    await page.goto(fixtureUrl("overview"));
    const panel = page.locator("extended-openai-management-panel");
    const choosePage = async pageId => width < 600
      ? panel.locator("#top-section-mobile").selectOption(pageId)
      : panel.locator(`.top-nav button[data-page="${pageId}"]`).click();
    for (const route of routes) {
      const [pageId, subsection] = route.split("/");
      await choosePage(pageId);
      if (subsection) {
        await panel.locator("#local-section").selectOption(subsection, {force: true});
      }
      await expect(page, route).toHaveURL(new RegExp(`/extended-openai/${route}$`));
      const main = panel.locator("[data-eoc-main]");
      await expect(main, route).toBeVisible();
      await expect(main, route).not.toContainText("Loading…");
      await expect(panel.locator('[role="alert"]:visible'), route).toHaveCount(0);
      // The local fixture server has no SPA fallback. Re-enter through the
      // shipped fixture document to exercise a fresh page load for this route.
      await page.goto(fixtureUrl(route));
      await expect(main, `${route} after refresh`).toBeVisible();
      await expect(main, `${route} after refresh`).not.toContainText("Loading…");
      await expect(panel.locator('[role="alert"]:visible'), route).toHaveCount(0);
      if (route !== "guide") {
        const calls = await page.evaluate(() => window.browserHarness.calls.length);
        expect(calls, `${route} should finish a management request`).toBeGreaterThan(0);
      }
      await choosePage("overview");
      await expect(page).toHaveURL(/\/extended-openai\/overview$/);
      await choosePage(pageId);
      if (subsection) await panel.locator("#local-section").selectOption(subsection, {force: true});
      await expect(page).toHaveURL(new RegExp(`/extended-openai/${route}$`));
    }
    await expectHarnessClean(page, errors);
  });
}
