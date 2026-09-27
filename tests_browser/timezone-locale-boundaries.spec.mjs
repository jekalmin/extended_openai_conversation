import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

for (const locale of ["en-GB", "en-US"]) {
  test.describe(`browser Honolulu / HA Dublin / ${locale}`, () => {
    test.use({timezoneId: "Pacific/Honolulu", locale});

    test("Usage sends HA-local dates and renders the same instant in HA time", async ({page}) => {
      const errors = trackPageErrors(page);
      await page.clock.install({time: new Date("2026-03-29T23:30:00Z")});
      const url = fixtureUrl("usage-maintenance/usage", "&usage_boundary=1&bundle=1");
      await page.goto(url);
      let panel = page.locator("extended-openai-management-panel");
      const daily = () => page.evaluate(() => window.browserHarness.calls.filter((call) => call.section === "usage" && call.action === "daily"));
      await expect.poll(daily).toContainEqual(expect.objectContaining({start_date: "2026-03-01", end_date: "2026-03-30"}));
      await expect(panel.locator(".chart-column")).toHaveCount(1);
      const expected = await page.evaluate((selectedLocale) => new Intl.DateTimeFormat(selectedLocale, {
        year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", timeZone: "Europe/Dublin",
      }).format(new Date("2026-03-29T23:30:00Z")), locale);
      await expect(panel.locator("[data-eoc-usage-runs] time")).toHaveText(expected);
      await expect(panel.locator("[data-eoc-usage-runs] time")).toHaveAttribute("datetime", "2026-03-29T23:30:00Z");
      await panel.locator("#usage-window").selectOption("7");
      await expect.poll(daily).toContainEqual(expect.objectContaining({start_date: "2026-03-24", end_date: "2026-03-30"}));
      await expect(panel.locator(".chart-column")).toHaveCount(1);

      await page.goto(url);
      panel = page.locator("extended-openai-management-panel");
      await expect(panel.locator("[data-eoc-usage-runs] time")).toHaveText(expected);
      await expect.poll(daily).toContainEqual(expect.objectContaining({start_date: "2026-03-01", end_date: "2026-03-30"}));
      await expectHarnessClean(page, errors);
    });
  });
}

test.describe("browser Kiritimati / HA Dublin wall clock", () => {
  test.use({timezoneId: "Pacific/Kiritimati", locale: "en-GB"});

  test("Quiet Hours saves a wall-clock time without browser timezone conversion", async ({page}) => {
    const errors = trackPageErrors(page);
    const url = fixtureUrl("capabilities/quiet-hours", "&bundle=1");
    await page.goto(url);
    let panel = page.locator("extended-openai-management-panel");
    await panel.locator("#qh-start").fill("23:00");
    await panel.locator("#save-page").click();
    await expect.poll(() => page.evaluate(() => window.browserHarness.getState().quiet.config.start)).toBe("23:00");
    const updates = await page.evaluate(() => window.browserHarness.calls.filter((call) => call.section === "quiet_hours" && call.action === "update"));
    expect(updates).toHaveLength(1);
    expect(updates[0].config.start).toBe("23:00");
    await page.goto(url);
    panel = page.locator("extended-openai-management-panel");
    await expect(panel.locator("#qh-start")).toHaveValue("23:00");
    await expectHarnessClean(page, errors);
  });
});
