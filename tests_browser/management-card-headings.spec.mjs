import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

const CARDS = [
  ["usage-maintenance/backup-restore", "Export, Backup, Import & Restore"],
  ["usage-maintenance/retention", "Retention periods"],
  ["usage-maintenance/diagnostics", "Test assistant"],
  ["usage-maintenance/usage", "Recent runs"],
  ["usage-maintenance/request-debug", "Recent debug runs"],
];

test("management content cards share heading type and structural spacing", async ({page}) => {
  const errors = trackPageErrors(page);
  const metrics = [];
  for (const [route, title] of CARDS) {
    await page.goto(fixtureUrl(route));
    const heading = page.getByRole("heading", {name: title, exact: true});
    await expect(heading).toBeVisible();
    metrics.push(await heading.evaluate((element) => {
      const card = element.closest(".content-card,.card");
      const block = element.closest(".card-heading");
      const description = block.querySelector("p");
      const body = block.nextElementSibling;
      const style = getComputedStyle(element);
      return {
        size: style.fontSize,
        weight: style.fontWeight,
        lineHeight: style.lineHeight,
        titleMargin: style.margin,
        top: Math.round(block.getBoundingClientRect().top - card.getBoundingClientRect().top),
        descriptionGap: Math.round(description.getBoundingClientRect().top - element.getBoundingClientRect().bottom),
        bodyGap: Math.round(body.getBoundingClientRect().top - block.getBoundingClientRect().bottom),
      };
    }));
  }
  for (const metric of metrics) expect(metric).toEqual(metrics[0]);
  expect(metrics[0]).toMatchObject({size: "19px", weight: "600", titleMargin: "0px", descriptionGap: 6, bodyGap: 20});
  await expectHarnessClean(page, errors);
});
