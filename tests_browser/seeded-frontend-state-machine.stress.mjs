import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

const seeds = [7319, 20260930, 0x5eed];
const orderForSeed = seed => {
  let value = seed >>> 0;
  const operations = ["knowledge", "memory", "assistant"];
  for (let index = operations.length - 1; index > 0; index -= 1) {
    value = (Math.imul(value, 1664525) + 1013904223) >>> 0;
    const swap = value % (index + 1);
    [operations[index], operations[swap]] = [operations[swap], operations[index]];
  }
  return operations;
};

for (const seed of seeds) {
  test(`seeded cross-surface state machine seed=${seed}`, async ({page}, testInfo) => {
    const errors = trackPageErrors(page);
    const history = [];
    const model = {knowledgeTitle: `Seed ${seed} source`, memoryContent: `Seed ${seed} memory`, assistantTitle: `Seed ${seed} assistant`};
    try {
      for (const operation of orderForSeed(seed)) {
        if (operation === "knowledge") {
          await page.goto(fixtureUrl("data-memory/knowledge"));
          const panel = page.locator("extended-openai-management-panel");
          await panel.locator("#list-search").fill("Seed");
          await panel.locator("#add-source").click();
          await panel.locator("#knowledge-title").fill(model.knowledgeTitle);
          await panel.locator("#knowledge-description").fill(`Reference ${seed}`);
          await panel.locator("#knowledge-content").fill(`Seeded state machine body ${seed}`);
          await panel.locator("#knowledge-save").click();
          await expect(panel.locator("#knowledge-dialog")).not.toHaveJSProperty("open", true);
          await expect(panel.locator("[data-source-id]").filter({hasText: model.knowledgeTitle})).toHaveCount(1);
          history.push("Knowledge create and reconcile");
          await panel.evaluate(host => host._loadSection(true));
          await expect(panel.locator("[data-source-id]").filter({hasText: model.knowledgeTitle})).toHaveCount(1);
          history.push("Knowledge authoritative list reload");
        } else if (operation === "memory") {
          await page.goto(fixtureUrl("data-memory/memories"));
          const panel = page.locator("extended-openai-management-panel");
          await panel.locator("#add-memory").click();
          await panel.locator("#memory-content").fill(model.memoryContent);
          await panel.locator("#memory-category").fill(`seed-${seed}`);
          await panel.locator("#memory-save").click();
          await expect(panel.locator(".memory-list")).toContainText(model.memoryContent);
          history.push("Memory create and reconcile");
          await panel.locator("#list-search").fill(model.memoryContent);
          await expect(panel.locator("[data-memory-id]:visible")).toHaveCount(1);
          history.push("Memory search");
          await panel.evaluate(host => host._loadSection(true));
          await expect(panel.locator(".memory-list")).toContainText(model.memoryContent);
          history.push("Memory authoritative list reload");
        } else {
          await page.goto(fixtureUrl("assistant/basics"));
          const panel = page.locator("extended-openai-management-panel");
          await panel.locator('[data-config="__title"]').fill(model.assistantTitle);
          await panel.getByRole("button", {name: "Save changes", exact: true}).click();
          await expect(panel.locator('[data-config="__title"]')).toHaveValue(model.assistantTitle);
          history.push("Assistant title save");
          await panel.evaluate(async host => { await host._navigate("assistant", "conversation"); await host._navigate("assistant", "basics"); });
          await expect(panel.locator('[data-config="__title"]')).toHaveValue(model.assistantTitle);
          history.push("Assistant route reopen");
        }
      }
      await expectHarnessClean(page, errors);
    } catch (error) {
      await testInfo.attach(`state-machine-seed-${seed}`, {body: JSON.stringify({seed, plannedOrder: orderForSeed(seed), history, model}, null, 2), contentType: "application/json"});
      throw error;
    }
  });
}
