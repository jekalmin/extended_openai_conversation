import {test, expect} from "@playwright/test";
import {acceptConfirmation, expectHarnessClean, trackPageErrors} from "./browser-helpers.mjs";
import {openDataCollection, beginDataMeasure, finishDataMeasure} from "./data-collection-helpers.mjs";

let errors;
test.beforeEach(async ({page}) => { errors = trackPageErrors(page); });
test.afterEach(async ({page}) => { await expectHarnessClean(page, errors); });
const card = (panel, kind, id) => panel.locator(kind === "knowledge" ? `[data-source-id="source-${id}"]` : `[data-memory-id="memory-${id}"]`);
const list = (panel, kind) => panel.locator(kind === "knowledge" ? ".knowledge-list" : ".memory-list");
const action = (kind, operation) => kind === "knowledge" ? operation === "edit" ? ".source-edit-button" : ".delete-source" : operation === "edit" ? ".memory-edit-button" : ".delete-memory";
const editor = kind => kind === "knowledge" ? "knowledge" : "memory";
const field = kind => kind === "knowledge" ? "#knowledge-title" : "#memory-content";

for (const kind of ["knowledge", "persistent"]) {
  for (const bundled of [false, true]) {
    test(`${kind}: edit, delete and add preserve unrelated cards (${bundled ? "bundle" : "source"})`, async ({page}) => {
      const panel = await openDataCollection(page, kind, 80, bundled);
      if (kind === "persistent") await panel.evaluate(host => {
        host._data.scopes.find(scope => scope.scope_id === host._scopeId).memory_count = 80;
        host._selectedAgent().memory_count = 80;
      });
      const initialReads = await page.evaluate(() => ({
        lists: dataCollectionBackend.calls.filter(call => ["list", "search", "temporary_list"].includes(call.action)).length,
        scopes: browserHarness.calls.filter(call => call.section === "scopes" && call.action === "catalog").length,
      }));
      await card(panel, kind, 30).locator(action(kind, "edit")).click();
      await expect(panel.locator(field(kind))).toBeEnabled();
      await panel.locator(field(kind)).fill("Changed one record");
      await beginDataMeasure(page);
      await panel.locator(`#${editor(kind)}-save`).click();
      await expect(list(panel, kind).getByText("Changed one record", {exact: true})).toBeVisible();
      expect(await finishDataMeasure(page)).toMatchObject({retainedCards: 79, initialCards: 80, mainChildReplacements: 0, inputRetained: true});
      await card(panel, kind, 20).locator(action(kind, "delete")).click();
      await beginDataMeasure(page); await acceptConfirmation(panel);
      await expect(card(panel, kind, 20)).toHaveCount(0);
      if (kind === "knowledge") await expect(panel.locator("[data-source-count]")).toHaveText("79 sources");
      expect(await finishDataMeasure(page)).toMatchObject({
        retainedCards: 79, mainChildReplacements: 0,
        ...(kind === "persistent" ? {routeRenders: 0} : {}),
      });
      if (kind === "persistent") expect(await panel.evaluate(host => ({
        selectedAgentCount: host._selectedAgent().memory_count,
        scopeCount: host._data.scopes.find(scope => scope.scope_id === host._scopeId).memory_count,
        localTotal: host._result.total,
      }))).toEqual({selectedAgentCount: 79, scopeCount: 79, localTotal: 79});
      await panel.locator(kind === "knowledge" ? "#add-source" : "#add-memory").click();
      await panel.locator(field(kind)).fill("Brand new record");
      if (kind === "knowledge") await panel.locator("#knowledge-content").fill("Locally stored source content");
      await beginDataMeasure(page);
      await panel.locator(`#${editor(kind)}-save`).click();
      await expect(list(panel, kind).getByText("Brand new record", {exact: true})).toBeVisible();
      if (kind === "knowledge") await expect(panel.locator("[data-source-count]")).toHaveText("80 sources");
      else expect(await panel.evaluate(host => host._data.scopes.find(scope => scope.scope_id === host._scopeId).memory_count)).toBe(80);
      expect(await finishDataMeasure(page)).toMatchObject({retainedCards: 79, initialCards: 79, mainChildReplacements: 0});
      expect(await page.evaluate(() => ({
        lists: dataCollectionBackend.calls.filter(call => ["list", "search", "temporary_list"].includes(call.action)).length,
        scopes: browserHarness.calls.filter(call => call.section === "scopes" && call.action === "catalog").length,
      }))).toEqual(initialReads);
    });
  }

  test(`${kind}: search keeps cards, focus, caret and route out of the typing path`, async ({page}) => {
    const panel = await openDataCollection(page, kind);
    const search = panel.locator("#list-search");
    await beginDataMeasure(page);
    await search.fill(kind === "knowledge" ? "Source 1" : "Memory 1");
    await search.evaluate(node => node.setSelectionRange(2, 5));
    if (kind === "persistent") await page.waitForFunction(() => browserHarness.panel._managementBrowserState.memoryQuery === "memory 1");
    await expect(list(panel, kind).locator("article:visible")).toHaveCount(11);
    expect(await search.evaluate(node => [node.selectionStart, node.selectionEnd])).toEqual([2, 5]);
    expect(await finishDataMeasure(page)).toMatchObject({retainedCards: 100, initialCards: 100, elementsAdded: 0, elementsRemoved: 0, routeRenders: 0, mainChildReplacements: 0, inputRetained: true, inputFocused: true});
    await search.fill("no fixture matches");
    await expect(list(panel, kind).locator("article:visible")).toHaveCount(0);
    await expect(list(panel, kind).locator(".empty:visible")).toHaveCount(1);
    await search.fill("");
    if (kind === "persistent") await page.waitForFunction(() => browserHarness.panel._managementBrowserState.memoryQuery === "");
    await expect(list(panel, kind).locator("article:visible")).toHaveCount(100);
  });

  test(`${kind}: repeated binding and keyboard/card actions fire once`, async ({page}) => {
    const panel = await openDataCollection(page, kind);
    await panel.evaluate(host => { for (let i = 0; i < 4; i++) { host._render(); host._bindActions(); } });
    const before = await page.evaluate(() => dataCollectionBackend.calls.length);
    const target = card(panel, kind, 0).locator(kind === "knowledge" ? ".edit-source" : ".edit-memory");
    await target.focus(); await target.press("Enter");
    await expect(panel.locator(`#${editor(kind)}-dialog`)).toHaveJSProperty("open", true);
    await expect(panel.locator(field(kind))).toBeEnabled();
    await panel.locator(field(kind)).fill("Exactly once");
    await panel.locator(`#${editor(kind)}-save`).click();
    await expect(list(panel, kind).getByText("Exactly once", {exact: true})).toBeVisible();
    expect(await page.evaluate(before => dataCollectionBackend.calls.slice(before).filter(call => call.action === "update").length, before)).toBe(1);
    if (kind === "knowledge") expect(await page.evaluate(before => dataCollectionBackend.calls.slice(before).filter(call => call.action === "get").length, before)).toBe(1);
  });

  test(`${kind}: failed save leaves the editor and collection authoritative`, async ({page}) => {
    const panel = await openDataCollection(page, kind);
    await card(panel, kind, 1).locator(action(kind, "edit")).click();
    await expect(panel.locator(field(kind))).toBeEnabled();
    await panel.locator(field(kind)).fill("Rejected edit");
    await page.evaluate(() => {
      const original = browserHarness.hass.callWS;
      browserHarness.hass.callWS = message => message.action === "update" && ["knowledge", "memories"].includes(message.section) ? Promise.reject(new Error("Fixture rejected save")) : original(message);
    });
    await beginDataMeasure(page); await panel.locator(`#${editor(kind)}-save`).click();
    await expect(panel.locator(`#${editor(kind)}-error`)).toContainText("Fixture rejected save");
    expect(await finishDataMeasure(page)).toMatchObject({retainedCards: 100, elementsAdded: 0, elementsRemoved: 0, mainChildReplacements: 0});
    await expect(panel.locator(field(kind))).toHaveValue("Rejected edit");
    await expect(card(panel, kind, 1)).not.toContainText("Rejected edit");
  });
}

test("Knowledge: stale cache is shown while a changed authoritative response reconciles", async ({page}) => {
  const panel = await openDataCollection(page, "knowledge", 60);
  await page.evaluate(() => {
    const backend = dataCollectionBackend;
    backend.state.sources[1].title = "Refreshed source";
    backend.state.sources[1].enabled = false;
    backend.state.sources.splice(2, 1);
    backend.state.sources.push({...backend.state.sources[0], source_id: "new-source", title: "New source"});
    const original = browserHarness.hass.callWS;
    browserHarness.hass.callWS = async message => {
      const result = await original(message);
      if (message.section === "knowledge" && message.action === "list") return new Promise(resolve => { window.releaseKnowledge = () => resolve(result); });
      return result;
    };
  });
  await beginDataMeasure(page);
  await panel.evaluate(host => {
    host._eocSectionCacheTimes.set(host._sectionCacheKey(), 1);
    window.pendingRefresh = host._loadSection(true);
  });
  await page.waitForFunction(() => window.releaseKnowledge);
  await expect(card(panel, "knowledge", 1)).toContainText("Source 1");
  await expect(panel.locator("main .loading")).toHaveCount(0);
  await page.evaluate(async () => { releaseKnowledge(); await pendingRefresh; });
  await expect(card(panel, "knowledge", 1)).toContainText("Refreshed source");
  await expect(card(panel, "knowledge", 1)).toContainText("Unavailable");
  await expect(card(panel, "knowledge", 2)).toHaveCount(0);
  await expect(panel.locator('[data-source-id="new-source"]')).toBeVisible();
  expect(await finishDataMeasure(page)).toMatchObject({retainedCards: 58, initialCards: 60, mainChildReplacements: 0, inputRetained: true});
  await beginDataMeasure(page);
  await panel.evaluate(host => { host._eocSectionCacheTimes.set(host._sectionCacheKey(), 1); });
  await panel.evaluate(host => { window.unchangedRefresh = host._loadSection(true); });
  await page.evaluate(async () => { releaseKnowledge(); await unchangedRefresh; });
  expect(await finishDataMeasure(page)).toMatchObject({retainedCards: 60, elementsAdded: 0, elementsRemoved: 0, mainChildReplacements: 0});
});

for (const kind of ["knowledge", "persistent", "temporary"]) {
  test(`${kind}: background refresh preserves an open editor and flushes on close`, async ({page}) => {
    const panel = await openDataCollection(page, kind);
    const temporary = kind === "temporary";
    const selector = temporary ? '[data-memory-id="temporary-1"]' : kind === "knowledge" ? '[data-source-id="source-1"]' : '[data-memory-id="memory-1"]';
    const fieldSelector = temporary ? "#temporary-memory-content" : field(kind);
    await panel.locator(selector).locator(temporary ? "button.edit-temporary-memory" : action(kind, "edit")).click();
    await expect(panel.locator(fieldSelector)).toBeEnabled();
    await panel.locator(fieldSelector).fill("Unsaved editor contents");
    await panel.locator(fieldSelector).evaluate(node => { window.editorNode = node; });
    await page.evaluate(async kind => {
      const records = kind === "knowledge" ? dataCollectionBackend.state.sources : kind === "temporary" ? dataCollectionBackend.state.temporary : dataCollectionBackend.state.memories;
      records[1][kind === "knowledge" ? "title" : "content"] = "Backend updated contents";
      const host = browserHarness.panel;
      if (kind === "knowledge") host._eocSectionCacheTimes.set(host._sectionCacheKey(), 1);
      await host._loadSection(true);
    }, kind);
    await expect(panel.locator(fieldSelector)).toHaveValue("Unsaved editor contents");
    expect(await panel.locator(fieldSelector).evaluate(node => node === window.editorNode)).toBe(true);
    await expect(panel.locator(selector)).not.toContainText("Backend updated contents");
    await panel.locator(temporary ? "#temporary-memory-dialog .dialog-actions .close-temporary-editor" : `#${editor(kind)}-dialog .dialog-actions .close-editor`).click();
    await expect(panel.locator(selector)).toContainText("Backend updated contents");
  });
}

test("Persistent Memory: category edits and pagination retain other loaded records", async ({page}) => {
  const panel = await openDataCollection(page, "persistent", 230);
  await beginDataMeasure(page); await panel.locator("#load-more-memories").click();
  await expect(panel.locator(".memory-list article")).toHaveCount(200);
  expect(await finishDataMeasure(page)).toMatchObject({retainedCards: 100, initialCards: 100, mainChildReplacements: 0, routeRenders: 0});
  await panel.locator("#load-more-memories").click();
  await expect(panel.locator(".memory-list article")).toHaveCount(230);
  await expect(panel.locator("#load-more-memories")).toBeHidden();
  // Search beyond the initial page preserves the already-loaded card nodes.
  await beginDataMeasure(page); await panel.locator("#list-search").fill("Memory 220");
  await page.waitForFunction(() => browserHarness.panel._managementBrowserState.memoryQuery === "memory 220");
  await expect(panel.locator(".memory-list article:visible")).toHaveCount(1);
  expect(await finishDataMeasure(page)).toMatchObject({retainedCards: 230, elementsAdded: 0, elementsRemoved: 0, routeRenders: 0});
  await card(panel, "persistent", 220).locator(".memory-edit-button").click();
  await panel.locator("#memory-category").fill("Moved category");
  await panel.locator("#memory-save").click();
  await expect(card(panel, "persistent", 220)).toContainText("Moved category");
});

async function installOtherScope(page, temporary) {
  await page.evaluate(temporary => {
    const host = browserHarness.panel, original = browserHarness.hass.callWS;
    const other = {scope_id: "shared:household", scope_type: "shared", display_name: "Shared household", memory_count: 1, temporary_memory_count: 1};
    const scopes = [...host._data.scopes, other];
    browserHarness.hass.callWS = message => message.section === "scopes" && message.action === "catalog" ? Promise.resolve({scopes}) : original(message);
    host._data.scopes = scopes; host._baseScopes = scopes;
    host._scopeCatalogCache.clear(); host._eocScopeCatalogTimes.clear();
    const records = temporary ? dataCollectionBackend.state.temporary : dataCollectionBackend.state.memories;
    records.push({...records[0], scope_id: other.scope_id, owner_scope_id: other.scope_id, content: "Memory belonging to Shared household"});
    host._render();
  }, temporary);
}

for (const temporary of [false, true]) {
  test(`${temporary ? "Temporary" : "Persistent"} Memory: same IDs do not leak across scope or kind`, async ({page}) => {
    const panel = await openDataCollection(page, temporary ? "temporary" : "persistent");
    await installOtherScope(page, temporary);
    await panel.locator(".memory-list article").first().evaluate(node => { window.oldScopeCard = node; });
    await panel.locator("#scope").selectOption("shared:household");
    await expect(panel.locator(".memory-list article")).toHaveCount(1);
    await expect(panel.locator(".memory-list article")).toContainText("Memory belonging to Shared household");
    expect(await page.evaluate(() => oldScopeCard.isConnected)).toBe(false);
    await panel.locator("#scope").selectOption("user:test-user");
    await expect(panel.locator(".memory-list article")).toHaveCount(temporary ? 12 : 100);
    await panel.locator(`.memory-kind[data-kind="${temporary ? "persistent" : "temporary"}"]`).click();
    await expect(panel.locator(".memory-list article")).toHaveCount(temporary ? 100 : 12);
    await expect(panel.locator(temporary ? "[data-persistent-memories]" : "[data-temporary-memories]")).toBeVisible();
  });
}

test("Persistent Memory: delayed search from the old scope is ignored", async ({page}) => {
  const panel = await openDataCollection(page, "persistent"); await installOtherScope(page, false);
  await page.evaluate(() => {
    const original = browserHarness.hass.callWS;
    browserHarness.hass.callWS = async message => {
      const result = await original(message);
      if (message.action === "search" && message.scope_id === "user:test-user") return new Promise(resolve => { window.releaseSearch = () => resolve(result); });
      return result;
    };
  });
  await panel.locator("#list-search").fill("Memory");
  await page.waitForFunction(() => window.releaseSearch);
  await panel.locator("#scope").selectOption("shared:household");
  await expect(panel.locator(".memory-list article:visible")).toHaveCount(1);
  await page.evaluate(() => releaseSearch());
  await expect(panel.locator(".memory-list article")).toHaveCount(1);
  await expect(panel.locator(".memory-list article")).toContainText("Memory belonging to Shared household");
});

test("Temporary Memory: delete, expiry refresh and clear touch only affected items", async ({page}) => {
  const panel = await openDataCollection(page, "temporary");
  await panel.evaluate(host => { for (let i = 0; i < 3; i++) { host._render(); host._bindActions(); } });
  await panel.locator('[data-memory-id="temporary-1"] .delete-temporary').click();
  await beginDataMeasure(page); await acceptConfirmation(panel);
  await expect(panel.locator(".memory-list article")).toHaveCount(11);
  expect(await finishDataMeasure(page)).toMatchObject({retainedCards: 11, initialCards: 12, mainChildReplacements: 0});
  expect(await page.evaluate(() => dataCollectionBackend.calls.filter(call => call.action === "temporary_delete").length)).toBe(1);
  await beginDataMeasure(page);
  await page.evaluate(async () => { dataCollectionBackend.state.temporary.splice(0, 1); await browserHarness.panel._loadSection(true); });
  expect(await finishDataMeasure(page)).toMatchObject({retainedCards: 10, initialCards: 11, mainChildReplacements: 0});
  await beginDataMeasure(page);
  await panel.locator("#clear-temporary").click();
  await acceptConfirmation(panel);
  await expect(panel.locator(".memory-list article")).toHaveCount(0);
  expect(await finishDataMeasure(page)).toMatchObject({mainChildReplacements: 0, inputRetained: true});
});

for (const kind of ["persistent", "temporary"]) {
  test(`${kind}: ordinary same-scope refresh retains cards even when source content is identical`, async ({page}) => {
    const panel = await openDataCollection(page, kind);
    const count = kind === "persistent" ? 100 : 12;
    await beginDataMeasure(page);
    await panel.evaluate(host => host._loadSection());
    expect(await finishDataMeasure(page)).toMatchObject({retainedCards: count, elementsAdded: 0, elementsRemoved: 0, mainChildReplacements: 0, inputRetained: true});
    await installOtherScope(page, kind === "temporary");
    await page.evaluate(kind => {
      const records = kind === "temporary" ? dataCollectionBackend.state.temporary : dataCollectionBackend.state.memories;
      const other = records.find(record => record.scope_id === "shared:household");
      Object.assign(other, {...records[0], scope_id: "shared:household", owner_scope_id: "shared:household"});
    }, kind);
    await panel.locator(".memory-list article").first().evaluate(node => { window.scopeNode = node; });
    await panel.locator("#scope").selectOption("shared:household");
    await expect(panel.locator(".memory-list article")).toHaveCount(1);
    expect(await page.evaluate(() => scopeNode.isConnected)).toBe(false);
  });
}

test("Temporary Memory: edit metadata and expiry submits once and retains unrelated items", async ({page}) => {
  const panel = await openDataCollection(page, "temporary");
  await panel.evaluate(host => { host._bindActions(); host._bindActions(); });
  await panel.locator('[data-memory-id="temporary-4"] button.edit-temporary-memory').click();
  await panel.locator("#temporary-memory-content").fill("Edited short-term detail");
  await panel.locator("#temporary-memory-category").fill("new-category");
  await panel.locator("#temporary-memory-expiry").fill("2026-10-01T18:00:00+01:00");
  await beginDataMeasure(page); await panel.locator("#temporary-memory-save").click();
  await expect(panel.locator('[data-memory-id="temporary-4"]')).toContainText("Edited short-term detail");
  expect(await finishDataMeasure(page)).toMatchObject({retainedCards: 11, initialCards: 12, mainChildReplacements: 0});
  expect(await page.evaluate(() => dataCollectionBackend.calls.filter(call => call.action === "temporary_update").length)).toBe(1);
});

for (const kind of ["knowledge", "persistent"]) {
  test(`${kind}: a pending save does not optimistically change the card`, async ({page}) => {
    const panel = await openDataCollection(page, kind);
    await card(panel, kind, 1).locator(action(kind, "edit")).click();
    await expect(panel.locator(field(kind))).toBeEnabled();
    await panel.locator(field(kind)).fill("Confirmed change only");
    await page.evaluate(() => {
      const original = browserHarness.hass.callWS;
      browserHarness.hass.callWS = async message => {
        const result = await original(message);
        if (["knowledge", "memories"].includes(message.section) && message.action === "update") return new Promise(resolve => { window.releaseMutation = () => resolve(result); });
        return result;
      };
    });
    await panel.locator(`#${editor(kind)}-save`).click();
    await page.waitForFunction(() => window.releaseMutation);
    await expect(card(panel, kind, 1)).not.toContainText("Confirmed change only");
    await expect(panel.locator(`#${editor(kind)}-dialog`)).toHaveJSProperty("open", true);
    await page.evaluate(() => releaseMutation());
    await expect(card(panel, kind, 1)).toContainText("Confirmed change only");
  });
}

test("Knowledge: an old agent's background response cannot repopulate the new agent", async ({page}) => {
  const panel = await openDataCollection(page, "knowledge", 60);
  await page.evaluate(() => {
    const host = browserHarness.panel;
    host._data.agents.push({...host._selectedAgent(), subentry_id: "second-agent", title: "Second agent"}); host._render();
    const original = browserHarness.hass.callWS;
    browserHarness.hass.callWS = async message => {
      if (message.section === "knowledge" && message.action === "list" && message.subentry_id === "second-agent") return {sources: [{...dataCollectionBackend.state.sources[0], title: "Second agent source"}]};
      const result = await original(message);
      if (message.section === "knowledge" && message.action === "list") return new Promise(resolve => { window.releaseOldAgent = () => resolve(result); });
      return result;
    };
    host._eocSectionCacheTimes.set(host._sectionCacheKey(), 1); window.oldAgentLoad = host._loadSection(true);
  });
  await page.waitForFunction(() => window.releaseOldAgent);
  await panel.locator(".knowledge-list article").first().evaluate(node => { window.oldAgentSource = node; });
  await panel.locator("#agent").selectOption("second-agent");
  await expect(panel.locator(".knowledge-list article")).toHaveCount(1);
  await expect(panel.locator(".knowledge-list article")).toContainText("Second agent source");
  await page.evaluate(async () => { releaseOldAgent(); await oldAgentLoad; });
  await expect(panel.locator(".knowledge-list article")).toHaveCount(1);
  expect(await page.evaluate(() => oldAgentSource.isConnected)).toBe(false);
});

test("Persistent Memory: a late load-more response cannot replace a newer search", async ({page}) => {
  const panel = await openDataCollection(page, "persistent", 230);
  await page.evaluate(() => {
    const original = browserHarness.hass.callWS;
    browserHarness.hass.callWS = async message => {
      const result = await original(message);
      if (message.action === "list" && message.offset === 100) return new Promise(resolve => { window.releaseMore = () => resolve(result); });
      return result;
    };
  });
  await panel.locator("#load-more-memories").click();
  await page.waitForFunction(() => window.releaseMore);
  await panel.locator("#list-search").fill("Memory 220");
  await page.waitForFunction(() => browserHarness.panel._managementBrowserState.memoryQuery === "memory 220");
  await page.evaluate(() => releaseMore());
  await expect(panel.locator(".memory-list article:visible")).toHaveCount(1);
  await expect(panel.locator(".memory-list article:visible")).toContainText("Memory 220");
  await expect(panel.locator("#load-more-memories")).toBeHidden();
});

test("Knowledge: retained availability control can save repeatedly without duplicate requests", async ({page}) => {
  const panel = await openDataCollection(page, "knowledge", 60);
  await panel.evaluate(host => { host._bindActions(); host._bindActions(); });
  const toggle = panel.locator("#knowledge-enabled-toggle");
  const initial = await toggle.isChecked();
  await toggle.setChecked(!initial);
  await expect.poll(() => page.evaluate(() => browserHarness.calls.filter(call => call.section === "knowledge" && call.action === "set_enabled").length)).toBe(1);
  await expect(toggle).toBeEnabled();
  await toggle.setChecked(initial);
  await expect.poll(() => page.evaluate(() => browserHarness.calls.filter(call => call.section === "knowledge" && call.action === "set_enabled").length)).toBe(2);
  await expect(toggle).toBeEnabled();
});
