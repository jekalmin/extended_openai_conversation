import {expect, test} from "@playwright/test";
import {expectHarnessClean, fixtureUrl, trackPageErrors} from "./browser-helpers.mjs";

// Reproduce the real HA cold-load case where HTTP/1.1 delays the full stylesheet.
for (const bundled of [false, true]) {
test(`critical CSS prevents shell and route-title FOUC while full stylesheet is delayed (${bundled ? "bundle" : "source"})`, async ({page}) => {
  const errors = trackPageErrors(page);
  let releaseStylesheet;
  const stylesheetRequested = new Promise((resolve) => {
    page.route(/\/management[^/]*\.css$/, async (route) => {
      resolve();
      await new Promise((release) => { releaseStylesheet = release; });
      await route.continue();
    });
  });

  await page.goto(fixtureUrl("capabilities/home-assistant", bundled ? "&bundle=1" : ""), {waitUntil:"domcontentloaded"});
  await stylesheetRequested;
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".page-heading h1")).toHaveText("Extended OpenAI");
  await expect(panel.locator(".page-intro h1")).toHaveText("Home Assistant access");

  const before = await panel.evaluate((host) => {
    const root = host.shadowRoot;
    const shellTitle = root.querySelector(".page-heading h1");
    const routeTitle = root.querySelector(".page-intro h1");
    const routeIntro = root.querySelector(".page-intro");
    const rect = (node) => {
      const box = node.getBoundingClientRect();
      return {x:box.x,y:box.y,width:box.width,height:box.height};
    };
    const textRect = (node) => {
      const range = document.createRange();
      range.selectNodeContents(node);
      const box = range.getBoundingClientRect();
      return {x:box.x,y:box.y,width:box.width,height:box.height};
    };
    return {
      critical: Boolean(root.querySelector("style[data-eoc-critical-styles]")),
      fullLoaded: Boolean(root.querySelector("link[data-eoc-persistent-styles]")?.sheet),
      hostPadding: getComputedStyle(host).paddingTop,
      headerDisplay: getComputedStyle(root.querySelector("header")).display,
      titleSize: getComputedStyle(shellTitle).fontSize,
      mainDisplay: getComputedStyle(root.querySelector("main")).display,
      mainGap: getComputedStyle(root.querySelector("main")).gap,
      mobileNav: getComputedStyle(root.querySelector(".mobile-nav")).display,
      shellTitle: textRect(shellTitle),
      routeTitle: textRect(routeTitle),
      routeIntro: rect(routeIntro),
    };
  });

  expect(before).toMatchObject({
    critical:true,
    fullLoaded:false,
    hostPadding:"28px",
    headerDisplay:"flex",
    titleSize:"30px",
    mainDisplay:"grid",
    mainGap:"30px",
    mobileNav:"none",
  });

  releaseStylesheet();
  await expect.poll(() => panel.evaluate((host) => Boolean(host.shadowRoot.querySelector("link[data-eoc-persistent-styles]")?.sheet))).toBe(true);

  const after = await panel.evaluate((host) => {
    const root = host.shadowRoot;
    const rect = (selector) => {
      const box = root.querySelector(selector).getBoundingClientRect();
      return {x:box.x,y:box.y,width:box.width,height:box.height};
    };
    const textRect = (selector) => {
      const range = document.createRange();
      range.selectNodeContents(root.querySelector(selector));
      const box = range.getBoundingClientRect();
      return {x:box.x,y:box.y,width:box.width,height:box.height};
    };
    return {
      shellTitle:textRect(".page-heading h1"),
      routeTitle:textRect(".page-intro h1"),
      routeIntro:rect(".page-intro"),
    };
  });

  for (const key of ["shellTitle","routeTitle","routeIntro"]) {
    for (const dimension of ["x","y","width","height"]) {
      expect(Math.abs(after[key][dimension] - before[key][dimension]), `${key} ${dimension}`).toBeLessThanOrEqual(1);
    }
  }
  await expectHarnessClean(page, errors);
});
}

for (const bundled of [false, true]) {
  test(`management shell uses external stylesheet (${bundled ? "bundle" : "source"})`, async ({page}) => {
    const errors = trackPageErrors(page);
    const requested = [];
    page.on("request", request => requested.push(new URL(request.url()).pathname));
    await page.goto(fixtureUrl("overview", bundled ? "&bundle=1" : ""));
    const panel = page.locator("extended-openai-management-panel");
    await expect(panel.locator(".dashboard-grid")).toBeVisible();

    const style = await panel.evaluate((host) => {
      const root = host.shadowRoot;
      const link = root.querySelector('link[data-eoc-persistent-styles]');
      return {
        href: link?.href || "",
        inlineStyles: root.querySelectorAll("style[data-eoc-persistent-styles]").length,
      };
    });
    expect(style.inlineStyles).toBe(0);
    expect(style.href).toMatch(bundled
      ? /\/frontend\/dist\/assets\/management-[^/]+\.css$/
      : /\/frontend\/management\.css$/);
    expect(requested.some(pathname => bundled
      ? /\/frontend\/dist\/assets\/management-[^/]+\.css$/.test(pathname)
      : pathname.endsWith("/frontend/management.css"))).toBe(true);
    await expectHarnessClean(page, errors);
  });
}

test("the stylesheet starts with the panel and survives a shell render without another request", async ({page}) => {
  const requested = [];
  page.on("request", request => {
    if (/\/management[^/]*\.css$/.test(new URL(request.url()).pathname)) requested.push(request.url());
  });
  await page.goto(fixtureUrl("overview"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".dashboard-grid")).toBeVisible();
  const stable = await panel.evaluate(host => {
    const link = host.shadowRoot.querySelector("link[data-eoc-persistent-styles]");
    const fresh = document.createElement("extended-openai-management-panel");
    const eager = !!fresh.shadowRoot.querySelector("link[data-eoc-persistent-styles]")
      && !fresh.shadowRoot.querySelector("[data-eoc-persistent-shell]");
    host._renderShell();
    return {eager, sameLink:link === host.shadowRoot.querySelector("link[data-eoc-persistent-styles]"),
      links:host.shadowRoot.querySelectorAll("link[data-eoc-persistent-styles]").length};
  });
  expect(stable).toEqual({eager:true, sameLink:true, links:1});
  expect(requested).toHaveLength(1);
});

test("large route styles arrive with their lazy feature and keep mobile layouts", async ({page}) => {
  await page.setViewportSize({width:600, height:850});
  await page.goto(fixtureUrl("overview", "&bundle=1"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".dashboard-grid")).toBeVisible();
  const coldRadius = await panel.evaluate(host => {
    const sample = document.createElement("div");
    sample.className = "function-group-card";
    host.shadowRoot.append(sample);
    const radius = getComputedStyle(sample).borderTopLeftRadius;
    sample.remove();
    return radius;
  });
  expect(coldRadius).toBe("0px");

  await panel.evaluate(host => host._navigate("capabilities", "functions"));
  await expect(panel.locator(".function-group-card").first()).toBeVisible();
  const functions = await panel.evaluate(host => ({
    radius:getComputedStyle(host.shadowRoot.querySelector(".function-group-card")).borderTopLeftRadius,
    heading:getComputedStyle(host.shadowRoot.querySelector(".function-group-heading")).display,
  }));
  expect(functions).toEqual({radius:"12px", heading:"grid"});

  await panel.evaluate(host => host._navigate("capabilities", "request-rules"));
  await expect(panel.locator(".request-rule-card").first()).toBeVisible();
  const rules = await panel.evaluate(host => ({
    radius:getComputedStyle(host.shadowRoot.querySelector(".request-rule-card")).borderTopLeftRadius,
    actions:getComputedStyle(host.shadowRoot.querySelector(".request-rule-card>.actions")).display,
  }));
  expect(rules).toEqual({radius:"14px", actions:"grid"});

  await panel.evaluate(host => host._navigate("guide"));
  await expect(panel.locator(".guide-quick-card").first()).toBeVisible();
  expect(await panel.evaluate(host => getComputedStyle(
    host.shadowRoot.querySelector(".guide-quick-tasks"),
  ).display)).toBe("grid");

  await panel.evaluate(host => host._navigate("overview"));
  await expect(panel.locator(".broadcast-toggle-row")).toBeVisible();
  expect(await panel.evaluate(host => getComputedStyle(
    host.shadowRoot.querySelector(".broadcast-toggle-row"),
  ).display)).toBe("flex");
});

test("Knowledge editor retains drafts on rerender and releases ownership on navigation", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("data-memory/knowledge"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator("#add-source")).toBeVisible();
  await expect(panel.locator("#knowledge-dialog")).toHaveCount(0);
  await panel.locator("#add-source").click();
  await panel.locator("#knowledge-title").fill("Retained draft");
  await panel.locator("#knowledge-content").fill("An editor must survive unrelated renders.");
  const identity = await panel.evaluate((host) => {
    const root = host.shadowRoot;
    const button = root.querySelector("#add-source");
    const dialog = root.querySelector("#knowledge-dialog");
    const main = root.querySelector("main");
    host._render();
    host._render();
    return {button: button === root.querySelector("#add-source"), dialog: dialog === root.querySelector("#knowledge-dialog"), main: main === root.querySelector("main")};
  });
  expect(identity).toEqual({button:true, dialog:true, main:true});
  await expect(panel.locator("#knowledge-dialog")).toHaveJSProperty("open", true);
  await expect(panel.locator("#knowledge-title")).toHaveValue("Retained draft");
  await panel.locator("#knowledge-save").click();
  await expect(panel.locator("#knowledge-dialog")).toHaveJSProperty("open", false);
  expect(await page.evaluate(() => browserHarness.calls.filter(c => c.section === "knowledge" && c.action === "create").length)).toBe(1);
  expect(await panel.evaluate(async host => {
    const dialog = host.shadowRoot.querySelector("#knowledge-dialog");
    await host._navigate("overview");
    const removed = !host.shadowRoot.querySelector("#knowledge-dialog");
    await host._navigate("data-memory", "knowledge");
    return removed && !dialog.isConnected && !host.shadowRoot.querySelector("#knowledge-dialog");
  })).toBe(true);
  await panel.locator("#add-source").click();
  await expect(panel.locator("#knowledge-title")).toHaveValue("");
  await expectHarnessClean(page, errors);
});

test("Knowledge editor does not carry form state to another agent", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("data-memory/knowledge"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator("#add-source")).toBeVisible();
  await panel.locator("#add-source").click();
  await panel.locator("#knowledge-title").fill("First agent draft");
  await panel.locator("#knowledge-dialog .close-editor.icon").click();
  await panel.locator("#confirm-accept").click();
  const previous = await panel.evaluate(host => {
    const dialog = host.shadowRoot.querySelector("#knowledge-dialog");
    host._data.agents.push({...host._selectedAgent(), subentry_id:"second-agent", title:"Second agent"});
    host._agentId = "second-agent";
    host._render();
    return {removed:!dialog.isConnected, absent:!host.shadowRoot.querySelector("#knowledge-dialog")};
  });
  expect(previous).toEqual({removed:true, absent:true});
  await panel.locator("#add-source").click();
  await expect(panel.locator("#knowledge-title")).toHaveValue("");
  await expectHarnessClean(page, errors);
});

test("scope catalogs are route-specific with independent TTL and mutation invalidation", async ({page}) => {
  await page.goto(fixtureUrl("data-memory/memories"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator("#add-memory")).toBeVisible();
  const counts = await panel.evaluate(async host => {
    const calls = window.browserHarness.calls;
    const count = () => calls.filter(c => c.section === "scopes").length;
    const memoryKey = host._scopeCatalogKey();
    const fetchedAt = host._eocScopeCatalogTimes.get(memoryKey);
    await host._navigate("data-memory", "conversations");
    const archiveSeparate = count();
    const memoryTimestampUnchanged = host._eocScopeCatalogTimes.get(memoryKey) === fetchedAt;
    host._eocScopeCatalogTimes.set(memoryKey, Date.now() - 31_000);
    await host._navigate("data-memory", "memories");
    const expired = count();
    await host._call("memories", "add", {scope_id:host._scopeId, content:"Cache invalidation test", category:"general"});
    await host._loadSection();
    return {archiveSeparate, memoryTimestampUnchanged, expired, mutated:count()};
  });
  expect(counts).toEqual({archiveSeparate:2, memoryTimestampUnchanged:true, expired:3, mutated:4});
});

test("configuration live metadata is fetched only by routes that use it", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("assistant/basics"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator('[data-config="__title"]')).toBeVisible();

  const calls = () => page.evaluate(() =>
    browserHarness.calls
      .filter((call) => call.section === "configuration" && call.action === "live_metadata")
      .map((call) => call.metadata_keys)
  );

  expect(await calls()).toEqual([]);

  await panel.evaluate(host => host._navigate("capabilities", "home-assistant"));
  await expect(panel.locator('[data-config="local_intents_enabled"]')).toBeVisible();
  await expect.poll(calls).toEqual([["local_handling"]]);

  await panel.evaluate(host => host._navigate("assistant", "model-responses"));
  await expect(panel.locator("#reset-model-parameters")).toBeVisible();
  await expect.poll(calls).toEqual([["local_handling"]]);

  await panel.evaluate(host => host._navigate("assistant", "prompt-context"));
  await expect(panel.locator("#prompt-editor")).toBeVisible();
  await expect.poll(calls).toEqual([
    ["local_handling"],
    ["exposed_attribute_catalog"],
  ]);

  await panel.evaluate(host => host._navigate("capabilities", "home-assistant"));
  await expect(panel.locator('[data-config="local_intents_enabled"]')).toBeVisible();
  await expect.poll(calls).toEqual([
    ["local_handling"],
    ["exposed_attribute_catalog"],
  ]);
  await expectHarnessClean(page, errors);
});

test("Conversation History paints the selected scope before secondary data settles", async ({page}) => {
  await page.goto(fixtureUrl("overview"));
  await expect(page.locator("extended-openai-management-panel .dashboard-grid")).toBeVisible();
  const result = await page.evaluate(async () => {
    const host = browserHarness.panel;
    const original = host._hass.callWS;
    const releases = {};
    const started = new Set();
    const observedScopeKinds = [];
    host._hass.callWS = async message => {
      const key = message.section === "scopes"
        ? "scopes"
        : message.section === "configuration" && message.action === "get"
          ? "config"
          : message.section === "conversations" && message.action === "active"
            ? "active"
            : null;
      if (key) {
        started.add(key);
        if (key === "scopes") observedScopeKinds.push(message.scope_kind);
        await new Promise(resolve => { releases[key] = resolve; });
      }
      return original(message);
    };

    const pending = host._navigate("data-memory", "conversations");
    while (!["scopes", "config", "active"].every(key => started.has(key))) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    for (let turn = 0; turn < 40 && (!host.shadowRoot.querySelector("#archive-query") || host._busy); turn++) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    const primaryVisible = Boolean(host.shadowRoot.querySelector("#archive-query")) && host._busy === false;
    const searchInput = host.shadowRoot.querySelector("#archive-query");
    const loadingSettings = host.shadowRoot.textContent.includes("Loading archive settings");
    const scopeKinds = observedScopeKinds;

    releases.scopes();
    releases.config();
    releases.active();
    await pending;
    for (let turn = 0; turn < 20 && host._contentData?.loading?.active; turn++) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    host._hass.callWS = original;
    return {
      primaryVisible,
      loadingSettings,
      scopeKinds,
      activeLoading: host._contentData?.loading?.active,
      configReady: Boolean(host._configData),
      searchPreserved: host.shadowRoot.querySelector("#archive-query") === searchInput,
    };
  });

  expect(result.primaryVisible).toBe(true);
  expect(result.loadingSettings).toBe(true);
  expect(result.scopeKinds.at(-1)).toBe("archive");
  expect(result.activeLoading).toBe(false);
  expect(result.configReady).toBe(true);
  expect(result.searchPreserved).toBe(true);
});

test("conversation configuration starts before a pending scope catalog finishes", async ({page}) => {
  await page.goto(fixtureUrl("overview"));
  await expect(page.locator("extended-openai-management-panel .dashboard-grid")).toBeVisible();
  const result = await page.evaluate(async () => {
    const host = browserHarness.panel;
    const original = host._hass.callWS;
    let releaseScope;
    let configStarted = false;
    let markScopeStarted;
    const scopeStarted = new Promise(resolve => { markScopeStarted = resolve; });
    host._hass.callWS = async message => {
      if (message.section === "scopes") {
        await new Promise(resolve => { releaseScope = resolve; markScopeStarted(); });
      }
      if (message.section === "configuration") configStarted = true;
      return original(message);
    };
    const pending = host._navigate("data-memory", "conversations");
    await scopeStarted;
    const concurrent = configStarted;
    releaseScope();
    await pending;
    host._hass.callWS = original;
    return {concurrent, sessions:!!host._contentData?.sessions, config:!!host._configData};
  });
  expect(result).toEqual({concurrent:true, sessions:true, config:true});
});

test("conversation collection starts before a pending scope catalog finishes", async ({page}) => {
  await page.goto(fixtureUrl("overview"));
  await expect(page.locator("extended-openai-management-panel .dashboard-grid")).toBeVisible();
  const result = await page.evaluate(async () => {
    const host = browserHarness.panel;
    const original = host._hass.callWS;
    const initialScope = host._scopeId;
    let releaseScope;
    let listStarted = false;
    let markScopeStarted;
    const scopeStarted = new Promise(resolve => { markScopeStarted = resolve; });
    const listScopes = [];
    host._hass.callWS = async message => {
      if (message.section === "scopes") {
        await new Promise(resolve => { releaseScope = resolve; markScopeStarted(); });
      }
      if (message.section === "conversations" && message.action === "list") {
        listStarted = true;
        listScopes.push(message.scope_id);
      }
      return original(message);
    };
    const pending = host._navigate("data-memory", "conversations");
    await scopeStarted;
    const concurrent = listStarted;
    releaseScope();
    await pending;
    host._hass.callWS = original;
    return {concurrent, initialScope, selectedScope:host._scopeId, listScopes};
  });
  expect(result.concurrent).toBe(true);
  expect(result.selectedScope).toBe(result.initialScope);
  expect(result.listScopes).toEqual([result.initialScope]);
});

test("invalidated speculative Memory scope is discarded and refetched once", async ({page}) => {
  await page.goto(fixtureUrl("overview"));
  await expect(page.locator("extended-openai-management-panel .dashboard-grid")).toBeVisible();
  const result = await page.evaluate(async () => {
    const host = browserHarness.panel;
    const original = host._hass.callWS;
    const initialScope = host._scopeId;
    const replacementScope = "user:replacement";
    let releaseScope;
    let markScopeStarted;
    const scopeStarted = new Promise(resolve => { markScopeStarted = resolve; });
    const listScopes = [];
    host._hass.callWS = async message => {
      if (message.section === "scopes") {
        await new Promise(resolve => { releaseScope = resolve; markScopeStarted(); });
        return {scopes:[{
          scope_id:replacementScope,
          scope_type:"user",
          display_name:"Replacement user",
          is_current_user:true,
          memory_count:0,
          conversation_count:0,
        }]};
      }
      if (message.section === "memories" && message.action === "list") {
        listScopes.push(message.scope_id);
        if (message.scope_id === initialScope) throw new Error("stale scope");
        return {memories:[], marker:message.scope_id};
      }
      return original(message);
    };
    const pending = host._navigate("data-memory", "memories");
    await scopeStarted;
    const speculativeStarted = listScopes.includes(initialScope);
    releaseScope();
    await pending;
    host._hass.callWS = original;
    return {
      speculativeStarted,
      initialScope,
      selectedScope:host._scopeId,
      listScopes,
      marker:host._result?.marker,
      error:host._error,
    };
  });
  expect(result).toMatchObject({
    speculativeStarted:true,
    selectedScope:"user:replacement",
    marker:"user:replacement",
    error:null,
  });
  expect(result.listScopes).toEqual([result.initialScope, "user:replacement"]);
});

test("late conversation configuration cannot replace a newer route result", async ({page}) => {
  await page.goto(fixtureUrl("overview"));
  await expect(page.locator("extended-openai-management-panel .dashboard-grid")).toBeVisible();
  const result = await page.evaluate(async () => {
    const host = browserHarness.panel;
    const original = host._hass.callWS;
    let release;
    let markConfigStarted;
    const configStarted = new Promise(resolve => { markConfigStarted = resolve; });
    host._hass.callWS = async message => {
      if (message.section === "configuration") {
        await new Promise(resolve => { release = resolve; markConfigStarted(); });
      }
      return original(message);
    };
    const pending = host._navigate("data-memory", "conversations");
    await configStarted;
    await host._navigate("data-memory", "knowledge");
    const current = host._result;
    release();
    await pending;
    host._hass.callWS = original;
    return {same:host._result === current, view:host._viewKey(), draft:host._draft};
  });
  expect(result).toEqual({same:true, view:"data-memory/knowledge", draft:null});
});

test("voice and memory settings implementations load only when their routes are visited", async ({page}) => {
  const errors = trackPageErrors(page);
  const loaded = [];
  page.on("request", request => loaded.push(request.url()));
  await page.goto(fixtureUrl("overview"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".dashboard-grid")).toBeVisible();
  expect(loaded.some(url => url.endsWith("/voice-identity-ui.js"))).toBe(false);
  expect(loaded.some(url => url.endsWith("/memory-settings-ui.js"))).toBe(false);
  const featureLoaded = view => page.evaluate(async name => Boolean(
    (await import("/custom_components/extended_openai_conversation_responses/frontend/management-route.js")).getRouteFeature(name)
  ), view);
  expect(await featureLoaded("assistant/voice")).toBe(false);
  expect(await featureLoaded("data-memory/memory-settings")).toBe(false);
  await panel.evaluate(host => host._navigate("assistant", "voice"));
  expect(loaded.some(url => url.endsWith("/voice-identity-core.js"))).toBe(true);
  expect(loaded.some(url => url.endsWith("/voice-identity-ui.js"))).toBe(false);
  await expect(panel.locator(".voice-identity-flow")).toBeHidden();
  await expect(panel.locator(".voice-identity-help > summary")).toBeVisible();
  await expect(panel.locator("#voice-current-summary")).toBeVisible();
  expect(await featureLoaded("assistant/voice")).toBe(true);
  await panel.locator('[data-config="voice_scope_policy"]').selectOption("device_mapping");
  await expect(panel.locator("#voice-mappings")).toBeVisible();
  expect(loaded.some(url => url.endsWith("/voice-identity-ui.js"))).toBe(true);
  await panel.evaluate(host => host._navigate("data-memory", "memory-settings"));
  await expect(panel.locator("[data-memory-config]").first()).toBeVisible();
  expect(await featureLoaded("data-memory/memory-settings")).toBe(true);
  await expectHarnessClean(page, errors);
});

test("expired Knowledge list renders immediately and unchanged refresh preserves its DOM", async ({page}) => {
  await page.goto(fixtureUrl("data-memory/knowledge"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator("#add-source")).toBeVisible();
  const result = await panel.evaluate(async host => {
    const key = host._sectionCacheKey();
    await host._navigate("overview");
    host._eocSectionCacheTimes.set(key, Date.now() - 31_000);
    const original = host._hass.callWS;
    let release;
    let markListStarted;
    const listStarted = new Promise(resolve => { markListStarted = resolve; });
    host._hass.callWS = async message => {
      if (message.section === "knowledge" && message.action === "list") {
        await new Promise(resolve => { release = resolve; markListStarted(); });
      }
      return original(message);
    };
    const pending = host._navigate("data-memory", "knowledge");
    await listStarted;
    const button = host.shadowRoot.querySelector("#add-source");
    const immediate = !!button && !host._busy;
    release();
    await pending;
    host._hass.callWS = original;
    return {immediate, preserved:button === host.shadowRoot.querySelector("#add-source"), refreshed:host._eocSectionCacheTimes.get(key) > Date.now() - 1000};
  });
  expect(result).toEqual({immediate:true, preserved:true, refreshed:true});
});

test("overview uses the native registry and defers unrelated feature code", async ({page}) => {
  const errors = trackPageErrors(page);
  const modules = new Set();
  page.on("request", request => modules.add(new URL(request.url()).pathname.split("/").pop()));
  await page.addInitScript(() => {
    for (const name of ["define", "get", "whenDefined"]) {
      Object.defineProperty(customElements, name, {value:customElements[name], writable:false, configurable:false});
    }
  });
  await page.goto(fixtureUrl("overview") + "&predefine=1");
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".dashboard-grid")).toBeVisible();
  for (const name of ["agent-config-editor.js", "backup-transfer-ui.js", "exposed-attributes-ui.js", "management-provider-credentials.js", "usage-chart.js", "usage-input-footprint.js",
    "management-history-pagination.js", "quiet-hours-ui.js", "debug-management.js", "debug-panel.js", "management-function-repair.js"]) {
    expect(modules.has(name), name).toBe(false);
  }
  await panel.evaluate(host => host._navigate("assistant", "prompt-context"));
  await expect(panel.locator("#prompt-preview-dialog")).toHaveCount(1);
  expect(modules.has("agent-config-editor.js")).toBe(true);
  await expect(panel.locator("#tool-dialog, #group-dialog, #restore-dialog")).toHaveCount(0);
  await panel.evaluate(host => host._navigate("usage-maintenance", "diagnostics"));
  expect(modules.has("management-provider-credentials.js")).toBe(true);
  await expect(panel.locator("#test-agent")).toBeVisible();
  await panel.evaluate(host => host._navigate("overview"));
  await expect(panel.locator("#prompt-preview-dialog, #tool-dialog, #restore-dialog, #rule-dialog")).toHaveCount(0);
  await expectHarnessClean(page, errors);
});

test("Usage becomes usable before recent runs and retention settle", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("overview"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".dashboard-grid")).toBeVisible();

  await panel.evaluate(host => {
    const original = host._hass.callWS;
    window.progressiveUsage = {calls:[], releaseRuns:null, releaseRetention:null};
    host._hass.callWS = async message => {
      if (message.section !== "usage") return original(message);
      window.progressiveUsage.calls.push(message.action);
      if (message.action === "summary") {
        return {today:{date:"2026-09-22", total_tokens:30}, lifetime:{total_tokens:300}, latest:{total_tokens:12}};
      }
      if (message.action === "daily") {
        return {days:[{date:"2026-09-22", total_tokens:30, input_tokens:20, output_tokens:10, cached_input_tokens:5, api_request_count:1, run_count:1}]};
      }
      if (message.action === "runs") {
        return new Promise(resolve => {
          window.progressiveUsage.releaseRuns = () => resolve({runs:[{
            run_id:"run-progressive",
            completed_at:"2026-09-22T12:00:00+00:00",
            total_tokens:30,
            cached_input_tokens:5,
            request_count:1,
            duration_ms:250,
            successful:true,
          }]});
        });
      }
      if (message.action === "retention") {
        return new Promise(resolve => {
          window.progressiveUsage.releaseRetention = () => resolve({detail_retention_days:30});
        });
      }
      return original(message);
    };
  });

  await panel.evaluate(host => host._navigate("usage-maintenance", "usage"));
  await expect(panel.locator("#usage-window")).toBeVisible();
  await panel.evaluate(host => { window.usageChartNode = host.shadowRoot.querySelector(".chart"); });
  await expect(panel.getByText("Loading recent runs…")).toBeVisible();
  expect(await panel.evaluate(host => host._busy)).toBe(false);
  expect(await page.evaluate(() => progressiveUsage.calls.sort())).toEqual(["daily", "retention", "runs", "summary"]);
  expect(await panel.evaluate(host => host._result.loading)).toEqual({runs:true, retention:true});

  await page.evaluate(() => progressiveUsage.releaseRuns());
  await expect.poll(() => panel.evaluate(host => host._result.loading)).toEqual({runs:false, retention:true});
  await expect(panel.getByText("Loading recent runs…")).toHaveCount(0);
  await expect(panel.getByText("Success", {exact:true})).toBeVisible();
  expect(await panel.evaluate(host => host.shadowRoot.querySelector(".chart") === window.usageChartNode)).toBe(true);
  expect(await panel.evaluate(host => host._result.retention)).toBeUndefined();

  await page.evaluate(() => progressiveUsage.releaseRetention());
  await expect.poll(() => panel.evaluate(host => host._result.loading)).toEqual({runs:false, retention:false});
  expect(await panel.evaluate(host => host._result.retention.detail_retention_days)).toBe(30);
  expect(await panel.evaluate(host => host.shadowRoot.querySelector(".chart") === window.usageChartNode)).toBe(true);
  await expectHarnessClean(page, errors);
});

test("navigation acknowledges the destination while keeping useful content mounted", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("overview"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".dashboard-grid")).toBeVisible();
  await panel.evaluate(host => {
    const original = host._hass.callWS;
    window.previousMain = host.shadowRoot.querySelector("main").firstElementChild;
    host._hass.callWS = message => {
      if (message.section === "usage" && message.action === "summary") {
        return new Promise(resolve => { window.releaseUsage = () => resolve(original(message)); });
      }
      return original(message);
    };
    window.pendingNavigation = host._navigate("usage-maintenance", "usage");
  });
  await expect.poll(() => panel.evaluate(host => host._viewKey())).toBe("usage-maintenance/usage");
  const pending = await panel.evaluate(host => {
    const root = host.shadowRoot;
    const main = root.querySelector("main");
    return {
      page: root.querySelector(".top-nav button.active")?.dataset.page,
      subsection: root.querySelector(".subsection-nav button.active")?.dataset.subsection,
      mobilePage: root.querySelector("#top-section-mobile")?.value,
      mobileSection: root.querySelector("#local-section")?.value,
      retained: main.firstElementChild === window.previousMain,
      busy: main.getAttribute("aria-busy"),
      inert: main.inert,
    };
  });
  expect(pending).toEqual({page:"usage-maintenance", subsection:"usage", mobilePage:"usage-maintenance", mobileSection:"usage", retained:true, busy:"true", inert:true});
  await page.evaluate(() => releaseUsage());
  await page.evaluate(() => pendingNavigation);
  await expect(panel.locator("#usage-window")).toBeVisible();
  await expectHarnessClean(page, errors);
});

test("secondary Usage failure does not replace the primary page", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("overview"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".dashboard-grid")).toBeVisible();

  await panel.evaluate(host => {
    const original = host._hass.callWS;
    window.progressiveUsageFailure = {releaseRuns:null, releaseRetention:null};
    host._hass.callWS = async message => {
      if (message.section !== "usage") return original(message);
      if (message.action === "summary") {
        return {today:{date:"2026-09-22", total_tokens:30}, lifetime:{total_tokens:300}};
      }
      if (message.action === "daily") {
        return {days:[{date:"2026-09-22", total_tokens:30, input_tokens:20, output_tokens:10}]};
      }
      if (message.action === "runs") {
        return new Promise((_, reject) => {
          window.progressiveUsageFailure.releaseRuns = () => reject(new Error("runs unavailable"));
        });
      }
      if (message.action === "retention") {
        return new Promise(resolve => {
          window.progressiveUsageFailure.releaseRetention = () => resolve({detail_retention_days:30});
        });
      }
      return original(message);
    };
  });

  await panel.evaluate(host => host._navigate("usage-maintenance", "usage"));
  await expect(panel.locator("#usage-window")).toBeVisible();
  await expect(panel.getByText("Loading recent runs…")).toBeVisible();

  await page.evaluate(() => progressiveUsageFailure.releaseRuns());
  await expect(panel.getByText("Recent runs unavailable", {exact:true})).toBeVisible();
  expect(await panel.evaluate(host => host._result.loading)).toEqual({runs:false, retention:true});
  await page.evaluate(() => progressiveUsageFailure.releaseRetention());
  await expect.poll(() => panel.evaluate(host => host._result.loading)).toEqual({runs:false, retention:false});
  await expect(panel.locator("#usage-window")).toBeVisible();
  expect(await panel.evaluate(host => host._error)).toBe(null);
  expect(await panel.evaluate(host => host._result.load_errors)).toEqual([
    {key:"runs", label:"Recent runs", message:"runs unavailable"},
  ]);
  await expectHarnessClean(page, errors);
});

test("Quiet Hours and request debugging initialize on first entry without global loader wrappers", async ({page}) => {
  const errors = trackPageErrors(page);
  await page.goto(fixtureUrl("overview"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".dashboard-grid")).toBeVisible();
  await panel.evaluate(host => {
    const original = host._hass.callWS;
    window.featureCalls = [];
    window.originalDataLoader = host._loadSectionData;
    host._hass.callWS = async message => {
      if (message.section === "quiet_hours") {
        window.featureCalls.push(message);
        return {
          config:message.config || {
            enabled:false,
            start:"22:00",
            end:"07:00",
            max_volume:0.3,
            wake_sound:"off",
            overrides:{
              "assist_satellite.kitchen":{media_player_entity_id:"media_player.legacy"},
            },
          },
          satellites:[{
            satellite_entity_id:"assist_satellite.kitchen",
            name:"Kitchen Voice",
            device_id:"device-kitchen",
            media_player_entity_id:"media_player.legacy",
            wake_sound_entity_id:"switch.kitchen_wake",
            media_player_source:"manual",
            wake_sound_source:"auto",
            media_player_candidates:["media_player.kitchen"],
            wake_sound_candidates:["switch.kitchen_wake"],
          }],
        };
      }
      if (message.type.endsWith("/request_debug")) {
        window.featureCalls.push(message);
        return message.action === "agents" ? {agents:[host._selectedAgent()]} : {runs:[], enabled:false};
      }
      return original(message);
    };
  });
  await panel.evaluate(host => host._navigate("capabilities", "quiet-hours"));
  await expect(panel.locator("#qh-enabled")).toBeVisible();
  await expect(panel.locator(".save-bar")).toHaveCount(0);
  await expect(panel.locator(".qh-grid").first()).toHaveCSS("display", "grid");
  const pickerState = await panel.evaluate(host => {
    const pickers = [...host.shadowRoot.querySelectorAll(".qh-override")];
    return pickers.map(picker => ({
      kind:picker.dataset.kind,
      value:picker.value,
      includeDomains:picker.includeDomains,
      includeEntities:picker.includeEntities,
      allowCustomEntity:picker.allowCustomEntity,
      placeholder:picker.placeholder,
    }));
  });
  expect(pickerState).toEqual([
    {
      kind:"media_player_entity_id",
      value:"media_player.legacy",
      includeDomains:["media_player"],
      includeEntities:["media_player.kitchen", "media_player.legacy"],
      allowCustomEntity:false,
      placeholder:"Automatic",
    },
    {
      kind:"wake_sound_entity_id",
      value:"",
      includeDomains:["switch"],
      includeEntities:["switch.kitchen_wake"],
      allowCustomEntity:false,
      placeholder:"Automatic · switch.kitchen_wake",
    },
  ]);
  await panel.locator("#qh-enabled").check();
  await panel.locator("#save-page").click();
  await expect.poll(() => page.evaluate(() => featureCalls.filter(c => c.section === "quiet_hours" && c.action === "update").length)).toBe(1);
  await panel.evaluate(host => host._navigate("usage-maintenance", "request-debug"));
  await expect(panel.locator("extended-openai-debug-panel[embedded]")).toBeVisible();
  await expect.poll(() => page.evaluate(() => featureCalls.filter(c => c.type.endsWith("/request_debug") && c.action === "runs").length)).toBe(1);
  await panel.evaluate(host => host._navigate("usage-maintenance", "usage"));
  await expect(panel.locator("#usage-window")).toBeVisible();
  await panel.evaluate(host => host._navigate("data-memory", "conversations"));
  expect(await panel.evaluate(host => host._loadSectionData === window.originalDataLoader)).toBe(true);
  await expectHarnessClean(page, errors);
});

test("a delayed first route import cannot start obsolete data work after navigation", async ({page}) => {
  let release;
  const requested = new Promise(resolve => {
    page.route("**/management-history-pagination.js", async route => {
      await new Promise(unblock => { release = unblock; resolve(); });
      await route.continue();
    });
  });
  await page.goto(fixtureUrl("overview"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".dashboard-grid")).toBeVisible();
  await panel.evaluate(host => { window.pendingNavigation = host._navigate("data-memory", "conversations"); });
  await requested;
  await panel.evaluate(host => host._navigate("guide"));
  const before = await page.evaluate(() => browserHarness.calls.length);
  release();
  await page.evaluate(() => window.pendingNavigation);
  expect(await panel.evaluate(host => host._viewKey())).toBe("guide");
  expect(await page.evaluate(() => browserHarness.calls.length)).toBe(before);
  await expect(panel.locator("extended-openai-debug-panel, #archive-query")).toHaveCount(0);
});

test("repairable configuration reads use the normal route before the Function Tools editor is imported", async ({page}) => {
  const modules = new Set();
  page.on("request", request => modules.add(new URL(request.url()).pathname.split("/").pop()));
  await page.goto(fixtureUrl("overview"));
  const panel = page.locator("extended-openai-management-panel");
  await expect(panel.locator(".dashboard-grid")).toBeVisible();
  const calls = await panel.evaluate(async host => {
    host._selectedAgent().configuration_issue = {field:"functions", repairable:true};
    const original = host._hass.callWS;
    const readCalls = [];
    host._hass.callWS = async message => {
      if (message.section === "configuration" || message.section === "function_repair") {
        readCalls.push([message.section, message.action]);
      }
      return original(message);
    };
    await host._navigate("assistant", "basics");
    return readCalls;
  });
  expect(calls).toEqual([["configuration", "get"]]);
  expect(modules.has("management-function-repair.js")).toBe(false);
  await expect(panel.locator('[data-config="__title"]')).toBeVisible();
});
