import assert from "node:assert/strict";
import {bindBroadcast} from "../custom_components/extended_openai_conversation_responses/frontend/overview-broadcast.js";

globalThis.window = {
  location: {pathname: "/extended-openai/assistant/basics"},
  addEventListener() {},
  removeEventListener() {},
};
globalThis.history = {pushState() {}};
globalThis.localStorage = {
  values: new Map(),
  getItem(key) { return this.values.get(key) ?? null; },
  setItem(key, value) { this.values.set(key, value); },
};
globalThis.HTMLElement = class {
  attachShadow() { this.shadowRoot = {hasChildNodes: () => false, querySelector: () => null}; }
};
let definedPanel;
let resolveDefined;
const definedPromise = new Promise((resolve) => { resolveDefined = resolve; });
globalThis.customElements = {
  define(_name, constructor) {
    definedPanel = constructor;
    resolveDefined();
  },
  get() { return definedPanel; },
  whenDefined() { return definedPanel ? Promise.resolve() : definedPromise; },
};

const [{ExtendedOpenAIManagementPanel}, {bindRequestRules}] = await Promise.all([
  import("../custom_components/extended_openai_conversation_responses/frontend/management-panel.js"),
  import("../custom_components/extended_openai_conversation_responses/frontend/request-rules-ui.js"),
]);

// These cases isolate cached data behavior after route assets are ready.
const {
  AGENT_KEY,
  ENTRY_KEY,
  applyRequestRuleSearch,
  requestRuleSearchText,
  routeAssetPromise,
  routeFeaturesReady,
  prefetchIntentRead,
  consumeIntentRead,
  startStoredConfigurationPrefetch,
  consumeStoredConfigurationPrefetch,
} = await import("../custom_components/extended_openai_conversation_responses/frontend/management-route.js");
await routeAssetPromise("assistant/basics");

const agents = [
  {entry_id:"entry-a", subentry_id:"agent-a", title:"A"},
  {entry_id:"entry-b", subentry_id:"agent-b", title:"B"},
];
const initialScopes = [{scope_id:"user:current", scope_type:"user", display_name:"Current", is_current_user:true}];

{
  const panel = panelFor("assistant", "basics");
  let resolveRead;
  let reads = 0;
  panel._call = () => { reads++; return new Promise((resolve) => { resolveRead = resolve; }); };
  const prefetched = prefetchIntentRead(panel, "overview");
  const consumed = consumeIntentRead(panel, "overview", "overview", "summary");
  assert.equal(reads, 1, "navigation consumes its in-flight read");
  resolveRead({usage:{today:{total_tokens:7}}});
  assert.deepEqual(await consumed, await prefetched);
  panel._agentId = "agent-b";
  const next = consumeIntentRead(panel, "overview", "overview", "summary");
  assert.equal(reads, 2, "a different agent cannot consume the previous read");
  resolveRead({usage:{today:{total_tokens:9}}});
  await next;
  const overviewKey = panel._sectionCacheKey("overview");
  panel._sectionCache.set(overviewKey, {usage:{today:{total_tokens:9}}});
  panel._eocSectionCacheTimes.set(overviewKey, Date.now());
  assert.equal(prefetchIntentRead(panel, "overview"), null,
    "a fresh Overview summary does not trigger speculative backend work");
  assert.equal(reads, 2);
  assert.equal(prefetchIntentRead(panel, "usage-maintenance/usage"), null);
}

{
  const panel = panelFor("overview", null);
  let resolveRetention;
  const calls = [];
  panel._call = (section, action) => {
    calls.push([section, action]);
    return new Promise((resolve) => { resolveRetention = resolve; });
  };
  const prefetched = prefetchIntentRead(panel, "usage-maintenance/retention");
  const consumed = consumeIntentRead(
    panel,
    "usage-maintenance/retention",
    "configuration",
    "retention_get",
  );
  assert.deepEqual(calls, [["configuration", "retention_get"]]);
  resolveRetention({
    title: "A",
    revision: "r1",
    projection: "retention",
    config: {usage_request_retention_days: 30, usage_run_retention_days: 90},
    options: {},
  });
  assert.deepEqual(await consumed, await prefetched);

  panel._cleanConfigSnapshots.set(
    panel._configurationSnapshotKey("agent-a", "retention"),
    {result: {projection:"retention", revision:"r1", config:{}}, loadedAt: Date.now(), generation:panel._cacheGeneration},
  );
  assert.equal(prefetchIntentRead(panel, "usage-maintenance/retention"), null,
    "a fresh retention projection does not trigger speculative backend work");
  assert.equal(calls.length, 1);
}

function panelFor(page = "assistant", subsection = "basics") {
  const panel = new ExtendedOpenAIManagementPanel();
  panel._page = page;
  panel._subsection = subsection;
  panel._data = {agents, scopes:initialScopes, is_admin:true};
  panel._agentId = "agent-a";
  panel._scopeId = "user:current";
  panel.renderStates = [];
  panel._render = () => panel.renderStates.push(panel._busy);
  return panel;
}

{
  const panel = panelFor("capabilities", "functions");
  panel._configData = {revision:"r1", config:{functions:[], function_groups:[]}};
  const calls = [];
  const releases = [];
  panel._hass = {callWS: (message) => {
    calls.push(message);
    return new Promise((resolve) => releases.push(resolve));
  }};

  const first = panel._call("tools", "delete", {name:"tool_a", confirm:true});
  const second = panel._call("tools", "delete", {name:"tool_b", confirm:true});
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(calls.length, 1, "the second Function mutation waits for the first response");

  releases.shift()({revision:"r2", functions:[], function_groups:[], _performance:{handler_ms:1}});
  await first;
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(calls.length, 2, "the mutation tail releases as soon as the first request settles");
  assert.equal(calls[1].revision, "r2", "a queued request uses the revision returned by the preceding mutation");

  releases.shift()({revision:"r3", functions:[], function_groups:[], _performance:{handler_ms:2}});
  await second;
  assert.equal(panel._eocFunctionMutationTail, null);
  assert.equal(panel._eocFunctionMutationDiagnostics.length, 2);
  assert.deepEqual(panel._eocFunctionMutationDiagnostics.map((item) => item.status), ["fulfilled", "fulfilled"]);
  assert.equal(panel._eocFunctionMutationDiagnostics[0].backend.handler_ms, 1);
  assert.ok(panel._eocFunctionMutationDiagnostics[1].queueWaitMs >= 0);
  assert.equal(panel._eocRequestDiagnostics.filter((item) => item.section === "tools").length, 2);
}

const configResult = (projection, title = "A") => ({
  projection, title, revision:`${title}-r1`, config:{usage_request_retention_days:30, chat_model:"gpt-4o"},
});

{
  const panel = panelFor();
  const baseline = configResult("full");
  panel._configData = baseline;
  panel._configDataStale = true;
  panel._draft = structuredClone(baseline.config);
  panel._draftTitle = baseline.title;
  panel._draftAgentId = panel._agentId;
  let finishRead;
  panel._call = () => new Promise((resolve) => { finishRead = resolve; });
  const loading = panel._loadConfigDraft();
  panel._draftTitle = "Edited while configuration refreshed";
  panel._setConfigDirty(true);
  finishRead(configResult("full", "Remote update"));
  await loading;
  assert.equal(panel._draftTitle, "Edited while configuration refreshed");
  assert.equal(panel._configData, baseline, "an in-flight read must not replace a newly dirty baseline");
  assert.equal(panel._configDirty, true);
}

{
  const panel = panelFor("assistant", "basics");
  localStorage.setItem(AGENT_KEY, "agent-a");
  localStorage.setItem(ENTRY_KEY, "entry-a");
  const calls = [];
  panel._hass = {callWS: async (message) => { calls.push(message); return configResult("full"); }};
  panel._call = () => { throw new Error("valid prefetch must suppress a second configuration/get"); };
  startStoredConfigurationPrefetch(panel, "agent-a");
  await panel._loadConfigDraft();
  assert.equal(calls.length, 1);
  assert.deepEqual([calls[0].section, calls[0].action], ["configuration", "get"]);
  assert.equal(panel._eocConfigurationReadDiagnostics.prefetch.status, "consumed");
  assert.deepEqual(
    [panel._eocConfigurationReadDiagnostics.prefetch.entryId,
      panel._eocConfigurationReadDiagnostics.prefetch.subentryId,
      panel._eocConfigurationReadDiagnostics.prefetch.view],
    ["entry-a", "agent-a", "assistant/basics"],
  );
  assert.equal(typeof panel._eocConfigurationReadDiagnostics.prefetch.startedAt, "number");
  assert.equal(panel._eocConfigurationReadDiagnostics.draft.source, "prefetched-request");
  assert.ok(performance.getEntriesByType("mark").some((entry) =>
    entry.name.includes("config-read:prefetch-started")
      && entry.detail?.action === "get" && entry.detail?.status === "started"));
  assert.ok(performance.getEntriesByType("mark").some((entry) =>
    entry.name.includes("config-read:draft-source")
      && entry.detail?.source === "prefetched-request"));
  assert.ok(performance.getEntriesByType("measure").some((entry) =>
    entry.name.includes("config-read:prefetch-response")
      && entry.detail?.status === "fulfilled" && entry.detail?.action === "get"));
  assert.ok(performance.getEntriesByType("mark")
    .filter((entry) => entry.name.includes("config-read:"))
    .every((entry) => Object.keys(entry.detail || {}).every((key) =>
      ["view", "action", "source", "status", "reason"].includes(key))));
  await panel._loadConfigDraft();
  assert.equal(panel._eocConfigurationReadDiagnostics.draft.source, "active-config");
  assert.equal(calls.length, 1);
}

{
  const panel = panelFor("assistant", "basics");
  panel._rememberCleanConfiguration(configResult("full"));
  panel._hass = {callWS: () => { throw new Error("clean snapshot suppresses prefetch"); }};
  panel._call = () => { throw new Error("clean snapshot suppresses backend read"); };
  assert.equal(startStoredConfigurationPrefetch(panel, "agent-a"), null);
  await panel._loadConfigDraft();
  assert.equal(panel._eocConfigurationReadDiagnostics.prefetch.reason, "clean-snapshot");
  assert.equal(panel._eocConfigurationReadDiagnostics.draft.source, "clean-snapshot");
}

{
  const panel = panelFor("assistant", "basics");
  const calls = [];
  panel._hass = {callWS: async (message) => { calls.push(message); throw new Error("prefetch failed"); }};
  panel._call = async (section, action) => {
    calls.push({section, action});
    return configResult("full", "Fallback");
  };
  startStoredConfigurationPrefetch(panel, "agent-a");
  await panel._loadConfigDraft();
  assert.deepEqual(calls.map((item) => item.action), ["get", "get"]);
  assert.equal(panel._draftTitle, "Fallback");
  assert.equal(panel._eocConfigurationReadDiagnostics.prefetch.reason, "request-failed");
  assert.equal(panel._eocConfigurationReadDiagnostics.draft.source, "new-backend-request");
  assert.ok(performance.getEntriesByType("measure").some((entry) =>
    entry.name.includes("config-read:fallback-response")
      && entry.detail?.source === "new-backend-request" && entry.detail?.status === "fulfilled"));
}

{
  const panel = panelFor("assistant", "basics");
  localStorage.values.clear();
  assert.equal(startStoredConfigurationPrefetch(panel), null);
  assert.equal(panel._eocConfigurationReadDiagnostics.prefetch.reason, "missing-stored-agent");
  localStorage.setItem(AGENT_KEY, "agent-a");
  assert.equal(startStoredConfigurationPrefetch(panel), null);
  assert.equal(panel._eocConfigurationReadDiagnostics.prefetch.reason, "missing-stored-entry");
  localStorage.setItem(ENTRY_KEY, "entry-a");
}

{
  const panel = panelFor("assistant", "basics");
  let warnings = 0;
  const originalWarn = console.warn;
  const originalError = console.error;
  console.warn = () => { warnings++; };
  console.error = () => { warnings++; };
  try {
    for (let index = 0; index < 110; index++) {
      localStorage.values.clear();
      startStoredConfigurationPrefetch(panel);
    }
  } finally {
    console.warn = originalWarn;
    console.error = originalError;
  }
  assert.equal(warnings, 0);
  assert.ok(performance.getEntriesByType("mark")
    .filter((entry) => entry.name.includes("config-read:")).length <= 100);
  localStorage.setItem(AGENT_KEY, "agent-a");
  localStorage.setItem(ENTRY_KEY, "entry-a");
}

{
  const panel = panelFor("assistant", "basics");
  panel._hass = {callWS: async () => configResult("full")};
  startStoredConfigurationPrefetch(panel, "agent-a");
  panel._cacheGeneration++;
  assert.equal(consumeStoredConfigurationPrefetch(panel, "get"), null);
  assert.equal(panel._eocConfigurationReadDiagnostics.prefetch.reason, "cache-generation-changed");
}

{
  const panel = panelFor("usage-maintenance", "usage");
  panel._rememberCleanConfiguration(configResult("retention"));
  panel._hass = {callWS: async () => { throw new Error("fresh retention must not fetch"); }};
  await panel._navigate("usage-maintenance", "retention");
  assert.equal(panel._draftAgentId, "agent-a");
  assert.equal(panel._draft.usage_request_retention_days, 30);
  assert.equal(panel._result, panel._configData);
  assert.ok(panel.renderStates.length > 0);
  assert.ok(panel.renderStates.every((busy) => !busy), "fresh Retention revisit never renders busy");
  panel._clearConfigDraft();
  panel._page = "usage-maintenance";
  panel._subsection = "usage";
  panel.renderStates = [];
  await panel._navigate("usage-maintenance", "retention");
  assert.ok(panel.renderStates.every((busy) => !busy), "repeated clean visit stays immediate");
}

{
  const panel = panelFor("usage-maintenance", "usage");
  panel._rememberCleanConfiguration(configResult("retention"));
  const key = panel._configurationSnapshotKey("agent-a", "retention");
  panel._cleanConfigSnapshots.get(key).loadedAt -= 31_000;
  let reads = 0;
  panel._hass = {callWS: async () => { reads++; return configResult("retention", "Fresh"); }};
  await panel._navigate("usage-maintenance", "retention");
  assert.equal(reads, 1, "expired Retention projection reloads");
  assert.ok(panel.renderStates.includes(true), "expired projection takes the loading path");
  assert.equal(panel._draftTitle, "Fresh");
}

{
  const panel = panelFor("usage-maintenance", "usage");
  panel._rememberCleanConfiguration(configResult("retention"));
  panel._invalidateAfterMutation("agent-a", "tools", "save");
  let reads = 0;
  panel._hass = {callWS: async () => { reads++; return configResult("retention", "After edit"); }};
  await panel._navigate("usage-maintenance", "retention");
  assert.equal(reads, 1, "mutation invalidation prevents instant stale hydration");
  assert.ok(panel.renderStates.includes(true));
  assert.equal(panel._draftTitle, "After edit");
}

{
  const panel = panelFor("usage-maintenance", "usage");
  panel._rememberCleanConfiguration(configResult("retention"));
  panel._agentId = "agent-b";
  let reads = 0;
  panel._hass = {callWS: async () => { reads++; return configResult("retention", "B"); }};
  await panel._navigate("usage-maintenance", "retention");
  assert.equal(reads, 1, "another agent cannot use the first agent's snapshot");
  assert.equal(panel._draftTitle, "B");
}

{
  const panel = panelFor("usage-maintenance", "usage");
  panel._rememberCleanConfiguration(configResult("full"));
  panel._hass = {callWS: async () => { throw new Error("fresh Assistant must not fetch"); }};
  await panel._navigate("assistant", "basics");
  assert.equal(panel._draftAgentId, "agent-a");
  assert.ok(panel.renderStates.every((busy) => !busy), "clean Assistant reuse never renders busy");
  panel._draft.chat_model = "unsaved";
  panel._setConfigDirty(true);
  panel._confirm = async () => false;
  await panel._navigate("overview");
  assert.equal(panel._viewKey(), "overview", "internal navigation keeps the assistant draft active");
  assert.equal(panel._draft.chat_model, "unsaved");
  await panel._navigate("assistant", "basics");
  assert.equal(panel._draft.chat_model, "unsaved", "returning does not replace the active draft");
}

{
  const panel = panelFor("usage-maintenance", "retention");
  panel._rememberCleanConfiguration(configResult("retention"));
  panel._rememberCleanConfiguration(configResult("full"));
  await panel._loadConfigDraft();
  panel._draft.usage_request_retention_days = 7;
  panel._setConfigDirty(true);
  panel._confirm = async () => false;
  await panel._navigate("assistant", "basics");
  assert.equal(panel._viewKey(), "assistant/basics",
    "an unsaved Retention draft remains with its assistant after loading the full projection");
  assert.equal(panel._draft.usage_request_retention_days, 7);
  assert.equal(panel._draft.chat_model, "gpt-4o");
}

{
  const panel = panelFor("assistant", "basics");
  panel._rememberCleanConfiguration(configResult("full"));
  await panel._loadConfigDraft();
  panel._draft.chat_model = "unsaved";
  panel._setConfigDirty(true);
  panel._invalidateAfterMutation("agent-a", "tools", "save");
  panel._hass = {callWS: async () => { throw new Error("dirty draft must remain owned"); }};
  await panel._loadSection();
  assert.equal(panel._draft.chat_model, "unsaved", "mutation invalidation keeps an unsaved active draft");
}

{
  const panel = panelFor();
  const reads = [];
  panel._hass = {callWS: async (message) => {
    reads.push(message.subentry_id);
    return {title:message.subentry_id, revision:message.subentry_id === "agent-a" ? "r1" : "r2",
      config:{model:message.subentry_id}};
  }};
  await panel._loadConfigDraft();
  assert.deepEqual(reads, ["agent-a"]);
  panel._draft.model = "unsaved";
  panel._configDirty = true;
  panel._clearConfigDraft();
  await panel._loadConfigDraft();
  assert.deepEqual(reads, ["agent-a"], "clean saved configuration is reused without another read");
  assert.equal(panel._draft.model, "agent-a", "discarded edits never enter the clean snapshot");

  panel._clearConfigDraft();
  panel._agentId = "agent-b";
  await panel._loadConfigDraft();
  assert.deepEqual(reads, ["agent-a", "agent-b"], "another agent gets its own configuration");
  panel._clearConfigDraft();
  panel._agentId = "agent-a";
  await panel._loadConfigDraft();
  assert.deepEqual(reads, ["agent-a", "agent-b"], "the original agent can reuse its own revision");
  panel._eocLiveMetadataCache = new Map([["local_handling", {
    agentId:"agent-a", revision:"r1", epoch:panel._eocLiveMetadataEpoch || 0,
    value:{intents:["cached"]},
  }]]);
  panel._clearConfigDraft();
  await panel._loadConfigDraft();
  assert.equal(panel._applyConfigurationLiveMetadata("capabilities/home-assistant"), true);
  assert.deepEqual(panel._configData.local_handling.intents, ["cached"],
    "same-revision live metadata survives unrelated navigation");

  const key = panel._configurationSnapshotKey("agent-a", "full");
  panel._cleanConfigSnapshots.get(key).loadedAt -= 31_000;
  panel._eocLiveMetadataCache = new Map([["local_handling", {agentId:"agent-a", revision:"r1", epoch:0, value:{}}]]);
  panel._clearConfigDraft();
  panel._hass.callWS = async () => ({title:"A updated", revision:"r3", config:{model:"new"}});
  await panel._loadConfigDraft();
  assert.equal(panel._draft.model, "new");
  assert.equal(panel._eocLiveMetadataCache.size, 0, "a new revision invalidates prior metadata");
  panel._invalidateAfterMutation("agent-a", "tools", "save");
  assert.equal(panel._cleanConfigSnapshots.has(key), false, "tool mutations invalidate config reuse");
}

{
  const panel = panelFor();
  panel._configData = {title:"A", config:{model:"cached"}};
  panel._draft = {model:"cached"};
  panel._draftTitle = "A";
  panel._draftAgentId = "agent-a";
  let calls = 0;
  panel._hass = {callWS: async () => { calls += 1; throw new Error("configuration should stay cached"); }};
  await panel._loadSection();
  assert.equal(calls, 0);
  assert.equal(panel._busy, false);
  assert.ok(panel.renderStates.length <= 1, "cached load does not churn renders");

  panel._agentId = "agent-b";
  panel._hass = {callWS: async (message) => {
    assert.equal(message.action, "get");
    assert.equal(message.subentry_id, "agent-b");
    return {title:"B", config:{model:"fresh"}};
  }};
  panel.renderStates = [];
  await panel._loadSection();
  assert.equal(panel._draftAgentId, "agent-b");
  assert.equal(panel._draft.model, "fresh");
  assert.equal(panel._result.config.model, "fresh");
  assert.ok(panel.renderStates.includes(true), "uncached load exposes a busy state");
  assert.equal(panel.renderStates.at(-1), false, "uncached load settles idle");
}

{
  const panel = panelFor("capabilities", "request-rules");
  let resolveAgents;
  const calls = [];
  panel._hass = {callWS: (message) => {
    calls.push(message);
    if (message.action === "agents") {
      return new Promise((resolve) => { resolveAgents = resolve; });
    }
    return Promise.resolve({});
  }};
  panel._loadSection = async () => {};
  const loading = panel._loadAgents("agent-a");
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(routeFeaturesReady("capabilities/request-rules"), true,
    "deep-link route asset starts before agents resolves");
  assert.deepEqual(calls.map((call) => call.action), ["agents"]);
  resolveAgents({agents, scopes:initialScopes, is_admin:true});
  await loading;
}

{
  const panel = panelFor("assistant", "basics");
  const listeners = new Map();
  panel.shadowRoot = {
    __eocRouteAssetWarmupBound:false,
    addEventListener(name, callback) { listeners.set(name, callback); },
  };
  panel._bindRouteAssetWarmup();
  panel._bindRouteAssetWarmup();
  assert.deepEqual([...listeners.keys()].sort(), ["focusin", "keydown", "pointerdown", "pointerout", "pointerover"]);

  const target = {
    dataset:{page:"guide"},
    closest() { return this; },
  };
  listeners.get("pointerdown")({target});
  await routeAssetPromise("guide");
  assert.equal(routeFeaturesReady("guide"), true,
    "navigation intent warms the target asset");
}

{
  const panel = panelFor("capabilities", "functions");
  const calls = [];
  let resolveConfig;
  panel._hass = {callWS: (message) => {
    calls.push(message);
    if (message.section === "configuration" && message.action === "get") {
      return new Promise((resolve) => { resolveConfig = resolve; });
    }
    return Promise.resolve({});
  }};
  const loading = panel._loadSection();
  assert.equal(
    calls.filter((call) => call.section === "configuration" && call.action === "get").length,
    1,
    "Functions configuration starts before its lazy UI module resolves",
  );
  resolveConfig({title:"A", config:{}, revision:"r1"});
  await loading;
}

{
  const panel = panelFor("data-memory", "conversations");
  const calls = [];
  let resolveConfig;
  let resolveScopes;
  panel._hass = {callWS: (message) => {
    calls.push(message);
    if (message.section === "configuration" && message.action === "get") {
      return new Promise((resolve) => { resolveConfig = resolve; });
    }
    if (message.section === "scopes" && message.action === "catalog") {
      return new Promise((resolve) => { resolveScopes = resolve; });
    }
    if (message.section === "conversations" && message.action === "list") {
      return Promise.resolve({sessions:[]});
    }
    if (message.section === "conversations" && message.action === "settings") {
      return Promise.resolve({});
    }
    if (message.section === "conversations" && message.action === "active") {
      return Promise.resolve({active:[]});
    }
    return Promise.resolve({});
  }};
  const loading = panel._loadSection();
  assert.deepEqual(
    calls.map((call) => [call.section, call.action]),
    [["configuration", "get"], ["scopes", "catalog"], ["conversations", "list"], ["conversations", "active"]],
    "History primary and secondary requests start together for a known scope",
  );
  assert.equal(calls.find((call) => call.section === "scopes")?.scope_kind, "archive");
  resolveConfig({title:"A", config:{}, revision:"r1"});
  resolveScopes({scopes:initialScopes});
  await loading;
  assert.equal(panel._busy, false);
  assert.ok(panel._contentData?.sessions);
}

{
  globalThis.localStorage.values.clear();
  globalThis.localStorage.setItem(AGENT_KEY, "agent-a");
  globalThis.localStorage.setItem(ENTRY_KEY, "entry-a");
  const panel = panelFor("overview", null);
  const calls = [];
  const resolvers = new Map();
  panel._hass = {callWS: (message) => {
    calls.push(message);
    return new Promise((resolve) => {
      const key = message.action === "detail"
        ? `detail:${message.kind}`
        : message.action;
      resolvers.set(key, resolve);
    });
  }};
  const loading = panel._loadAgents();

  for (const action of ["primary", "agents"]) {
    assert.equal(
      calls.filter((call) => call.action === action).length,
      1,
      `${action} starts before agents resolves`,
    );
  }
  assert.equal(calls.filter((call) => call.action === "snapshot").length, 0,
    "Broadcast waits until the principal Overview is rendered");

  resolvers.get("agents")({agents, scopes:initialScopes, is_admin:true});
  resolvers.get("primary")({
    agent:{...agents[0], guest_mode:{}},
    usage:{},
    conversations:{},
    load_errors:[],
    loading:{usage:true,memory:true,knowledge:true,guest_mode:true,setup_health:true},
    setup_health:{memory:{loading:true},knowledge:{loading:true}},
  });
  await loading;
  assert.equal(panel._result?.load_errors?.length, 0);
  for (const kind of ["usage", "memory", "knowledge", "guest_mode", "setup_health"]) {
    assert.equal(calls.some((call) => call.action === "detail" && call.kind === kind), true);
  }
  resolvers.get("detail:memory")({kind:"memory",agent:{memory_count:4},setup_health:{memory:{available:true,loading:false}}});
  await Promise.resolve();
  assert.equal(panel._selectedAgent().memory_count, 4,
    "a fast detail patches Overview without waiting for slower peers");
  resolvers.get("detail:usage")({kind:"usage",usage:{today:{total_tokens:7},month:{total_tokens:20}},agent:{tokens_today:7}});
  resolvers.get("detail:knowledge")({kind:"knowledge",agent:{knowledge_source_count:2},setup_health:{knowledge:{source_count:2,available:true,loading:false}}});
  resolvers.get("detail:guest_mode")({kind:"guest_mode",agent:{guest_mode:{state:"inactive"}}});
  resolvers.get("detail:setup_health")({kind:"setup_health",agent:{function_count:2},setup_health:{function_tools:{usable_count:2},exposed_entity_count:4}});
  await Promise.resolve();
  await Promise.resolve();
}

{
  globalThis.localStorage.values.clear();
  const panel = panelFor("overview", null);
  const calls = [];
  panel._hass = {callWS: async (message) => {
    calls.push(message);
    if (message.action === "agents") return {agents, scopes:initialScopes, is_admin:true};
    return {};
  }};
  panel._loadSection = async () => {};
  await panel._loadAgents("agent-a");
  assert.deepEqual(calls.map((call) => call.action), ["agents"]);
  assert.equal(panel._scopeId, "user:current");
}

{
  const panel = panelFor("data-memory", "memories");
  assert.equal(panel._sectionCacheKey(), null);
  const calls = [];
  panel._hass = {callWS: async (message) => {
    calls.push(message);
    if (message.section === "scopes") return {scopes:[{...initialScopes[0], memory_count:1, conversation_count:0}, {scope_id:"shared", scope_type:"shared", display_name:"Shared", memory_count:0, conversation_count:0}]};
    if (message.section === "memories") return {memories:[{memory_id:`memory-${message.subentry_id}`}]};
    return {};
  }};
  await panel._loadSection();
  assert.deepEqual(calls.map((call) => call.section), ["scopes", "memories"]);
  panel.renderStates = [];
  await panel._loadSection();
  assert.deepEqual(calls.map((call) => call.section), ["scopes", "memories", "memories"]);
  assert.deepEqual(panel.renderStates, [true, false]);

  panel._memoryKind = "temporary";
  await panel._loadSection();
  panel._scopeId = "shared";
  await panel._loadSection();
  assert.deepEqual(calls.map((call) => call.section), ["scopes", "memories", "memories", "scopes", "memories", "memories"]);
  assert.equal(calls[0].scope_kind, "memory");
  assert.equal(calls[3].scope_kind, "temporary");
  assert.deepEqual(calls.slice(4, 6).map((call) => [call.action, call.scope_id]), [
    ["temporary_list", "user:current"],
    ["temporary_list", "shared"],
  ]);

  panel._page = "guide";
  panel._subsection = null;
  await panel._loadSection();
  panel._page = "data-memory";
  panel._subsection = "memories";
  await panel._loadSection();
  assert.deepEqual(calls.slice(6).map((call) => call.section), ["memories"]);

  panel._agentId = "agent-b";
  panel._scopeId = "user:current";
  panel._applyScopes(initialScopes);
  await panel._loadSection();
  assert.deepEqual(calls.slice(7).map((call) => [call.section, call.subentry_id]), [
    ["scopes", "agent-b"],
    ["memories", "agent-b"],
  ]);
  assert.equal(panel._sectionCache.size, 0);
}

{
  const panel = panelFor("capabilities", "request-rules");
  panel._sectionCache.set("agent-a|capabilities/request-rules", {rules:[{id:"one"}]});
  panel._sectionCache.set("agent-b|capabilities/request-rules", {rules:[{id:"two"}]});
  panel._hass = {callWS: async () => ({rule:{id:"one"}})};
  await panel._call("request_rules", "update", {rule_id:"one", rule:{}});
  assert.equal(panel._sectionCache.has("agent-a|capabilities/request-rules"), false);
  assert.equal(panel._sectionCache.has("agent-b|capabilities/request-rules"), true);

  panel._sectionCache.set("agent-a|data-memory/knowledge", {sources:[]});
  panel._sectionCache.set("agent-b|data-memory/knowledge", {sources:[]});
  panel._invalidateAfterMutation("agent-a", "knowledge", "create");
  assert.equal(panel._sectionCache.has("agent-a|data-memory/knowledge"), false);
  assert.equal(panel._sectionCache.has("agent-b|data-memory/knowledge"), true);

  panel._scopeCatalogCache.set("agent-a|scopes|memory", initialScopes);
  panel._scopeCatalogCache.set("agent-b|scopes|memory", initialScopes);
  panel._scopeCatalogVisitKey = "agent-a|scopes|memory";
  panel._invalidateAfterMutation("agent-a", "memories", "delete");
  assert.equal(panel._scopeCatalogCache.has("agent-a|scopes|memory"), false);
  assert.equal(panel._scopeCatalogCache.has("agent-b|scopes|memory"), true);
  assert.equal(panel._scopeCatalogVisitKey, null);
}

{
  const panel = panelFor("data-memory", "conversations");
  panel._data.is_admin = false;
  const calls = [];
  panel._hass = {callWS: async (message) => {
    calls.push(message);
    if (message.section === "scopes") return {scopes:initialScopes};
    if (message.section === "conversations") return {sessions:[], settings:{}};
    return {};
  }};
  await panel._loadSection();
  await panel._loadSection();
  assert.equal(calls.filter((call) => call.section === "scopes").length, 1);

  panel._page = "guide";
  panel._subsection = null;
  await panel._loadSection();
  panel._page = "data-memory";
  panel._subsection = "conversations";
  await panel._loadSection();
  assert.equal(calls.filter((call) => call.section === "scopes").length, 1);
}

{
  const panel = panelFor("capabilities", "request-rules");
  let resolveRules;
  panel._hass = {callWS: () => new Promise((resolve) => { resolveRules = resolve; })};
  const oldLoad = panel._loadSection();
  panel._page = "data-memory";
  panel._subsection = "knowledge";
  const knowledge = {sources:[{source_id:"current"}]};
  const knowledgeKey = "agent-a|data-memory/knowledge";
  panel._sectionCache.set(knowledgeKey, knowledge);
  panel._eocSectionCacheTimes.set(knowledgeKey, Date.now());
  await panel._loadSection();
  resolveRules({rules:[{id:"old"}]});
  await oldLoad;
  assert.equal(panel._result, knowledge);
}

{
  const panel = panelFor("capabilities", "request-rules");
  let calls = 0;
  panel._hass = {callWS: async (message) => {
    calls += 1;
    assert.equal(message.section, "service_catalog");
    return {services:{light:{turn_on:{name:"Turn on", fields:{}}}}};
  }};
  const [first, second] = await Promise.all([panel._loadServiceCatalog(), panel._loadServiceCatalog()]);
  assert.equal(first, second);
  assert.equal(await panel._loadServiceCatalog(), first);
  assert.equal(calls, 1);
}

{
  let shadowRootListeners = 0;
  const shadowRoot = {
    addEventListener() { shadowRootListeners += 1; },
    querySelector() { return null; },
    querySelectorAll() { return []; },
  };
  const panel = {shadowRoot, _result:{rules:[]}, _serviceCatalog:null};
  bindRequestRules(panel);
  bindRequestRules(panel);
  assert.equal(shadowRootListeners, 0, "Request Rule collection ownership should not add a persistent root listener");
}

{
  const panel = panelFor("usage-maintenance", "usage");
  const started = [];
  const resolvers = new Map();
  panel._hass = {callWS: (message) => {
    started.push(message.action);
    return new Promise((resolve) => resolvers.set(message.action, resolve));
  }};
  const loading = panel._loadSection();

  for (const action of ["summary", "runs", "retention"]) {
    assert.ok(
      started.includes(action),
      `${action} starts before the lazy Usage UI module resolves`,
    );
  }
  assert.equal(
    routeFeaturesReady("usage-maintenance/usage"),
    false,
    "Usage requests start while the chart feature is still cold",
  );
  await routeAssetPromise("usage-maintenance/usage");
  assert.ok(started.includes("daily"), "daily starts once its lazy usage-data helper resolves");

  resolvers.get("summary")?.({});
  resolvers.get("daily")?.({days:[]});
  resolvers.get("runs")?.({runs:[]});
  resolvers.get("retention")?.({});
  await loading;
  assert.equal(routeFeaturesReady("usage-maintenance/usage"), true);
}

{
  assert.equal(requestRuleSearchText({name:"Good Night",phrases:["Bed Time"],action_type:"local_action"}), "good night bed time local_action");
  let listQueries = 0;
  const firstCard = {dataset:{ruleKey:"one"}, hidden:false};
  const secondCard = {dataset:{ruleKey:"two"}, hidden:false};
  const empty = {hidden:true};
  const count = {textContent:""};
  const search = {value:"night"};
  const list = {
    querySelectorAll(selector) {
      assert.equal(selector, "[data-rule-key]");
      listQueries += 1;
      return [firstCard, secondCard];
    },
    querySelector(selector) {
      return selector === "[data-eoc-rule-search-empty]" ? empty : null;
    },
    append() {},
  };
  const root = {
    querySelector(selector) {
      if (selector === "#rule-search") return search;
      if (selector === ".rule-list") return list;
      if (selector === ".rule-toolbar .count") return count;
      return null;
    },
    ownerDocument:{createElement:() => null},
  };
  const panel = {
    _viewKey:() => "capabilities/request-rules",
    _query:"",
    _result:{rules:[
      {id:"one",name:"Good Night",phrases:["Bed Time"],action_type:"local_action"},
      {id:"two",name:"Think Carefully",phrases:["reason"],action_type:"model_routing"},
    ]},
  };

  assert.equal(applyRequestRuleSearch(panel, root), 1);
  assert.equal(firstCard.hidden, false);
  assert.equal(secondCard.hidden, true);
  assert.equal(listQueries, 1);

  search.value = "think";
  assert.equal(applyRequestRuleSearch(panel, root), 1);
  assert.equal(firstCard.hidden, true);
  assert.equal(secondCard.hidden, false);
  assert.equal(listQueries, 1, "keystrokes reuse cached cards and normalized search text");

  panel._eocRequestRuleCollectionRevision = 1;
  applyRequestRuleSearch(panel, root);
  assert.equal(listQueries, 2, "collection changes rebuild the search representation");
}

// A large delivery history must resolve satellite names with bounded work.
// Reading each satellite ID for every delivery would make this quadratic.
{
  const size = 300;
  let idReads = 0;
  const satellites = Array.from({length:size}, (_, index) => ({
    get id() { idReads++; return `satellite-${index}`; },
    name:`Satellite ${index}`,
  }));
  const host = {innerHTML:""};
  const panel = {
    _viewKey:() => "overview",
    _e:String,
    shadowRoot:{querySelector:(selector) => selector === "#broadcast-card" ? host : null, querySelectorAll:() => []},
  };
  const deliveries = Object.fromEntries(satellites.map((_, index) => [`satellite-${index}`, {status:"delivered"}]));
  await bindBroadcast(panel, Promise.resolve({
    enabled:false,
    can_manage:false,
    catalog:{satellites, areas:[]},
    history:[{message:"Update", created_at:"2026-01-01T00:00:00Z", deliveries}],
  }));
  assert.match(host.innerHTML, /Satellite 299/);
  assert.ok(idReads <= size * 3, `broadcast history performed ${idReads} satellite ID reads for ${size} deliveries`);
}
// Exercise cache ownership through the actual host, with no performance installer.
{
  const panel = panelFor("overview", null);
  const key = panel._sectionCacheKey();
  let resolveRefresh;
  let reads = 0;
  panel._hass = {callWS: (message) => {
    if (message.section !== "overview") return Promise.resolve({});
    reads++;
    return reads === 1
      ? Promise.resolve({usage:{today:{total_tokens:3}}})
      : new Promise((resolve) => { resolveRefresh = resolve; });
  }};
  await panel._loadSection();
  assert.equal(panel._sectionCache.get(key).usage.today.total_tokens, 3);
  await panel._loadSection();
  assert.equal(reads, 1, "a recent Overview return reuses its summary");
  panel._eocSectionCacheTimes.set(key, Date.now() - 31_000);
  const refresh = panel._loadSection();
  await Promise.resolve();
  assert.equal(panel._busy, false, "expired Overview stays visible while refreshing");
  assert.equal(panel._result.usage.today.total_tokens, 3);
  resolveRefresh({usage:{today:{total_tokens:4}}});
  await refresh;
  assert.equal(panel._result.usage.today.total_tokens, 4);
  panel._invalidateAfterMutation("agent-a", "configuration", "save");
  assert.equal(panel._sectionCache.has(key), false, "configuration mutation invalidates Overview");
}

{
  const panel = panelFor("usage-maintenance", "usage");
  const reads = new Map();
  let releaseSummary;
  panel._hass = {callWS: async (message) => {
    reads.set(message.action, (reads.get(message.action) || 0) + 1);
    if (message.action === "summary") {
      if (reads.get("summary") === 2) return new Promise((resolve) => { releaseSummary = resolve; });
      return {lifetime:{total_tokens:10}};
    }
    if (message.action === "daily") return {days:[]};
    if (message.action === "runs") return {runs:[]};
    if (message.action === "retention") return {};
    throw new Error(`Unexpected usage action ${message.action}`);
  }};
  await panel._loadSection();
  const key = panel._sectionCacheKey();
  const loadedAt = panel._eocSectionCacheTimes.get(key);
  await panel._loadSection();
  assert.equal(reads.get("summary"), 1, "fresh Usage summary is reused");
  assert.equal(reads.get("daily"), 1, "fresh daily chart is reused");
  assert.equal(reads.get("runs"), 2, "recent details still refresh on revisit");
  assert.equal(panel._eocSectionCacheTimes.get(key), loadedAt, "cache visits do not slide its TTL");
  panel._eocSectionCacheTimes.set(key, Date.now() - 31_000);
  const refresh = panel._loadSection();
  await Promise.resolve();
  assert.equal(panel._busy, false, "stale aggregates remain visible while refreshing");
  assert.equal(panel._result.summary.lifetime.total_tokens, 10);
  while (!releaseSummary) await Promise.resolve();
  releaseSummary({lifetime:{total_tokens:12}});
  await refresh;
  assert.equal(panel._result.summary.lifetime.total_tokens, 12);
  panel._eocSectionCacheTimes.set(key, Date.now() - 31_000);
  panel._hass.callWS = async (message) => {
    if (message.action === "summary") throw new Error("summary unavailable");
    if (message.action === "daily") return {days:[]};
    if (message.action === "runs") return {runs:[]};
    if (message.action === "retention") return {};
  };
  await panel._loadSection();
  assert.equal(panel._result.summary.lifetime.total_tokens, 12,
    "a failed revalidation does not erase useful cached aggregates");
  assert.equal(panel._result.load_errors[0].key, "summary");
  panel._invalidateAfterMutation("agent-a", "usage", "clear_details");
  assert.equal(panel._sectionCache.has(key), false, "detail clearing invalidates the prior latest response");
}

{
  const panel = panelFor("capabilities", "request-rules");
  const key = panel._sectionCacheKey();
  panel._sectionCache.set(key, {rules:["stale"]});
  let loads = 0;
  panel._hass = {callWS: async () => ({rules:[++loads]})};
  await panel._loadSection();
  assert.equal(loads, 1, "cache without timestamp refreshes");
  const fetchedAt = panel._eocSectionCacheTimes.get(key);
  await panel._loadSection();
  assert.equal(loads, 1);
  assert.equal(panel._eocSectionCacheTimes.get(key), fetchedAt, "hits do not slide TTL");
  panel._eocSectionCacheTimes.set(key, Date.now() - 31_000);
  await panel._loadSection();
  assert.equal(loads, 2);
}

{
  const panel = panelFor("data-memory", "conversations");
  panel._contentData = {sessions:{sessions:[{session_id:"one"}], returned:1, total:1}};
  panel._confirm = async () => true;
  panel._toast = () => {};
  let resolveDelete;
  panel._call = () => new Promise((resolve) => { resolveDelete = resolve; });
  const deleting = panel._deleteSession("one");
  await Promise.resolve();
  panel._scopeId = "user:other";
  resolveDelete({deleted_sessions:1});
  await deleting;
  assert.equal(panel._contentData.sessions.sessions.length, 1,
    "late deletion cannot patch a different History scope");
}
