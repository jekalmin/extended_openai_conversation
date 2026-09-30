import {updateRequestRuleEmptyState} from "./request-rule-empty-state.js";
import {ensureGuideModule} from "./guide-page.js";
import {ensureOverviewModule, startOverviewDetailReads} from "./overview-page.js";
import {SECTION_CACHE_TTL_MS} from "./management-cache.js";
const REQUEST_RULES_VIEW = "capabilities/request-rules";
const CONFIG_TRACE_PREFIX = "extended-openai:config-read";
const CONFIG_TRACE_LIMIT = 100;
const configurationTraceEntries = [];
let configurationTraceSequence = 0;

export function markConfigurationRead(event, detail) {
  const api = globalThis.performance;
  if (typeof api?.mark !== "function") return;
  const name = `${CONFIG_TRACE_PREFIX}:${event}:${++configurationTraceSequence}`;
  try {
    api.mark(name, {detail});
    configurationTraceEntries.push({name, kind: "mark"});
    while (configurationTraceEntries.length > CONFIG_TRACE_LIMIT) {
      const old = configurationTraceEntries.shift();
      if (old.kind === "mark") api.clearMarks?.(old.name);
      else api.clearMeasures?.(old.name);
    }
  } catch (_err) { /* Optional tracing must not affect a configuration read. */ }
}

export function measureConfigurationRead(event, detail) {
  const api = globalThis.performance;
  if (typeof api?.mark !== "function" || typeof api?.measure !== "function") return () => {};
  const name = `${CONFIG_TRACE_PREFIX}:${event}:${++configurationTraceSequence}`;
  const start = `${name}:start`;
  try { api.mark(start); } catch (_err) { return () => {}; }
  return (status) => {
    const end = `${name}:end`;
    try {
      api.mark(end);
      api.measure(name, {start, end, detail: {...detail, status}});
      configurationTraceEntries.push({name, kind: "measure"});
      while (configurationTraceEntries.length > CONFIG_TRACE_LIMIT) {
        const old = configurationTraceEntries.shift();
        if (old.kind === "mark") api.clearMarks?.(old.name);
        else api.clearMeasures?.(old.name);
      }
    } catch (_err) { /* Best-effort browser trace only. */ }
    finally {
      try { api.clearMarks?.(start); api.clearMarks?.(end); } catch (_err) { /* Best-effort cleanup. */ }
    }
  };
}
const CONFIG_VIEWS = new Set([
  "capabilities/home-assistant",
  "capabilities/web-skills",
]);
// These views consume the normal full configuration snapshot. Retention uses a
// separate projection; unauthorised readers must never speculate on config.
export function needsFullConfiguration(view, isAdmin = true) {
  return isAdmin && (String(view || "").startsWith("assistant/")
    || ["capabilities/home-assistant", "capabilities/web-skills", "capabilities/functions",
      "data-memory/conversations", "data-memory/memory-settings"].includes(view));
}

export function getConfigurationEditor() { return getRouteFeature("agent-config"); }
export function getConfigurationTools() { return getRouteFeature("agent-config-tools"); }

const CONFIG_SECTION_FAMILIES = {
  general: "primary", conversation: "primary", context: "primary", model: "primary",
  local: "capabilities", capabilities: "capabilities", archive: "capabilities",
  prompt: "media", voice: "media", speech: "media",
  retention: "maintenance", backup: "maintenance",
};
export function configurationSectionFamily(view, sections = null) {
  const section = sections?.find((item) => CONFIG_SECTION_FAMILIES[item]);
  if (section) return `agent-config-sections-${CONFIG_SECTION_FAMILIES[section]}`;
  if (view === "assistant/advanced") return "agent-config-sections-capabilities";
  if (String(view || "").startsWith("assistant/")) {
    return `agent-config-sections-${["prompt-context", "voice", "speech"].includes(view.split("/")[1]) ? "media" : "primary"}`;
  }
  if (view === "usage-maintenance/retention") return "agent-config-sections-maintenance";
  return "agent-config-sections-capabilities";
}

export function routeAssetKind(view) {
  if (view === "capabilities/functions") return "agent-config-tools";
  if (view === "usage-maintenance/backup-restore") return null;
  if (String(view || "").startsWith("assistant/") || CONFIG_VIEWS.has(view)) return "agent-config";
  if (view === REQUEST_RULES_VIEW) return "request-rules";
  return null;
}

const featureModules = new Map();
const featurePromises = new Map();
// Only routes whose data loader itself lives in the lazy feature must wait for it.
// Ordinary panel-owned data requests should begin while route assets download.
const DATA_FEATURES = new Set([
  "capabilities/quiet-hours",
  "usage-maintenance/request-debug",
]);
const featureLoaders = {
  "agent-config": () => import("./agent-config-editor.js"),
  "agent-config-sections-primary": () => import("./agent-config-sections-primary.js"),
  "agent-config-sections-capabilities": () => import("./agent-config-sections-capabilities.js"),
  "agent-config-sections-media": () => import("./agent-config-sections-media.js"),
  "agent-config-sections-maintenance": () => import("./agent-config-sections-maintenance.js"),
  "agent-config-tools": () => import("./agent-config-tools.js"),
  "usage-maintenance/backup-restore": () => import("./backup-route-ui.js"),
  "status": () => import("./management-feature-status.js"),
  "capabilities": () => import("./management-capabilities-ia.js"),
  "configuration": () => import("./management-configuration-feature.js"),
  "memory-browser": () => import("./guest-mode-ui.js"),
  "data-memory/memories": () => import("./management-memory-feature.js"),
  "data-memory/knowledge": () => import("./management-knowledge-feature.js"),
  "capabilities/guest-mode": () => import("./management-guest-feature.js"),
  "capabilities/request-rules": () => import("./request-rules-ui.js"),
  "capabilities/quiet-hours": () => import("./quiet-hours-ui.js"),
  "capabilities/functions": () => import("./management-function-repair.js"),
  "data-memory/conversations": () => import("./management-history-pagination.js"),
  "usage-maintenance/usage": () => import("./usage-chart.js"),
  "usage-maintenance/retention": () => import("./retention-settings-ui.js"),
  "usage-maintenance/request-debug": () => import("./debug-management.js"),
  "usage-maintenance/diagnostics": () => import("./management-provider-credentials.js"),
  "assistant/voice": () => import("./voice-identity-core.js"),
  "data-memory/memory-settings": () => import("./memory-settings-ui.js"),
};
export function getRouteFeature(view) { return featureModules.get(view); }

function routeFeatureKeys(view) {
  const keys = [view];
  const assetKind = routeAssetKind(view);
  if (assetKind === "agent-config" || assetKind === "agent-config-tools") keys.push(assetKind);
  if (assetKind === "agent-config") keys.push(configurationSectionFamily(view));
  // Configuration guidance is additive and never blocks route readiness.
  if (["data-memory/memories", "capabilities/guest-mode"].includes(view)) keys.push("memory-browser");
  if (["capabilities/home-assistant", "capabilities/web-skills"].includes(view)) keys.push("capabilities");
  if (["data-memory/memories", "usage-maintenance/diagnostics"].includes(view)) keys.push("status");
  return keys;
}

function warmSupplementalRouteFeatures(panel, view, token) {
  if (view === "data-memory/conversations" && panel._data?.is_admin) {
    panel._eocHistoryEditorLoading = true;
    panel._eocHistoryEditorError = null;
    Promise.all([
      featureAssetPromise("agent-config"),
      featureAssetPromise(configurationSectionFamily(view, ["archive"])),
    ]).then(() => {
      if (!isCurrentLazyLoad(panel, view, token)) return;
      panel._eocHistoryEditorLoading = false;
      panel._patchHistorySettings?.();
    }, (error) => {
      if (!isCurrentLazyLoad(panel, view, token)) return;
      panel._eocHistoryEditorLoading = false;
      panel._eocHistoryEditorError = error?.message || String(error);
      panel._patchHistorySettings?.();
    });
    return;
  }
  if (routeAssetKind(view) !== "agent-config" && view !== "data-memory/memory-settings") return;
  const pending = featureAssetPromise("configuration");
  pending?.then((module) => {
    if (!isCurrentLazyLoad(panel, view, token)) return;
    // Guidance decorates already-rendered configuration controls and does not own
    // their data or markup, so it can safely arrive after first usable paint.
    module?.enhanceConfigurationGuidance?.(panel);
  }).catch(() => {});
}

export function routeFeaturesReady(view) {
  return routeFeatureKeys(view).every((key) => !featureLoaders[key] || featureModules.has(key));
}

function routeFeaturePromise(view) {
  const pending = routeFeatureKeys(view).map(featureAssetPromise).filter(Boolean);
  return pending.length ? Promise.all(pending) : null;
}

function featureAssetPromise(view) {
  if (!featureLoaders[view] || featureModules.has(view)) return null;
  if (!featurePromises.has(view)) {
    featurePromises.set(view, featureLoaders[view]().then((module) => {
      featureModules.set(view, module);
      return module;
    }).finally(() => featurePromises.delete(view)));
  }
  return featurePromises.get(view);
}

export function routeAssetPromise(view, panel) {
  const feature = routeFeaturePromise(view);
  const core = coreAssetPromise(view);
  return feature ? Promise.all([feature, core]) : core;
}

export function warmRouteAsset(view) {
  const pending = routeAssetPromise(view);
  pending?.catch?.(() => {});
  return pending;
}

const INTENT_READS = new Map([
  ["overview", ["overview", "summary"]],
  ["data-memory/knowledge", ["knowledge", "list"]],
  ["capabilities/request-rules", ["request_rules", "list"]],
  ["usage-maintenance/retention", ["configuration", "retention_get"]],
]);
const INTENT_READ_TTL_MS = 3_000;

// Only read-only, parameter-free route requests enter this registry. The
// generation and agent identity prevent a pre-mutation or other-agent result
// from being reused by a later navigation.
export function prefetchIntentRead(panel, view) {
  const operation = INTENT_READS.get(view);
  const agent = panel._selectedAgent?.();
  if (!operation || !agent || panel._viewKey?.() === view || panel._configDirty) return null;
  if (view === "usage-maintenance/retention") {
    const active = panel._configData;
    if (panel._draftAgentId === agent.subentry_id && active?.config
        && active.projection !== "retention") return null;
    if (panel._freshCleanConfiguration?.(agent.subentry_id, "retention")) return null;
  }
  const cacheKey = panel._sectionCacheKey?.(view);
  const loadedAt = panel._eocSectionCacheTimes?.get(cacheKey);
  if (loadedAt && Date.now() - loadedAt < SECTION_CACHE_TTL_MS
      && panel._sectionCache?.has(cacheKey)) return null;
  const key = `${agent.entry_id}|${agent.subentry_id}|${panel._cacheGeneration || 0}|${view}`;
  panel._eocPendingRouteReads ||= new Map();
  const pending = panel._eocPendingRouteReads.get(key);
  if (pending && Date.now() - pending.started < INTENT_READ_TTL_MS) return pending.promise;
  const record = {started: Date.now(), promise: null};
  record.promise = panel._call(...operation).catch((error) => {
    if (panel._eocPendingRouteReads.get(key) === record) panel._eocPendingRouteReads.delete(key);
    throw error;
  });
  // A speculative read may fail without a navigation ever consuming it.
  record.promise.catch(() => {});
  panel._eocPendingRouteReads.set(key, record);
  return record.promise;
}

export function consumeIntentRead(panel, view, section, action) {
  const operation = INTENT_READS.get(view);
  const agent = panel._selectedAgent?.();
  if (!operation || operation[0] !== section || operation[1] !== action || !agent) {
    return panel._call(section, action);
  }
  const key = `${agent.entry_id}|${agent.subentry_id}|${panel._cacheGeneration || 0}|${view}`;
  const pending = panel._eocPendingRouteReads?.get(key);
  if (!pending || Date.now() - pending.started >= INTENT_READ_TTL_MS) {
    panel._eocPendingRouteReads?.delete(key);
    return panel._call(section, action);
  }
  panel._eocPendingRouteReads.delete(key);
  return pending.promise;
}

function coreAssetPromise(view) {
  if (view === "overview") return ensureOverviewModule();
  if (view === "guide") return ensureGuideModule();
  return null;
}

export function requestRuleSearchText(rule) {
  return `${rule?.name || ""} ${(rule?.phrases || []).join(" ")} ${rule?.action_type || ""}`.toLocaleLowerCase();
}

export function matchesRequestRuleSearch(rule, query) {
  const normalized = String(query || "").trim().toLocaleLowerCase();
  return !normalized || requestRuleSearchText(rule).includes(normalized);
}

function requestRuleSearchEntries(panel, root, rules) {
  const list = root.querySelector(".rule-list");
  const revision = panel._eocRequestRuleCollectionRevision || 0;
  const cached = panel._eocRequestRuleSearchCache;
  if (
    cached?.rules === rules
    && cached.list === list
    && cached.revision === revision
  ) return cached.entries;

  const cards = new Map(
    [...(list?.querySelectorAll?.("[data-rule-key]") || [])]
      .map((card) => [String(card.dataset.ruleKey), card]),
  );
  const entries = rules.map((rule) => ({
    rule,
    card: cards.get(String(rule.id)),
    searchText: requestRuleSearchText(rule),
  })).filter((entry) => entry.card);
  panel._eocRequestRuleSearchCache = {rules, list, revision, entries};
  return entries;
}

export function applyRequestRuleSearch(panel, root = panel?.shadowRoot) {
  if (!root || panel?._viewKey?.() !== REQUEST_RULES_VIEW) return 0;
  const query = String(root.querySelector("#rule-search")?.value ?? panel._query ?? "");
  const normalized = query.trim().toLocaleLowerCase();
  const rules = panel._result?.rules || [];
  const entries = requestRuleSearchEntries(panel, root, rules);
  let visible = 0;

  for (const {rule, card, searchText} of entries) {
    const filter = panel._ruleGroupFilter || "all";
    const matches = (!normalized || searchText.includes(normalized)) && (filter === "all" || (filter === "ungrouped" ? !rule?.group_id : rule?.group_id === filter));
    if (card.hidden === matches) card.hidden = !matches;
    if (matches) visible += 1;
  }

  const canReorder = !normalized && (panel._ruleGroupFilter || "all") === "all";
  root.querySelectorAll?.("[data-rule-key]").forEach((card) => { card.draggable = canReorder; card.querySelectorAll(".rule-move").forEach((button) => { button.disabled = !canReorder || button.dataset.direction === "up" && card === card.parentElement.querySelector("[data-rule-key]") || button.dataset.direction === "top" && card === card.parentElement.querySelector("[data-rule-key]") || button.dataset.direction === "down" && card === card.parentElement.querySelector("[data-rule-key]:last-of-type") || button.dataset.direction === "bottom" && card === card.parentElement.querySelector("[data-rule-key]:last-of-type"); }); });
  const helper=root.querySelector(".rule-filter-help");if(helper)helper.hidden=canReorder;

  const list = root.querySelector(".rule-list");
  const empty = list?.querySelector("[data-eoc-rule-search-empty]");
  if (empty) {
    updateRequestRuleEmptyState(empty, query, panel._ruleGroupFilter);
    empty.hidden = visible > 0 || !rules.length && (panel._ruleGroupFilter || "all") === "all" && !normalized;
  }

  const count = root.querySelector(".rule-toolbar .count");
  if (count) {
    const total = rules.length;
    count.hidden = !String(query).trim() && (panel._ruleGroupFilter || "all") === "all";
    count.textContent = `Showing ${visible} of ${total} rules`;
  }
  return visible;
}

export function bindRequestRuleSearch(panel) {
  const root = panel.shadowRoot;
  if (!root || root.__eocInPlaceRuleSearchBound) return;
  root.__eocInPlaceRuleSearchBound = true;
  root.addEventListener("input", (event) => {
    const input = event.target;
    if (input?.id !== "rule-search") return;
    event.stopImmediatePropagation();
    panel._query = input.value;
    applyRequestRuleSearch(panel, root);
  }, true);
}

function isCurrentLazyLoad(panel, view, assetToken) {
  return panel._viewKey() === view && panel._eocViewAssetToken === assetToken;
}

export function loadSectionAlongsideAsset(
  panel,
  silent,
  loadSectionData,
  view,
  assetPromise,
  assetToken,
) {
  let sectionPromise;
  try {
    sectionPromise = Promise.resolve(
      loadSectionData.call(panel, silent),
    );
  } catch (err) {
    sectionPromise = Promise.reject(err);
  }

  return Promise.allSettled([assetPromise, sectionPromise]).then(([assetResult, sectionResult]) => {
    if (!isCurrentLazyLoad(panel, view, assetToken)) return undefined;
    const failure = assetResult.status === "rejected"
      ? assetResult.reason
      : sectionResult.status === "rejected"
        ? sectionResult.reason
        : null;
    if (!failure) {
      // A route-owned feature can become available after the data renderer ran.
      // If the settled data already rendered with its feature loaded, another
      // full render would only rebuild the same route markup.
      if (panel._eocRenderedRoute !== `${panel._agentId}|${view}`
          || !panel._eocRenderedFeatureReady || panel._busy
          || panel._eocDeferredEditorRender) panel._render();
      return sectionResult.value;
    }
    panel._busy = false;
    panel._error = `Unable to load this frontend section: ${failure?.message || String(failure)}`;
    panel._render();
    return undefined;
  });
}

async function loadRouteData(panel, silent, view, token) {
  const feature = getRouteFeature(view);
  if (view === "capabilities/quiet-hours") return feature.loadQuietHours(panel, silent);
  if (view === "usage-maintenance/request-debug") return feature.loadRequestDebug(panel, silent);
  if (view === "data-memory/conversations") {
    panel._eocHistoryMode = "list";
    panel._eocHistoryQuery = "";
  }
  return panel._loadSectionData(silent);
}

// One native route entry point owns lazy assets and stale completion handling.
export function loadRoute(panel, silent = false) {
  const view = panel._viewKey();
  if (view !== "overview") panel._eocOverviewBroadcastPromise = null;
  const token = (panel._eocViewAssetToken || 0) + 1;
  panel._eocViewAssetToken = token;
  const feature = routeFeaturePromise(view);
  const asset = coreAssetPromise(view);
  warmSupplementalRouteFeatures(panel, view, token);
  if (!feature && !asset) return loadRouteData(panel, silent, view, token);
  let loadData = () => loadRouteData(panel, silent, view, token);
  if (feature && DATA_FEATURES.has(view)) {
    // Invalidate prior data work immediately, before awaiting route code.
    ++panel._loadToken;
    if (!silent) { panel._busy = true; panel._render(); }
    loadData = async function() {
      await feature;
      if (!isCurrentLazyLoad(panel, view, token)) return;
      return loadRouteData(panel, silent, view, token);
    };
  }
  return loadSectionAlongsideAsset(panel, silent, loadData, view, Promise.all([feature, asset]), token);
}

const WS_TYPE = "extended_openai_conversation_responses/management";
export const AGENT_KEY = "extended-openai-agent";
export const ENTRY_KEY = "extended-openai-agent-entry";

export function applyOverviewResult(panel, result, {cache = true} = {}) {
  const agent = panel._selectedAgent?.();
  if (!agent || !result) return false;
  if (result.agent) Object.assign(agent, result.agent);
  const {agent: _agent, ...overview} = result;
  panel._contentData = null;
  panel._result = overview;
  if (cache && panel._sectionCacheKey && panel._sectionCache) {
    const key = panel._sectionCacheKey("overview");
    if (key) {
      panel._sectionCache.set(key, overview);
      panel._eocSectionCacheTimes?.set(key, Date.now());
    }
  }
  panel._error = null;
  panel._busy = false;
  panel._render();
  return true;
}

export function startStoredOverviewPrefetch(
  panel,
  preferredSubentryId,
  overviewAsset = ensureOverviewModule(),
) {
  if (panel._viewKey?.() !== "overview") return null;
  const subentryId = preferredSubentryId || globalThis.localStorage?.getItem?.(AGENT_KEY);
  const entryId = globalThis.localStorage?.getItem?.(ENTRY_KEY);
  if (!subentryId || !entryId) return null;
  panel._markColdLifecycle?.("overview-summary-start");
  const overviewPrimary = panel._hass.callWS({
    type: WS_TYPE,
    section: "overview",
    action: "primary",
    entry_id: entryId,
    subentry_id: subentryId,
  }).then(
    (result) => {
      panel._markColdLifecycle?.("overview-summary-complete", {status: "fulfilled"});
      return result;
    },
    (err) => {
      panel._markColdLifecycle?.("overview-summary-complete", {status: "rejected"});
      throw err;
    },
  );
  return {
    entryId,
    subentryId,
    overviewAsset,
    overviewPrimary,
    promise: Promise.allSettled([
      overviewAsset,
      overviewPrimary,
    ]),
  };
}

export function startStoredConfigurationPrefetch(panel, preferredSubentryId) {
  const view = panel._viewKey?.();
  const action = view === "usage-maintenance/retention"
    ? "retention_get"
    : needsFullConfiguration(view) ? "get" : null;
  if (!action) return null;
  const subentryId = preferredSubentryId || globalThis.localStorage?.getItem?.(AGENT_KEY);
  const entryId = globalThis.localStorage?.getItem?.(ENTRY_KEY);
  // Keep the last cold-read decision inspectable without logging expected misses.
  const diagnostics = panel._eocConfigurationReadDiagnostics ||= {};
  const detail = {entryId, subentryId, view, action, status: "started", requestStatus: "not-started", reason: null};
  diagnostics.prefetch = detail;
  if (!subentryId || !entryId) {
    detail.status = "skipped";
    detail.reason = !subentryId ? "missing-stored-agent" : "missing-stored-entry";
    markConfigurationRead("prefetch-skipped", {view, action, status: detail.status, reason: detail.reason});
    return null;
  }
  if (panel._draftAgentId === subentryId && panel._configData?.config && !panel._configDataStale
      && (panel._configData.projection === "retention" ? "retention_get" : "get") === action) {
    detail.status = "skipped";
    detail.reason = "active-config";
    markConfigurationRead("prefetch-skipped", {view, action, status: detail.status, reason: detail.reason});
    return null;
  }
  if (panel._freshCleanConfiguration?.(subentryId, action === "retention_get" ? "retention" : "full")) {
    detail.status = "skipped";
    detail.reason = "clean-snapshot";
    markConfigurationRead("prefetch-skipped", {view, action, status: detail.status, reason: detail.reason});
    return null;
  }
  detail.startedAt = Date.now();
  detail.requestStatus = "pending";
  markConfigurationRead("prefetch-started", {view, action, status: "started"});
  const finishRead = measureConfigurationRead("prefetch-response", {view, action});
  const request = panel._hass.callWS({
    type: WS_TYPE,
    section: "configuration",
    action,
    entry_id: entryId,
    subentry_id: subentryId,
  });
  const prefetch = {
    view,
    entryId,
    subentryId,
    action,
    cacheGeneration: panel._cacheGeneration,
    detail,
    promise: request.then(
      (value) => { detail.requestStatus = "fulfilled"; finishRead("fulfilled"); return {status: "fulfilled", value}; },
      (reason) => { detail.requestStatus = "rejected"; finishRead("rejected"); return {status: "rejected", reason}; },
    ),
  };
  panel._eocStoredConfigPrefetch = prefetch;
  return prefetch;
}

export function discardStoredConfigurationPrefetch(panel, reason) {
  const prefetch = panel._eocStoredConfigPrefetch;
  if (!prefetch) return;
  prefetch.detail.status = "discarded";
  prefetch.detail.reason = reason;
  prefetch.detail.discardedAt = Date.now();
  markConfigurationRead("prefetch-discarded", {view: prefetch.view, action: prefetch.action, status: "discarded", reason});
  panel._eocStoredConfigPrefetch = null;
}

function storedConfigurationDiscardReason(panel, prefetch, selected) {
  if (panel._data?.is_admin === false) return "non-admin";
  if (!selected || selected.subentry_id !== prefetch.subentryId || selected.entry_id !== prefetch.entryId) return "stored-agent-mismatch";
  if (panel._viewKey?.() !== prefetch.view) return "route-changed";
  if (panel._cacheGeneration !== prefetch.cacheGeneration) return "cache-generation-changed";
  if (panel._configDirty) return "dirty-configuration";
  return null;
}

export function consumeStoredConfigurationPrefetch(panel, action) {
  const prefetch = panel._eocStoredConfigPrefetch;
  if (!prefetch) return null;
  const reason = storedConfigurationDiscardReason(panel, prefetch, panel._selectedAgent?.())
    || (prefetch.action !== action ? "configuration-action-changed" : null);
  if (reason) { discardStoredConfigurationPrefetch(panel, reason); return null; }
  panel._eocStoredConfigPrefetch = null;
  prefetch.detail.status = "consumed";
  prefetch.detail.consumedAt = Date.now();
  markConfigurationRead("prefetch-consumed", {view: prefetch.view, action, status: "consumed"});
  return prefetch.promise;
}

function applyPrefetchedConfiguration(panel, prefetch, configData) {
  panel._configData = configData;
  panel._rememberCleanConfiguration?.(configData, prefetch.subentryId);
  panel._draft = JSON.parse(JSON.stringify(configData.config));
  panel._draftTitle = configData.title;
  panel._draftAgentId = prefetch.subentryId;
  panel._setConfigDirty?.(false);
}

export async function loadAgentsWithOverviewPrefetch(panel, selectedId = null) {
  const initialToken = panel._loadToken;
  const previousAgentId = panel._agentId;
  const saved = globalThis.localStorage?.getItem?.(AGENT_KEY);
  const preferred = selectedId || saved;
  if (panel._viewKey?.() === "overview") panel._markColdLifecycle?.("overview-asset-start");
  const routeAsset = warmRouteAsset(panel._viewKey?.());
  if (panel._viewKey?.() === "overview" && routeAsset?.then) {
    routeAsset.then(
      () => panel._markColdLifecycle?.("overview-asset-complete", {status: "fulfilled"}),
      () => panel._markColdLifecycle?.("overview-asset-complete", {status: "rejected"}),
    );
  }
  const prefetch = startStoredOverviewPrefetch(panel, preferred, routeAsset);
  const configurationPrefetch = startStoredConfigurationPrefetch(panel, preferred);

  panel._data = await panel._hass.callWS({type: WS_TYPE, action: "agents"});
  panel._baseScopes = panel._data.scopes || [];
  const agents = panel._data.agents || [];
  panel._agentId = agents.some((item) => item.subentry_id === preferred)
    ? preferred
    : agents[0]?.subentry_id;

  const selected = panel._selectedAgent?.();
  if (panel._agentId) globalThis.localStorage?.setItem?.(AGENT_KEY, panel._agentId);
  if (selected?.entry_id) globalThis.localStorage?.setItem?.(ENTRY_KEY, selected.entry_id);
  if (previousAgentId !== panel._agentId) panel._scopeId = null;
  panel._applyScopes(panel._scopeCatalogCache.get(panel._scopeCatalogKey()) || panel._baseScopes);

  const overviewSelected = panel._viewKey?.() === "overview" && Boolean(selected);
  if (overviewSelected) {
    // The agent catalogue is enough for a useful first Overview. Do not hide it
    // behind the full Overview module or storage-backed summary.
    panel._contentData = null;
    panel._result = null;
    panel._error = null;
    panel._busy = false;
    panel._render();
  }

  if (configurationPrefetch) {
    const reason = storedConfigurationDiscardReason(panel, configurationPrefetch, selected);
    if (reason) discardStoredConfigurationPrefetch(panel, reason);
    else if (configurationPrefetch.view !== "data-memory/conversations") {
      // History's list can paint before configuration completes. Its settings
      // loader consumes this same in-flight request after the route starts.
      const pending = consumeStoredConfigurationPrefetch(panel, configurationPrefetch.action);
      const settled = await pending;
      const staleReason = storedConfigurationDiscardReason(panel, configurationPrefetch, panel._selectedAgent?.())
        || (panel._loadToken !== initialToken ? "load-token-changed" : null);
      if (staleReason) {
        configurationPrefetch.detail.status = "discarded";
        configurationPrefetch.detail.reason = staleReason;
        configurationPrefetch.detail.discardedAt = Date.now();
      } else if (settled.status === "fulfilled" && settled.value?.config && typeof settled.value.config === "object") {
        applyPrefetchedConfiguration(panel, configurationPrefetch, settled.value);
      } else {
        configurationPrefetch.detail.status = "discarded";
        configurationPrefetch.detail.reason = settled.status === "rejected" ? "request-failed" : "invalid-response";
        configurationPrefetch.detail.discardedAt = Date.now();
      }
    }
  }

  if (
    prefetch
    && selected?.subentry_id === prefetch.subentryId
    && selected?.entry_id === prefetch.entryId
    && overviewSelected
  ) {
    const overviewResult = await prefetch.overviewPrimary.then(
      (value) => ({status: "fulfilled", value}),
      (reason) => ({status: "rejected", reason}),
    );
    if (panel._viewKey?.() !== "overview" || panel._loadToken !== initialToken
        || panel._agentId !== prefetch.subentryId) return;
    if (overviewResult.status === "fulfilled") {
      applyOverviewResult(panel, overviewResult.value, {cache: false});
      void startOverviewDetailReads(panel, {
        loadToken: initialToken,
        cacheGeneration: panel._cacheGeneration,
      });
      return;
    }
    // A speculative stored-ID request can fail after an agent was recreated.
    // Keep the useful snapshot visible while the authoritative route load retries.
    await panel._loadSection(true);
    return;
  }

  await panel._loadSection(overviewSelected);
}


export function isRestrictedManagementView(page, subsection = null) {
  if (page === "data-memory" && subsection === "knowledge") return true;
  if (page === "usage-maintenance" && subsection === null) return true;
  return page === "usage-maintenance" && ["usage", "diagnostics"].includes(subsection);
}

export function nonAdminOverviewKnowledgeSnapshot(panel) {
  const agent = panel?._selectedAgent?.();
  return {
    sources: [],
    stats: {source_count: Number(agent?.knowledge_source_count || 0)},
  };
}
