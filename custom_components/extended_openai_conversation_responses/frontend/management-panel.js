import {bindConfigurationClarity, enhanceConfigurationClarity} from "./management-draft-navigation.js";
import {formatManagementTimestamp, prepareMemoryBrowser, ensureTemporaryScope, storeRuntimeGuidance} from "./management-data-state.js";
import {enhanceConfirmationScope} from "./management-confirmation-scope.js";
import {configurationDestinations} from "./management-config-destinations.js";
import {
  bindPageDrafts,
  initializePageDraft,
  refreshPageSaveBar,
  savePageChanges,
} from "./management-page-drafts.js";
import {readSectionCache, writeSectionCache, pruneCacheTimes, SCOPE_CACHE_TTL_MS, CLEAN_CONFIG_TTL_MS} from "./management-cache.js";
import {bindPanelDialogs, knowledgeSourceAvailabilityControl, updateDialogs} from "./management-dialogs.js";
import {ASSISTANT_INTRO_MARKUP, renderManagement, showPendingDestination, reconcileScopePicker, reconcileHistoryConfiguration} from "./management-renderer.js";
import {bindSingleRequestSave, bindFrontendCorrectness, normalizeGuestModeTimestamp, setControlPending, isAgentMutation, syncAgentPicker} from "./management-actions.js";
import {loadAgentsWithOverviewPrefetch, loadRoute, bindRequestRuleSearch, applyRequestRuleSearch, warmRouteAsset, prefetchIntentRead, consumeIntentRead, consumeStoredConfigurationPrefetch, discardStoredConfigurationPrefetch, markConfigurationRead, measureConfigurationRead} from "./management-route.js";
import {getConfigurationEditor, getConfigurationTools, getRouteFeature, routeAssetKind, routeFeaturesReady, isRestrictedManagementView, nonAdminOverviewKnowledgeSnapshot} from "./management-route.js";
import {NAVIGATION, pageMetadata, routeFromPath, routePath} from "./frontend-navigation.js";
import {clone, same} from "./unsaved-state.js";
import {bindGuide, renderGuide} from "./guide-page.js";
import {bindOverview, renderOverview, enhanceOverviewHealthClarity} from "./overview-page.js";
import {formatUsageNumber} from "./usage-format.js";
import {
  bindStateSafety,
  cleanupStateSafety,
  confirmDialogClose,
  confirmStateSafeNavigation,
  openDialogBaseline,
  rebuildConfigDirtyKeys,
} from "./management-state-safety.js";

const WS_TYPE = "extended_openai_conversation_responses/management";
const TOOL_MUTATIONS = new Set(["save", "set_enabled", "delete", "save_group", "delete_group", "ha_add"]);
const REQUEST_RULE_CACHE_KEY = "capabilities/request-rules";

const KNOWLEDGE_TITLE_LIMIT = 120;
const KNOWLEDGE_DESCRIPTION_LIMIT = 500;
const KNOWLEDGE_LIMIT = 100000;
const NAVIGATION_MARK_PREFIX = "extended-openai:navigation";
const LOAD_MARK_PREFIX = "extended-openai:load-section";
const RENDER_MARK_PREFIX = "extended-openai:render";
const MAX_MEASURE_ENTRIES = 100;
const COLD_MARK_PREFIX = "extended-openai:cold";
// Performance entries are global to the document, not to a panel instance.
let performanceSequence = 0;
let functionMutationSequence = 0;
const functionMutationResults = new WeakMap();
let navigationSearchModule = null;
let navigationSearchPromise = null;

function settingsSearchShellMarkup(panel) {
  const query = panel._settingsSearchQuery || "";
  return `<div class="global-search eoc-global-search"><label><span class="search-label">Find a setting</span><input id="settings-search" type="search" value="${panel._e(query)}" placeholder="Search settings by name or purpose" aria-label="Search all settings" autocomplete="off"></label><div class="search-results" role="listbox" aria-label="Settings search results" ${query ? "" : "hidden"}></div></div>`;
}

function ensureNavigationSearchModule(panel) {
  if (navigationSearchModule) {
    navigationSearchModule.enhanceNavigationSearch(panel);
    return Promise.resolve(navigationSearchModule);
  }
  if (!navigationSearchPromise) {
    navigationSearchPromise = import("./management-navigation-search.js")
      .then((module) => {
        navigationSearchModule = module;
        return module;
      })
      .finally(() => { navigationSearchPromise = null; });
  }
  return navigationSearchPromise.then((module) => {
    module.enhanceNavigationSearch(panel);
    return module;
  });
}

try {
  globalThis.performance?.mark?.(`${COLD_MARK_PREFIX}:module-evaluated`);
} catch (_err) {
  // Cold-start instrumentation must never affect panel startup.
}
const MANAGEMENT_STYLESHEET_URL = new URL("./management.css", import.meta.url).href;
// Keep this deliberately small and geometry-identical to management.css: it exists
// only to prevent FOUC/layout shift while the external stylesheet is still pending.
const CRITICAL_STYLE = `
  :host{display:block;min-height:100%;padding:28px;color:var(--primary-text-color);font-family:var(--paper-font-body1_-_font-family,system-ui);font-size:14px;line-height:1.45;box-sizing:border-box;background:color-mix(in srgb,var(--secondary-background-color) 42%,var(--primary-background-color))}
  *{box-sizing:border-box}
  [hidden]{display:none!important}
  .page-shell{max-width:1380px;margin:auto}
  header{display:flex;justify-content:space-between;gap:36px;align-items:end;margin-bottom:28px}
  .page-heading h1{margin:0;font-size:30px;font-weight:500}
  .page-heading p{margin:6px 0 0;color:var(--secondary-text-color);line-height:1.5}
  header .global-search.eoc-global-search{width:min(380px,100%);max-width:100%;min-width:280px;margin:0;align-self:end;position:relative}
  header .eoc-global-search>label{display:block}
  header .eoc-global-search .search-label{display:none}
  label{display:grid;gap:7px;font-size:14px;color:var(--secondary-text-color)}
  input,select,textarea,button{font:inherit}
  input,select,textarea{width:100%;min-height:42px;color:var(--primary-text-color);background:var(--card-background-color);border:1px solid var(--divider-color);border-radius:9px;padding:10px 12px}
  button{min-height:42px;border:0;border-radius:9px;padding:9px 16px;cursor:pointer;background:var(--primary-color);color:var(--text-primary-color)}
  .mobile-nav{display:none}
  .eoc-agent-context-row{display:flex;align-items:end;gap:12px;margin:0 0 14px}
  .eoc-agent-context-row .agent-picker{width:min(390px,100%);min-width:0;margin:0}
  .eoc-agent-context-row .agent-picker.eoc-agent-context{min-width:0;padding:0;border:0;border-radius:0;background:transparent;box-shadow:none}
  .eoc-agent-actions{display:grid;justify-items:end;gap:5px;margin-left:auto}
  nav{display:flex;overflow:auto;border-bottom:1px solid var(--divider-color);margin-bottom:28px}
  .top-nav{overflow:visible}
  nav button{background:transparent;color:var(--secondary-text-color);border-radius:0;padding:13px 18px;white-space:nowrap}
  nav button.active{color:var(--primary-color);border-bottom:3px solid var(--primary-color)}
  .subsection-nav{display:flex;gap:8px;flex-wrap:wrap;overflow:visible;border:0;margin:0 0 12px;padding:0}
  .subsection-nav button{min-height:38px;padding:8px 13px;border:1px solid var(--divider-color);border-radius:999px;background:var(--card-background-color);color:var(--secondary-text-color)}
  .subsection-nav button.active{border-color:var(--primary-color);background:color-mix(in srgb,var(--primary-color) 10%,var(--card-background-color));color:var(--primary-color)}
  .section-layout{display:block}
  .section-selector{margin:0 0 20px}
  main{display:grid;gap:30px}
  .page-intro{display:grid;gap:7px;max-width:780px}
  .page-intro h1,.page-intro p{margin:0}
  .page-intro h1{font-size:24px;font-weight:600;line-height:1.25}
  .page-intro p{color:var(--secondary-text-color);line-height:1.5}
  .dashboard-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
  .dashboard-card{display:flex;justify-content:space-between;align-items:end;gap:18px;padding:22px;background:var(--card-background-color);border:1px solid var(--divider-color);border-radius:13px}
  .dashboard-card h2,.dashboard-card p{margin:0}
  .dashboard-card strong{display:block;margin-top:12px;font-size:18px}
  .dashboard-card p{margin-top:7px;color:var(--secondary-text-color);line-height:1.45}
  @media (min-width:680px) and (max-width:1100px){.dashboard-grid{grid-template-columns:1fr}}
  .loading{display:flex;align-items:center;justify-content:center;gap:10px;min-height:130px;color:var(--secondary-text-color)}
  .spinner{width:20px;height:20px;border:2px solid var(--divider-color);border-top-color:var(--primary-color);border-radius:50%}
  @media (min-width:801px){.section-selector{display:none}}
  @media (max-width:800px){
    header{flex-direction:column;align-items:stretch;gap:18px}
    header .global-search.eoc-global-search{width:100%;min-width:0;align-self:stretch}
    .eoc-agent-context-row{display:grid;grid-template-columns:1fr;align-items:stretch;gap:10px;margin-bottom:18px}
    .eoc-agent-context-row .agent-picker{width:100%}
    .eoc-agent-actions{width:100%;margin-left:0;justify-items:stretch}
    .subsection-nav{display:none}
  }
  @media (max-width:760px){
    .dashboard-grid{grid-template-columns:1fr}
    .dashboard-card{align-items:stretch;flex-direction:column}
    .dashboard-card button{width:100%}
    .top-nav{display:none}
    .mobile-nav{display:grid;margin-bottom:18px}
  }
`;


function performanceApi() {
  const api = globalThis.performance;
  return api && typeof api.mark === "function" && typeof api.measure === "function" ? api : null;
}

function performanceNow() {
  const api = globalThis.performance;
  return typeof api?.now === "function" ? api.now() : Date.now();
}

function startMeasure(panel, prefix) {
  const api = performanceApi();
  if (!api) return null;
  const id = `${prefix}:${++performanceSequence}`;
  const start = `${id}:start`;
  try {
    api.mark(start);
    return {api, id, start, panel};
  } catch (_err) {
    return null;
  }
}

function finishMeasure(measure, detail = null) {
  if (!measure) return;
  const {api, id, start, panel} = measure;
  const end = `${id}:end`;
  try {
    api.mark(end);
    try {
      api.measure(id, {start, end, detail});
    } catch (_err) {
      api.measure(id, start, end);
    }
    panel._eocPerformanceMeasureIds ||= [];
    panel._eocPerformanceMeasureIds.push(id);
    while (panel._eocPerformanceMeasureIds.length > MAX_MEASURE_ENTRIES) {
      api.clearMeasures?.(panel._eocPerformanceMeasureIds.shift());
    }
  } catch (_err) {
    // Optional profiling must never turn a successful load into a UI error or
    // replace the original failure, even if the host cleared an active mark.
  } finally {
    for (const name of [start, end]) {
      try { api.clearMarks?.(name); } catch (_err) { /* Best-effort cleanup. */ }
    }
  }
}

function trackAsync(panel, prefix, operation, navigation = false) {
  const view = panel._viewKey?.() || null;
  const measure = startMeasure(panel, prefix);
  if (navigation) panel._eocNavigationDepth = (panel._eocNavigationDepth || 0) + 1;
  const finish = (status) => {
    if (navigation) panel._eocNavigationDepth = Math.max(0, (panel._eocNavigationDepth || 1) - 1);
    finishMeasure(measure, {view, status});
  };
  let result;
  try {
    result = operation();
  } catch (err) {
    finish("threw");
    throw err;
  }
  if (!result || typeof result.finally !== "function") {
    finish("sync");
    return result;
  }
  return result.finally(() => finish("settled"));
}


function settledSectionResult(entries, settled) {
  const result = {};
  const load_errors = [];
  entries.forEach(([key, label], index) => {
    const item = settled[index];
    if (item.status === "fulfilled") result[key] = item.value;
    else load_errors.push({key, label, message: item.reason?.message || String(item.reason || "Unknown error")});
  });
  return {...result, load_errors};
}

export class ExtendedOpenAIManagementPanel extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    // The URL is known at construction. Start its request before agent data and
    // route code settle, and keep the same link connected through shell renders.
    if (typeof document !== "undefined") {
      const criticalStyle = document.createElement("style");
      criticalStyle.dataset.eocCriticalStyles = "";
      criticalStyle.textContent = CRITICAL_STYLE;
      const stylesheet = document.createElement("link");
      stylesheet.rel = "stylesheet";
      stylesheet.href = MANAGEMENT_STYLESHEET_URL;
      stylesheet.dataset.eocPersistentStyles = "";
      this.shadowRoot.append(criticalStyle, stylesheet);
    }
    this._eocColdLifecycleMarks = new Set();
    this._markColdLifecycle("constructed");
    const route = routeFromPath(window.location.pathname);
    this._page = route.page;
    this._subsection = route.section;
    this._data = null;
    this._result = null;
    this._busy = false;
    this._query = "";
    this._memoryKind = "persistent";
    this._showEmptyScopes = false;
    this._confirmResolver = null;
    this._configDirty = false;
    this._configData = null;
    this._draft = null;
    this._cleanConfigSnapshots = new Map();
    this._draftTitle = null;
    this._draftAgentId = null;
    this._sectionCache = new Map();
    this._eocSectionCacheTimes = new Map();
    this._scopeCatalogCache = new Map();
    this._scopeCatalogVisitKey = null;
    this._baseScopes = [];
    this._serviceCatalog = null;
    this._serviceCatalogPromise = null;
    this._loadToken = 0;
    this._cacheGeneration = 0;
    this._eocScopeCatalogTimes = new Map();
    this._eocInPlaceRequestRuleSearch = true;
    this._configSearchQuery = "";
    this._settingsSearchQuery = "";
    this._guideQuery = "";

  }

  _markColdLifecycle(name, detail = null) {
    const api = performanceApi();
    if (!api || !name || this._eocColdLifecycleMarks?.has(name)) return false;
    this._eocColdLifecycleMarks ||= new Set();
    this._eocColdLifecycleMarks.add(name);
    try {
      const mark = `${COLD_MARK_PREFIX}:${name}`;
      if (detail == null) api.mark(mark);
      else api.mark(mark, {detail});
      return true;
    } catch (_err) {
      return false;
    }
  }

  set hass(value) {
    const first = !this._hass;
    this._hass = value;
    if (first) this._loadAgents();
  }

  set route(value) {
    this._route = value;
    const route = routeFromPath(window.location.pathname);
    if (route.page !== this._page || route.section !== this._subsection) this._handleRouteChange(route);
    else if (!this.shadowRoot.querySelector("[data-eoc-persistent-shell]")) this._render();
  }

  connectedCallback() {
    this._markColdLifecycle("connected");
    // Home Assistant can assign properties before custom-element upgrade. Replay
    // own properties so the class setters receive the values after definition.
    for (const name of ["hass", "route"]) {
      if (!Object.prototype.hasOwnProperty.call(this, name)) continue;
      const value = this[name];
      delete this[name];
      this[name] = value;
    }
    bindStateSafety(this);
    bindPanelDialogs(this);
    bindSingleRequestSave(this);
    bindFrontendCorrectness(this);
    bindPageDrafts(this);
    bindConfigurationClarity(this);
    this._bindSettingsSearchLazyLoad();
    this._bindRouteAssetWarmup();
    if (this._viewKey() === "usage-maintenance/diagnostics") {
      getRouteFeature("usage-maintenance/diagnostics")?.enhanceDiagnostics(this);
    }
  }

  disconnectedCallback() {
    getRouteFeature("usage-maintenance/diagnostics")?.stopDiagnosticsWatch(this);
    cleanupStateSafety(this);
  }

  _warmNavigationTarget(target) {
    const page = target?.dataset?.page || this._page;
    const subsection = target?.dataset?.subsection
      || (target?.dataset?.page ? this._visibleSubsections(page)[0]?.id || null : null);
    const view = this._viewKey(page, subsection);
    warmRouteAsset(view);
    void this._loadConfigurationLiveMetadata(view);
  }

  _currentRouteUsableForSpeculation() {
    return !!this._data && !this._busy && !this._error
      && this._eocRenderedRoute === `${this._agentId}|${this._viewKey()}`
      && this._eocRenderedFeatureReady
      && !this.shadowRoot.querySelector("[data-eoc-main][aria-busy='true']");
  }

  _bindSettingsSearchLazyLoad() {
    const root = this.shadowRoot;
    if (!root || root.__eocSettingsSearchLazyBound) return;
    root.__eocSettingsSearchLazyBound = true;
    const load = (event) => {
      const input = event.target?.closest?.("#settings-search");
      if (!input) return;
      if (event.type === "input") {
        this._settingsSearchQuery = input.value;
        if (navigationSearchModule || navigationSearchPromise) return;
      }
      void ensureNavigationSearchModule(this);
    };
    root.addEventListener("focusin", load, true);
    root.addEventListener("input", load, true);
  }

  _bindRouteAssetWarmup() {
    const root = this.shadowRoot;
    if (!root || root.__eocRouteAssetWarmupBound) return;
    root.__eocRouteAssetWarmupBound = true;
    let hoverTimer = null;
    let hoverTarget = null;
    const cancelHover = () => {
      if (hoverTimer !== null) clearTimeout(hoverTimer);
      hoverTimer = null;
      hoverTarget = null;
    };
    root.addEventListener("pointerover", (event) => {
      const target = event.target?.closest?.("[data-page],[data-subsection]");
      if (!target || target === hoverTarget) return;
      cancelHover();
      hoverTarget = target;
      hoverTimer = setTimeout(() => {
        hoverTimer = null;
        if (hoverTarget === target && this._currentRouteUsableForSpeculation()) this._warmNavigationTarget(target);
      }, 100);
    });
    root.addEventListener("pointerout", (event) => {
      const target = event.target?.closest?.("[data-page],[data-subsection]");
      if (target && target === hoverTarget && !target.contains(event.relatedTarget)) cancelHover();
    });
    root.addEventListener("focusin", (event) => {
      const target = event.target?.closest?.("[data-page],[data-subsection]");
      if (target) this._warmNavigationTarget(target);
    });
    root.addEventListener("pointerdown", (event) => {
      const target = event.target?.closest?.("[data-page],[data-subsection]");
      if (target) {
        cancelHover();
        this._warmNavigationTarget(target);
        prefetchIntentRead(this, this._navigationTargetView(target));
      }
    });
    root.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      const target = event.target?.closest?.("[data-page],[data-subsection]");
      if (target) prefetchIntentRead(this, this._navigationTargetView(target));
    });
  }

  _navigationTargetView(target) {
    const page = target?.dataset?.page || this._page;
    const subsection = target?.dataset?.subsection
      || (target?.dataset?.page ? this._visibleSubsections(page)[0]?.id || null : null);
    return this._viewKey(page, subsection);
  }

  _setConfigDirty(value) {
    if (!value) {
      this._eocDirtyConfigKeys = new Set();
      this._configDirty = false;
      return;
    }
    this._configDirty = true;
    if (this._eocDirtyConfigKeys instanceof Set) {
      queueMicrotask(() => {
        if (!(this._eocDirtyConfigKeys instanceof Set) || this._eocDirtyConfigKeys.size) return;
        const changed = rebuildConfigDirtyKeys(this);
        const dirty = changed.size > 0;
        const wasDirty = Boolean(this._configDirty);
        this._configDirty = dirty;
        if (wasDirty && !dirty) this._render();
      });
    }
  }

  _clearConfigDraft() {
    this._setConfigDirty(false);
    this._configData = null;
    this._configDataStale = false;
    this._draft = null;
    this._draftTitle = null;
    this._draftAgentId = null;
    this._configSearchQuery = "";
    this._settingsSearchConfig = null;
    this._settingsSearchConfigAgentId = null;
    this._settingsSearchConfigError = null;
    this._settingsSearchConfigErrorAgentId = null;
  }

  _configurationSnapshotKey(agentId = this._agentId, projection = "full") {
    const agent = this._data?.agents?.find((item) => item.subentry_id === agentId);
    return agent ? `${agent.entry_id}|${agentId}|${projection}` : null;
  }

  _rememberCleanConfiguration(configData, agentId = this._agentId) {
    if (!configData?.config || configData.revision == null) return;
    const projection = configData.projection === "retention" ? "retention" : "full";
    const key = this._configurationSnapshotKey(agentId, projection);
    if (key) this._cleanConfigSnapshots.set(key, {result: configData, loadedAt: Date.now(), generation: this._cacheGeneration});
  }

  _freshCleanConfiguration(agentId, projection) {
    const key = this._configurationSnapshotKey(agentId, projection);
    const cached = key ? this._cleanConfigSnapshots.get(key) : null;
    const age = cached ? Date.now() - cached.loadedAt : -1;
    return cached?.result?.config && cached.result.revision != null
      && (cached.result.projection === "retention" ? "retention" : "full") === projection
      && cached.generation === this._cacheGeneration && age >= 0 && age <= CLEAN_CONFIG_TTL_MS
      ? cached.result : null;
  }

  _hydrateCleanConfiguration(view) {
    if (this._configDirty || !this._selectedAgent() || !this._isDraftView()
        || ["data-memory/conversations", "capabilities/request-rules", "usage-maintenance/backup-restore"].includes(view)) return false;
    const projection = view === "usage-maintenance/retention" ? "retention" : "full";
    const configData = this._freshCleanConfiguration(this._agentId, projection);
    if (!configData) return false;
    this._configData = configData;
    this._configDataStale = false;
    this._draft = JSON.parse(JSON.stringify(configData.config));
    this._draftTitle = configData.title;
    this._draftAgentId = this._agentId;
    this._setConfigDirty(false);
    this._contentData = null;
    this._result = configData;
    this._error = null;
    this._busy = false;
    this._applyConfigurationLiveMetadata(view);
    return true;
  }

  _invalidateCleanConfiguration(agentId) {
    if (!agentId) return;
    if (agentId === this._draftAgentId) this._configDataStale = true;
    for (const projection of ["full", "retention"]) {
      const key = this._configurationSnapshotKey(agentId, projection);
      if (key) this._cleanConfigSnapshots.delete(key);
    }
    this._eocLiveMetadataEpoch = (this._eocLiveMetadataEpoch || 0) + 1;
    this._eocLiveMetadataCache?.clear();
    this._eocLiveMetadataPending?.clear();
  }

  _configurationDirtyDestinations() {
    return configurationDestinations(this);
  }

  _syncConfigDirty() {
    const changed = rebuildConfigDirtyKeys(this);
    this._configDirty = changed.size > 0;
    return this._configDirty;
  }

  _captureDialogBaseline(dialog) {
    openDialogBaseline(this, dialog);
  }

  _confirmEditorClose(dialog) {
    return confirmDialogClose(this, dialog);
  }

  _confirmUnsavedNavigation(destination) {
    return confirmStateSafeNavigation(this, destination);
  }

  async _handleRouteChange(route) {
    const destination = route.section ? `${route.page}/${route.section}` : route.page;
    if (!await confirmStateSafeNavigation(this, destination)) {
      history.pushState({}, "", routePath(this._page, this._subsection));
      return;
    }
    this._page = route.page;
    this._subsection = route.section;
    this._query = "";
    this._result = null;
    if (this._hydrateCleanConfiguration(this._viewKey())) this._render();
    else showPendingDestination(this);
    await this._loadSection();
  }

  _viewKey(page = this._page, subsection = this._subsection) {
    return subsection ? `${page}/${subsection}` : page;
  }

  _isDraftView(page = this._page, subsection = this._subsection) {
    if ((page === "data-memory" && subsection === "memory-settings") || (page === "capabilities" && subsection === "web-skills")) return this._data?.is_admin !== false;
    if (this._data && !this._data.is_admin) return false;
    return page === "assistant" ||
      (page === "capabilities" && ["home-assistant", "request-rules", "functions"].includes(subsection)) ||
      (page === "data-memory" && subsection === "conversations") ||
      (page === "usage-maintenance" && ["backup-restore", "retention"].includes(subsection));
  }

  _configSectionsForView() {
    return {
      "capabilities/home-assistant": ["local"],
      "capabilities/web-skills": ["capabilities"],
      "assistant/basics": ["general"],
      "assistant/model-responses": ["model"],
      "assistant/conversation": ["conversation", "context"],
      "assistant/prompt-context": ["prompt"],
      "assistant/voice": ["voice"],
      "assistant/speech": ["speech"],
      "assistant/advanced": ["capabilities"],
      "data-memory/conversations": ["archive"],
      "usage-maintenance/retention": ["retention"],
    }[this._viewKey()] || [];
  }

  _configurationActions() {
    const sections = this._configSectionsForView();
    if (!sections.length || this._viewKey() === "usage-maintenance/retention") return "";
    if (this._viewKey() === "data-memory/conversations" && !this._data?.is_admin) return "";
    return getConfigurationEditor()?.renderConfigurationActions?.(this, sections) || "";
  }

  _canAccessView(page, subsection = null) {
    if (page === "data-memory" && subsection === "memory-settings" && this._data?.is_admin === false) return false;
    if (this._data?.is_admin === false && isRestrictedManagementView(page, subsection)) return false;
    if (page === "usage-maintenance" && subsection === "request-debug") return this._data?.is_admin === true;
    if (this._data?.is_admin !== false) return true;
    if (page === "assistant") return false;
    if (page === "capabilities" && subsection && subsection !== "guest-mode") return false;
    if (page === "usage-maintenance" && ["backup-restore", "retention"].includes(subsection)) return false;
    return true;
  }

  _enhanceSubsectionNavigation() {
    const root = this.shadowRoot;
    if (!root) return;
    const local = this._visibleSubsections();
    const topNav = root.querySelector(".top-nav");
    let nav = root.querySelector(".subsection-nav");
    const markup = local.length > 1
      ? local.map((item) => `<button type="button" data-subsection="${this._e(item.id)}" class="${item.id === this._subsection ? "active" : ""}" ${item.id === this._subsection ? 'aria-current="page"' : ""}>${this._e(item.label)}</button>`).join("")
      : "";
    if (!nav && topNav) {
      nav = document.createElement("nav");
      nav.className = "subsection-nav";
      topNav.after(nav);
    }
    if (nav && nav._eocMarkup !== markup) {
      nav.innerHTML = markup;
      nav._eocMarkup = markup;
      this._eocNavigationRevision = (this._eocNavigationRevision || 0) + 1;
      nav.hidden = !markup;
      nav.setAttribute("aria-label", `${pageMetadata(this._page).label} sections`);
    }
  }

  _visibleSubsections(page = this._page) {
    return pageMetadata(page).sections.filter((item) => this._canAccessView(page, item.id));
  }

  async _call(section, action, extra = {}) {
    if (section === "usage" && action === "daily" && !extra.start_date && !extra.end_date) {
      const {loadUsageDaily} = await import("./usage-data.js");
      return loadUsageDaily(this, extra);
    }
    const guidanceCall = section === "configuration"
      && ["get", "validate", "update", "save"].includes(action);
    const guidanceAgentId = this._agentId;
    const guidanceRevision = guidanceCall
      ? (this._eocGuidanceCallRevision = (this._eocGuidanceCallRevision || 0) + 1)
      : null;

    let result;
    if (
      this._data?.is_admin === false
      && this._viewKey() === "overview"
      && section === "knowledge"
      && action === "list"
    ) {
      result = nonAdminOverviewKnowledgeSnapshot(this);
    } else {
      let payload = extra;
      if (
        section === "tools"
        && TOOL_MUTATIONS.has(action)
        && extra.revision === undefined
        && typeof this._configData?.revision === "string"
      ) {
        payload = {revision: this._configData.revision, ...extra};
      }
      result = await this._callWithMutationSafety(section, action, payload);
      this._applyToolRevision(section, action, result);
    }

    if (
      guidanceCall
      && guidanceRevision === this._eocGuidanceCallRevision
      && this._agentId === guidanceAgentId
    ) {
      storeRuntimeGuidance(this, result, guidanceAgentId);
    }
    return result;
  }

  _callWithMutationSafety(section, action, extra = {}) {
    if (!isAgentMutation(section, action)) return this._callCore(section, action, extra);

    this._pendingMutations ||= new Map();
    const key = JSON.stringify([section, action, extra]);
    if (this._pendingMutations.has(key)) return this._pendingMutations.get(key);

    const mutationTrace = section === "tools" ? {
      id: ++functionMutationSequence,
      section,
      action,
      queuedAt: performanceNow(),
      status: "queued",
    } : null;
    const previous = section === "tools"
      ? (this._eocFunctionMutationTail || Promise.resolve())
      : Promise.resolve();
    const pending = previous.catch(() => {}).then(
      () => {
        if (mutationTrace) {
          mutationTrace.startedAt = performanceNow();
          mutationTrace.queueWaitMs = mutationTrace.startedAt - mutationTrace.queuedAt;
          mutationTrace.status = "running";
        }
        return this._runAgentMutation(section, action, extra, mutationTrace);
      },
    );
    const tracked = pending.finally(() => {
      this._pendingMutations.delete(key);
      if (this._eocFunctionMutationTail === tracked) this._eocFunctionMutationTail = null;
      if (mutationTrace) {
        mutationTrace.tailReleasedAt = performanceNow();
        mutationTrace.totalUntilTailReleaseMs = mutationTrace.tailReleasedAt - mutationTrace.queuedAt;
        this._eocFunctionMutationDiagnostics ||= [];
        this._eocFunctionMutationDiagnostics.push(mutationTrace);
        if (this._eocFunctionMutationDiagnostics.length > 50) {
          this._eocFunctionMutationDiagnostics.shift();
        }
      }
    });
    if (section === "tools") this._eocFunctionMutationTail = tracked;
    this._pendingMutations.set(key, tracked);
    return tracked;
  }

  async _runAgentMutation(section, action, extra, mutationTrace = null) {
    this._eocAgentMutations = Number(this._eocAgentMutations || 0) + 1;
    syncAgentPicker(this);
    try {
      const result = await this._callCore(section, action, extra, mutationTrace);
      this._applyToolRevision(section, action, result);
      if (mutationTrace) mutationTrace.status = "fulfilled";
      return result;
    } catch (err) {
      if (mutationTrace) {
        mutationTrace.status = "rejected";
        mutationTrace.error = err?.message || String(err);
      }
      throw err;
    } finally {
      this._eocAgentMutations = Math.max(0, Number(this._eocAgentMutations || 1) - 1);
      syncAgentPicker(this);
    }
  }

  _applyToolRevision(section, action, result) {
    if (
      section === "tools"
      && TOOL_MUTATIONS.has(action)
      && typeof result?.revision === "string"
      && this._configData
    ) {
      this._configData = {...this._configData, revision: result.revision};
    }
  }

  _callCore(section, action, extra = {}, mutationTrace = null) {
    // Configuration remains editable even when persisted Function Tools need repair.
    const issue = this._selectedAgent()?.configuration_issue;
    if (section === "configuration" && issue?.field === "functions" && issue.repairable === true) {
      const repairAction = {validate:"configuration_validate", save:"configuration_save", update:"configuration_save"}[action];
      if (repairAction) return this._request("function_repair", repairAction, extra, mutationTrace);
    }
    let payload = extra;
    if (section === "guest_mode" && action === "update") {
      payload = {...extra};
      for (const key of ["active_from", "active_until"]) {
        if (payload[key]) payload[key] = normalizeGuestModeTimestamp(payload[key]);
      }
    }
    const ruleSave = section === "request_rules" && ["create", "update"].includes(action)
      && this.shadowRoot?.querySelector?.("#rule-dialog")?.open;
    if (!ruleSave) return this._request(section, action, payload, mutationTrace);
    if (this._eocRuleSavePromise) return this._eocRuleSavePromise;
    const button = this.shadowRoot.querySelector("#rule-save");
    setControlPending(this, button, true);
    const request = Promise.resolve().then(() => this._request(section, action, payload));
    const tracked = request.finally(() => {
      setControlPending(this, button, false);
      if (this._eocRuleSavePromise === tracked) this._eocRuleSavePromise = null;
    });
    this._eocRuleSavePromise = tracked;
    return tracked;
  }

  async _request(section, action, extra = {}, mutationTrace = null) {
    if (!this._hass) return null;
    const agent = this._selectedAgent();
    const requestTrace = {
      section,
      action,
      view: this._viewKey?.() || null,
      sentAt: performanceNow(),
      mutationId: mutationTrace?.id || null,
    };
    if (mutationTrace) mutationTrace.wsSentAt = requestTrace.sentAt;
    try {
      const result = await this._hass.callWS({
        type: WS_TYPE,
        section,
        action,
        ...(agent ? { entry_id: agent.entry_id, subentry_id: agent.subentry_id } : {}),
        ...extra,
      });
      requestTrace.responseAt = performanceNow();
      requestTrace.durationMs = requestTrace.responseAt - requestTrace.sentAt;
      requestTrace.status = "fulfilled";
      if (mutationTrace) {
        mutationTrace.wsResponseAt = requestTrace.responseAt;
        mutationTrace.wsDurationMs = requestTrace.durationMs;
        mutationTrace.backend = result?._performance || null;
        if (result && typeof result === "object") functionMutationResults.set(result, mutationTrace);
      }
      const invalidationStarted = performanceNow();
      this._invalidateAfterMutation(agent?.subentry_id, section, action);
      pruneCacheTimes(this);
      requestTrace.invalidationMs = performanceNow() - invalidationStarted;
      if (mutationTrace) mutationTrace.invalidationMs = requestTrace.invalidationMs;
      return result;
    } catch (err) {
      requestTrace.responseAt = performanceNow();
      requestTrace.durationMs = requestTrace.responseAt - requestTrace.sentAt;
      requestTrace.status = "rejected";
      if (mutationTrace) {
        mutationTrace.wsResponseAt = requestTrace.responseAt;
        mutationTrace.wsDurationMs = requestTrace.durationMs;
      }
      throw err;
    } finally {
      this._eocRequestDiagnostics ||= [];
      this._eocRequestDiagnostics.push(requestTrace);
      if (this._eocRequestDiagnostics.length > 100) this._eocRequestDiagnostics.shift();
    }
  }

  _recordFunctionMutationUi(result, detail) {
    if (!result || typeof result !== "object") return;
    const trace = functionMutationResults.get(result);
    if (!trace) return;
    trace.ui = {...detail, completedAt: performanceNow()};
  }

  _invalidateAfterMutation(agentId, section, action) {
    if (agentId && ((section === "configuration" && ["save", "update", "import"].includes(action))
        || (section === "function_repair" && action === "configuration_save")
        || (section === "tools" && TOOL_MUTATIONS.has(action))
        || (section === "knowledge" && action === "set_enabled")
        || (section === "guest_mode" && action === "save_policy")
        || (section === "settings" && action === "update")
        || (section === "backup" && action === "restore"))) {
      this._invalidateCleanConfiguration(agentId);
    }
    if (agentId && section === "usage" && action === "clear_details") {
      const key = `${agentId}|usage-maintenance/usage`;
      this._sectionCache.delete(key);
      this._eocSectionCacheTimes.delete(key);
    }
    if (agentId && isAgentMutation(section, action)) {
      this._cacheGeneration += 1;
      this._sectionCache.delete(`${agentId}|overview`);
      this._eocSectionCacheTimes.delete(`${agentId}|overview`);
      this._eocPendingRouteReads?.clear();
    }
    if (agentId && section === "backup" && action === "restore") {
      this._cacheGeneration += 1;
      const prefix = `${agentId}|`;
      for (const key of this._sectionCache.keys()) if (key.startsWith(prefix)) this._sectionCache.delete(key);
      for (const key of this._scopeCatalogCache.keys()) if (key.startsWith(prefix)) this._scopeCatalogCache.delete(key);
      if (this._scopeCatalogVisitKey?.startsWith(prefix)) this._scopeCatalogVisitKey = null;
    } else {
      const mutations = {
        request_rules: new Set(["defaults", "wording_groups", "create", "update", "delete", "duplicate"]),
        knowledge: new Set(["create", "update", "delete", "set_enabled"]),
        memories: new Set(["add", "update", "delete", "clear", "temporary_update", "temporary_delete", "temporary_clear", "reassign_legacy"]),
      };
      if (agentId && mutations[section]?.has(action)) {
        this._cacheGeneration += 1;
        const prefix = `${agentId}|`;
        const view = {request_rules:"capabilities/request-rules", knowledge:"data-memory/knowledge"}[section];
        if (view) this._sectionCache.delete(`${prefix}${view}`);
        if (["memories", "conversations"].includes(section)) {
          for (const key of this._scopeCatalogCache.keys()) {
            if (key.startsWith(prefix)) this._scopeCatalogCache.delete(key);
          }
          if (this._scopeCatalogVisitKey?.startsWith(prefix)) this._scopeCatalogVisitKey = null;
        }
      }
    }

    const affectsRequestRules = agentId && (
      (section === "tools" && TOOL_MUTATIONS.has(action))
      || (section === "request_rules" && action === "move")
    );
    if (affectsRequestRules) {
      const key = `${agentId}|${REQUEST_RULE_CACHE_KEY}`;
      this._sectionCache?.delete(key);
      this._eocSectionCacheTimes?.delete(key);
    }
  }

  _sectionCacheKey(view = this._viewKey()) {
    const agentId = this._agentId;
    if (!agentId) return null;
    if (["overview", "capabilities/request-rules", "data-memory/knowledge", "usage-maintenance/usage"].includes(view)) return `${agentId}|${view}`;
    return null;
  }

  _scopeCatalogKind(view = this._viewKey()) {
    if (view === "data-memory/conversations") return "archive";
    if (view === "data-memory/memories") return this._memoryKind === "temporary" ? "temporary" : "memory";
    return null;
  }

  _scopeCatalogKey(view = this._viewKey(), agentId = this._agentId) {
    const kind = this._scopeCatalogKind(view);
    if (!agentId || !kind) return null;
    return `${agentId}|scopes|${kind}`;
  }

  _prepareScopeCatalogVisit(view) {
    const key = this._scopeCatalogKey(view);
    this._scopeCatalogVisitKey = key;
    return key;
  }

  _applyScopes(scopes) {
    this._data.scopes = scopes || [];
    const current = this._data.scopes.find((scope) => scope.is_current_user);
    if (!this._data.scopes.some((scope) => scope.scope_id === this._scopeId)) {
      this._scopeId = current?.scope_id || this._data.scopes[0]?.scope_id;
    }
  }

  async _loadServiceCatalog() {
    if (this._serviceCatalog) return this._serviceCatalog;
    if (!this._serviceCatalogPromise) {
      this._serviceCatalogPromise = this._call("service_catalog", "get")
        .then((response) => {
          this._serviceCatalog = response?.services || {};
          return this._serviceCatalog;
        })
        .finally(() => { this._serviceCatalogPromise = null; });
    }
    return this._serviceCatalogPromise;
  }

  async _loadAgents(selectedId = null) {
    this._markColdLifecycle("agents-start");
    try {
      await loadAgentsWithOverviewPrefetch(this, selectedId);
      this._markColdLifecycle("agents-complete", {status: "fulfilled"});
    } catch (err) {
      this._markColdLifecycle("agents-complete", {status: "rejected"});
      this._error = err.message || String(err);
      this._render();
    }
  }

  async _loadScopes(scopeCatalogKey, scopeKind = this._scopeCatalogKind()) {
    if (!this._selectedAgent() || !scopeCatalogKey || !scopeKind) return;
    const agentId = this._agentId;
    const loadedAt = this._eocScopeCatalogTimes.get(scopeCatalogKey);
    if (!loadedAt || Date.now() - loadedAt > SCOPE_CACHE_TTL_MS) {
      this._scopeCatalogCache.delete(scopeCatalogKey);
      this._eocScopeCatalogTimes.delete(scopeCatalogKey);
    }
    if (this._scopeCatalogCache.has(scopeCatalogKey)) {
      this._applyScopes(this._scopeCatalogCache.get(scopeCatalogKey));
      return;
    }
    const generation = this._cacheGeneration;
    const loadToken = this._loadToken;
    const response = await this._call("scopes", "catalog", {scope_kind: scopeKind});
    const scopes = response.scopes || [];
    if (scopeCatalogKey !== this._scopeCatalogVisitKey || agentId !== this._agentId
        || generation !== this._cacheGeneration || loadToken !== this._loadToken) return;
    this._scopeCatalogCache.set(scopeCatalogKey, scopes);
    this._eocScopeCatalogTimes.set(scopeCatalogKey, Date.now());
    this._applyScopes(scopes);
  }

  _selectedAgent() {
    return this._data?.agents?.find((item) => item.subentry_id === this._agentId);
  }

  async _loadSection(silent = false) {
    // Same-scope Memory refreshes keep their read-only collection on screen.
    // Scope/agent/kind changes still take the normal loading/replacement path.
    if (this._viewKey() === "data-memory/memories" && getRouteFeature("data-memory/memories")?.hasMemoryCollection(this)) silent = true;
    if (this._viewKey() === "data-memory/memories" && this._memoryKind === "temporary") ensureTemporaryScope(this);
    prepareMemoryBrowser(this);
    const value = await trackAsync(this, LOAD_MARK_PREFIX, () => loadRoute(this, silent));
    await getRouteFeature("memory-browser")?.finishMemoryBrowserLoad(this);
    return value;
  }

  async _loadSectionData(silent = false) {
    if (!this._selectedAgent()) return this._render();
    const view = this._viewKey();
    const loadToken = ++this._loadToken;
    const cacheGeneration = this._cacheGeneration;
    const scopeCatalogKey = this._prepareScopeCatalogVisit(view);
    const configOnly = this._isDraftView() && view !== "data-memory/conversations" && !["capabilities/request-rules", "usage-maintenance/backup-restore"].includes(view);
    if (configOnly && this._configData && (this._configDirty || !this._configDataStale) && this._draftAgentId === this._agentId
        && (this._configData.projection !== "retention" || view === "usage-maintenance/retention")) {
      this._applyConfigurationLiveMetadata(view);
      this._contentData = null;
      this._result = this._configData;
      this._error = null;
      this._busy = false;
      this._render();
      void this._loadConfigurationLiveMetadata(view);
      return;
    }
    const needsScopes = ["data-memory/memories", "data-memory/conversations"].includes(view);
    const cache = readSectionCache(this, view);
    const cacheKey = cache.key;
    const showCached = cache.result !== undefined && (cache.fresh || ["overview", "data-memory/knowledge", "usage-maintenance/usage"].includes(view));
    if (showCached) {
      this._contentData = null;
      this._result = view === "usage-maintenance/usage"
        ? {...cache.result, loading: {runs: true, retention: true}} : cache.result;
      this._error = null;
      this._busy = false;
      this._render();
      if (cache.fresh && view !== "usage-maintenance/usage") return;
      // Expired read-only summaries/lists remain useful while refreshing.
      silent = true;
    }
    if (!silent) {
      this._busy = true;
      this._render();
    }
    try {
      const configPromise = view === "data-memory/conversations" && this._data?.is_admin
        ? this._loadConfigDraft() : Promise.resolve();
      const initialScopeId = needsScopes ? this._scopeId : null;
      const cachedScopeLoadedAt = scopeCatalogKey ? this._eocScopeCatalogTimes.get(scopeCatalogKey) : null;
      const cachedScopes = cachedScopeLoadedAt && Date.now() - cachedScopeLoadedAt <= SCOPE_CACHE_TTL_MS
        ? this._scopeCatalogCache.get(scopeCatalogKey) : null;
      const knownScopes = cachedScopes || this._data?.scopes || this._baseScopes || [];
      const canPrefetchScopedCollection = Boolean(
        initialScopeId && knownScopes.some((scope) => scope.scope_id === initialScopeId),
      );
      const loadScopedCollection = (scopeId) => view === "data-memory/conversations"
        ? this._call("conversations", "list", { scope_id: scopeId, limit: 50 })
        : this._call("memories", this._memoryKind === "temporary" ? "temporary_list" : "list", { scope_id: scopeId, limit: 100 });
      const scopeCatalogKind = this._scopeCatalogKind(view);
      const scopePromise = needsScopes ? this._loadScopes(scopeCatalogKey, scopeCatalogKind) : Promise.resolve();
      const prefetchedScopedCollection = canPrefetchScopedCollection
        ? loadScopedCollection(initialScopeId).then(
          (value) => ({status: "fulfilled", value}),
          (reason) => ({status: "rejected", reason}),
        )
        : null;
      const activeConversationsPromise = view === "data-memory/conversations" && this._data?.is_admin
        ? this._call("conversations", "active") : Promise.resolve({active: []});
      // Memory still waits for an authoritative scope selection. History may
      // render a known selected scope immediately while its archive counts refresh.
      if (needsScopes && (view !== "data-memory/conversations" || !prefetchedScopedCollection)) {
        await scopePromise;
      }
      if (loadToken !== this._loadToken) return;
      const scopedCollection = async () => {
        if (prefetchedScopedCollection && initialScopeId === this._scopeId) {
          const settled = await prefetchedScopedCollection;
          if (settled.status === "rejected") throw settled.reason;
          return settled.value;
        }
        return loadScopedCollection(this._scopeId);
      };
      let result;
      let contentData = null;
      let usageSecondary = null;
      let usagePrimaryComplete = false;
      const usageDetailGeneration = this._usageDetailGeneration || 0;
      let guestDetailsSecondary = null;
      if (view === "overview") {
        this._markColdLifecycle("overview-summary-start");
        const summary = await consumeIntentRead(this, view, "overview", "summary");
        this._markColdLifecycle("overview-summary-complete");
        if (loadToken !== this._loadToken) return;
        const {agent, ...overview} = summary;
        if (agent) Object.assign(this._selectedAgent(), agent);
        result = overview;
      } else if (view === "usage-maintenance/usage") {
        const summaryPromise = cache.fresh && cache.result?.summary
          ? Promise.resolve(cache.result.summary) : this._call("usage", "summary");
        const daysPromise = cache.fresh && cache.result?.days
          ? Promise.resolve(cache.result.days) : this._call("usage", "daily");
        const settle = (promise) => promise.then(
          (value) => ({status: "fulfilled", value}),
          (reason) => ({status: "rejected", reason}),
        );
        const runsPromise = settle(this._call("usage", "runs", { limit: 30 }));
        const retentionPromise = settle(this._call("usage", "retention"));
        const primary = await Promise.allSettled([summaryPromise, daysPromise]);
        const settledPrimary = settledSectionResult([["summary", "Usage summary"], ["days", "Daily usage"]], primary);
        usagePrimaryComplete = settledPrimary.load_errors.length === 0;
        result = {
          ...(showCached ? cache.result : {}),
          ...settledPrimary,
          loading: {runs: true, retention: true},
        };
        usageSecondary = [
          ["runs", "Recent runs", runsPromise],
          ["retention", "Usage retention", retentionPromise],
        ];
      } else if (view === "data-memory/conversations") {
        const sessions = await scopedCollection();
        const admin = Boolean(this._data?.is_admin);
        contentData = {
          sessions,
          active: {active: []},
          loading: {
            scopes: Boolean(prefetchedScopedCollection),
            active: admin,
            config: admin,
          },
          load_errors: [],
        };
        result = admin ? (this._configData || null) : contentData;

        const patchHistory = (key, settled, label) => {
          if (loadToken !== this._loadToken || cacheGeneration !== this._cacheGeneration
              || this._viewKey() !== "data-memory/conversations" || !this._contentData) return;
          const loadErrors = (this._contentData.load_errors || []).filter((issue) => issue.key !== key);
          const loading = {...(this._contentData.loading || {}), [key]: false};
          if (settled.status === "rejected") {
            this._contentData = {
              ...this._contentData,
              loading,
              load_errors: [...loadErrors, {
                key,
                label,
                message: settled.reason?.message || String(settled.reason || "Unknown error"),
              }],
            };
          } else {
            this._contentData = {...this._contentData, loading, load_errors: loadErrors};
            if (key === "active") this._contentData.active = settled.value;
            if (key === "config") this._result = this._configData;
          }
          if (key === "active" && getRouteFeature("data-memory/conversations")?.reconcileActiveConversations(this)) return;
          if (key === "config") {
            if (this._patchHistorySettings()) return;
            // An open conversation dialog keeps its list and editor DOM intact.
            if (this.shadowRoot?.querySelector?.("[data-eoc-history-config]")) return;
          }
          this._render();
        };

        if (prefetchedScopedCollection) {
          void scopePromise.then(async () => {
            if (loadToken !== this._loadToken || cacheGeneration !== this._cacheGeneration
                || this._viewKey() !== "data-memory/conversations") return;
            let refreshedSessions = null;
            if (this._scopeId !== initialScopeId) {
              refreshedSessions = await loadScopedCollection(this._scopeId);
              if (loadToken !== this._loadToken || cacheGeneration !== this._cacheGeneration
                  || this._viewKey() !== "data-memory/conversations") return;
            }
            if (this._contentData) {
              this._contentData = {
                ...this._contentData,
                ...(refreshedSessions ? {sessions: refreshedSessions} : {}),
                loading: {...(this._contentData.loading || {}), scopes: false},
              };
              if (!refreshedSessions && reconcileScopePicker(this)) return;
              this._render();
            }
          }).catch((reason) => patchHistory("scopes", {status: "rejected", reason}, "Scope catalogue"));
        }
        if (admin) {
          void Promise.resolve(activeConversationsPromise).then(
            (value) => patchHistory("active", {status: "fulfilled", value}, "Active conversations"),
            (reason) => patchHistory("active", {status: "rejected", reason}, "Active conversations"),
          );
          void Promise.resolve(configPromise).then(
            () => patchHistory("config", {status: "fulfilled", value: this._configData}, "Archive settings"),
            (reason) => patchHistory("config", {status: "rejected", reason}, "Archive settings"),
          );
        }
      } else if (view === "data-memory/memories") {
        result = await scopedCollection();
      } else if (view === "data-memory/knowledge") {
        result = await consumeIntentRead(this, view, "knowledge", "list");
      } else if (view === "capabilities/guest-mode") {
        const detailsPromise = this._call("guest_mode", "details").then(
          (value) => ({status: "fulfilled", value}),
          (reason) => ({status: "rejected", reason}),
        );
        result = await this._call("guest_mode", "get");
        result = {
          ...result,
          loading: {...(result.loading || {}), details: true},
          load_errors: result.load_errors || [],
        };
        guestDetailsSecondary = detailsPromise;
        if (this._unsavedState?.scopes.get("capabilities/guest-mode")?.agent !== this._agentId) this._guestDraft = JSON.parse(JSON.stringify(result.config || {}));
        if (!result.legacy_policy) {
          this._guestMigrationReview = false;
          this._guestStartingFresh = false;
        }
      } else if (view === "capabilities/request-rules") {
        result = await consumeIntentRead(this, view, "request_rules", "list");
      } else if (this._isDraftView() && view !== "usage-maintenance/backup-restore") {
        await this._loadConfigDraft();
        result = this._configData;
      } else {
        result = null;
      }
      if (loadToken !== this._loadToken) return;
      if (cacheGeneration !== this._cacheGeneration) return;
      this._contentData = contentData;
      this._result = result;
      if (view === "usage-maintenance/usage") {
        if (usagePrimaryComplete && result?.summary && result?.days && !cache.fresh) {
          writeSectionCache(this, cacheKey, {summary: result.summary, days: result.days});
        }
      } else writeSectionCache(this, cacheKey, result);
      this._error = null;
      if (guestDetailsSecondary) {
        void guestDetailsSecondary.then((settled) => {
          if (loadToken !== this._loadToken || cacheGeneration !== this._cacheGeneration
              || this._viewKey() !== "capabilities/guest-mode") return;
          const errors = (this._result?.load_errors || []).filter((issue) => issue.key !== "details");
          const loading = {...(this._result?.loading || {}), details: false};
          if (settled.status === "fulfilled") {
            this._result = {
              ...(this._result || {}),
              ...settled.value,
              load_errors: errors,
              loading,
            };
          } else {
            this._result = {
              ...(this._result || {}),
              load_errors: [...errors, {
                key: "details",
                label: "Guest Mode capabilities",
                message: settled.reason?.message || String(settled.reason || "Unknown error"),
              }],
              loading,
            };
          }
          this._render();
        });
      }
      if (usageSecondary) {
        for (const [key, label, pending] of usageSecondary) {
          void pending.then((settled) => {
            if (loadToken !== this._loadToken || cacheGeneration !== this._cacheGeneration
                || (key === "runs" && usageDetailGeneration !== (this._usageDetailGeneration || 0))
                || this._viewKey() !== "usage-maintenance/usage") return;
            const errors = (this._result?.load_errors || []).filter((issue) => issue.key !== key);
            const loading = {...(this._result?.loading || {}), [key]: false};
            if (settled.status === "fulfilled") {
              this._result = {...(this._result || {}), [key]: settled.value, load_errors: errors, loading};
            } else {
              this._result = {
                ...(this._result || {}),
                load_errors: [...errors, {
                  key,
                  label,
                  message: settled.reason?.message || String(settled.reason || "Unknown error"),
                }],
                loading,
              };
            }
            if (!getRouteFeature("usage-maintenance/usage")?.reconcileUsageSecondary(this, key)) this._render();
          });
        }
      }
    } catch (err) {
      if (loadToken === this._loadToken) this._error = err.message || String(err);
    } finally {
      if (loadToken === this._loadToken) {
        this._busy = false;
        this._render();
        if (this._isDraftView() && this._configData && this._draftAgentId === this._agentId) {
          void this._loadConfigurationLiveMetadata(view);
        }
      }
    }
  }

  _configurationLiveMetadataKeys(view = this._viewKey()) {
    return {
      "capabilities/home-assistant": ["local_handling"],
      "assistant/prompt-context": ["exposed_attribute_catalog"],
    }[view] || [];
  }

  _applyConfigurationLiveMetadata(view = this._viewKey()) {
    const cache = this._eocLiveMetadataCache;
    if (!cache || !this._configData) return false;
    let changed = false;
    for (const key of this._configurationLiveMetadataKeys(view)) {
      const record = cache.get(key);
      if (!record || record.agentId !== this._agentId
          || record.epoch !== (this._eocLiveMetadataEpoch || 0)
          || record.revision !== this._configData.revision
          || this._configData[key] !== undefined) continue;
      this._configData = {...this._configData, [key]: record.value};
      changed = true;
    }
    return changed;
  }

  async _loadConfigurationLiveMetadata(view = this._viewKey()) {
    if (!this._configData || this._draftAgentId !== this._agentId) return;
    const requested = this._configurationLiveMetadataKeys(view);
    if (!requested.length) return;
    this._eocLiveMetadataCache ||= new Map();
    this._eocLiveMetadataPending ||= new Map();
    const agentId = this._agentId;
    const epoch = this._eocLiveMetadataEpoch || 0;
    const revision = this._configData.revision;
    const activeToken = this._viewKey() === view ? this._loadToken : null;
    const pending = requested.filter(key => this._configData[key] === undefined).map(key => {
      const cached = this._eocLiveMetadataCache.get(key);
      if (cached?.agentId === agentId && cached.epoch === epoch && cached.revision === revision) return null;
      const existing = this._eocLiveMetadataPending.get(key);
      if (existing?.agentId === agentId && existing.epoch === epoch && existing.revision === revision) {
        if (activeToken !== null) existing.activeToken = activeToken;
        return existing.promise;
      }
      const record = {agentId, epoch, revision, activeToken, promise: null};
      record.promise = this._call("configuration", "live_metadata", {metadata_keys: [key]})
        .then(metadata => {
          const cleanRevision = this._cleanConfigSnapshots.get(this._configurationSnapshotKey(agentId, "full"))?.result?.revision;
          if (agentId !== this._agentId || epoch !== (this._eocLiveMetadataEpoch || 0)
              || (revision !== this._configData?.revision && revision !== cleanRevision)
              || !Object.hasOwn(metadata, key)) return;
          this._eocLiveMetadataCache.set(key, {agentId, epoch, revision, value: metadata[key]});
          if (this._viewKey() !== view || record.activeToken !== this._loadToken) return;
          const previous = this._configData;
          if (this._applyConfigurationLiveMetadata(view) && this._result === previous) {
            this._result = this._configData;
            this._render();
          }
        })
        .catch(() => {})
        .finally(() => {
          if (this._eocLiveMetadataPending.get(key) === record) this._eocLiveMetadataPending.delete(key);
        });
      this._eocLiveMetadataPending.set(key, record);
      return record.promise;
    }).filter(Boolean);
    await Promise.all(pending);
  }

  async _loadConfigDraft() {
    const diagnostics = this._eocConfigurationReadDiagnostics ||= {};
    if (!this._configData || (this._configDataStale && !this._configDirty) || this._draftAgentId !== this._agentId
        || (this._configData.projection === "retention" && this._viewKey() !== "usage-maintenance/retention")) {
      const agentId = this._agentId;
      const loadToken = this._loadToken;
      const cacheGeneration = this._cacheGeneration;
      const view = this._viewKey();
      const projection = this._viewKey() === "usage-maintenance/retention" ? "retention" : "full";
      const action = projection === "retention" ? "retention_get" : "get";
      const trace = (event, detail) => markConfigurationRead(event, {view, action, ...detail});
      const key = this._configurationSnapshotKey(agentId, projection);
      const cached = this._freshCleanConfiguration(agentId, projection);
      let configData = cached;
      if (cached) {
        discardStoredConfigurationPrefetch(this, "clean-snapshot");
        diagnostics.draft = {source: "clean-snapshot", action, sentAction: null};
        trace("draft-source", {source: "clean-snapshot"});
      } else {
        const prefetched = consumeStoredConfigurationPrefetch(this, action);
        if (prefetched) {
          const settled = await prefetched;
          const staleReason = agentId !== this._agentId ? "agent-changed"
            : view !== this._viewKey() ? "route-changed"
              : cacheGeneration !== this._cacheGeneration ? "cache-generation-changed"
                : loadToken !== this._loadToken ? "load-token-changed" : null;
          if (staleReason) {
            diagnostics.prefetch.status = "discarded";
            diagnostics.prefetch.reason = staleReason;
            diagnostics.prefetch.discardedAt = Date.now();
            trace("prefetch-discarded", {status: "discarded", reason: staleReason});
            return;
          }
          if (settled.status === "fulfilled" && settled.value?.config && typeof settled.value.config === "object") {
            configData = settled.value;
            diagnostics.draft = {source: "prefetched-request", action, sentAction: null};
            trace("draft-source", {source: "prefetched-request"});
          } else {
            diagnostics.prefetch.status = "discarded";
            diagnostics.prefetch.reason = settled.status === "rejected" ? "request-failed" : "invalid-response";
            diagnostics.prefetch.discardedAt = Date.now();
            trace("prefetch-discarded", {status: "discarded", reason: diagnostics.prefetch.reason});
          }
        }
        if (!configData) {
          diagnostics.draft = {source: "new-backend-request", action, sentAction: action};
          trace("draft-source", {source: "new-backend-request"});
          trace("fallback-started", {source: "new-backend-request", status: "started"});
          const finishFallback = measureConfigurationRead("fallback-response", {view, action, source: "new-backend-request"});
          try {
            configData = projection === "retention"
              ? await consumeIntentRead(this, view, "configuration", "retention_get")
              : await this._call("configuration", "get");
            finishFallback("fulfilled");
          } catch (err) {
            finishFallback("rejected");
            throw err;
          }
        }
      }
      if (agentId !== this._agentId || loadToken !== this._loadToken || cacheGeneration !== this._cacheGeneration || view !== this._viewKey()) return;
      // A user can edit the visible draft while an earlier configuration read
      // is in flight. The read is no longer allowed to replace that draft.
      const retainedRetentionDraft = this._configDirty && this._draftAgentId === agentId
        && this._configData?.projection === "retention" && projection === "full"
        ? {draft: this._draft, baseline: this._configData.config} : null;
      if (this._configDirty && this._draftAgentId === agentId && !retainedRetentionDraft) return;
      const prior = key ? this._cleanConfigSnapshots.get(key)?.result : null;
      if (prior && prior.revision !== configData.revision) this._invalidateCleanConfiguration(agentId);
      this._configData = configData;
      this._configDataStale = false;
      if (!cached) this._rememberCleanConfiguration(configData, agentId);
      this._draft = JSON.parse(JSON.stringify(configData.config));
      if (retainedRetentionDraft) {
        for (const [field, value] of Object.entries(retainedRetentionDraft.draft)) {
          if (!same(value, retainedRetentionDraft.baseline[field])) this._draft[field] = clone(value);
        }
      }
      this._draftTitle = configData.title;
      this._draftAgentId = agentId;
      this._setConfigDirty(false);
      if (retainedRetentionDraft) this._syncConfigDirty();
    } else {
      diagnostics.draft = {source: "active-config", action: this._viewKey() === "usage-maintenance/retention" ? "retention_get" : "get", sentAction: null};
      markConfigurationRead("draft-source", {view: this._viewKey(), action: diagnostics.draft.action, source: "active-config"});
    }
    this._result = this._configData;
  }

  async _navigate(page, subsection = null) {
    const targetSubsection = subsection || this._visibleSubsections(page)[0]?.id || null;
    const destination = targetSubsection ? `${page}/${targetSubsection}` : page;
    if (!await confirmStateSafeNavigation(this, destination)) {
      const local = this.shadowRoot?.querySelector?.("#local-section");
      const top = this.shadowRoot?.querySelector?.("#top-section-mobile");
      if (local) local.value = this._subsection;
      if (top) top.value = this._page;
      return;
    }
    return trackAsync(this, NAVIGATION_MARK_PREFIX, async () => {
      const metadata = pageMetadata(page);
      const resolvedSubsection = subsection || this._visibleSubsections(page)[0]?.id || metadata.sections[0]?.id || null;
      this._page = page;
      this._subsection = resolvedSubsection;
      this._query = "";
      history.pushState({}, "", routePath(page, resolvedSubsection));
      this._result = null;
      if (this._hydrateCleanConfiguration(this._viewKey())) this._render();
      else showPendingDestination(this);
      await this._loadSection();
    }, true);
  }

  _render(...args) {
    this._markColdLifecycle("first-render-start");
    getRouteFeature("usage-maintenance/diagnostics")?.stopDiagnosticsWatch(this);
    const view = this._viewKey?.() || null;
    const busy = Boolean(this._busy);
    const measure = startMeasure(this, RENDER_MARK_PREFIX);
    try {
      const root = this.shadowRoot;
      const main = root?.querySelector?.("[data-eoc-main]") || root?.querySelector?.("main");
      const preserve = Boolean(
        this._busy
        && this._eocNavigationDepth > 0
        && main
        && main.childNodes.length
        && !main.querySelector?.(".loading")
      );
      let result;
      if (preserve) {
        showPendingDestination(this);
      } else {
        result = this._renderContent(...args);
        if (view === "data-memory/conversations") {
          getRouteFeature(view)?.decorateConversationPager(this);
        }
      }
      if (view === "capabilities/functions") getRouteFeature(view)?.bindFunctionRepair(this);
      if (view === "usage-maintenance/request-debug") getRouteFeature(view)?.bindManagementDebug(this);
      if (view === "usage-maintenance/diagnostics") getRouteFeature(view)?.enhanceDiagnostics(this);
      if (preserve) return undefined;
      return result;
    } finally {
      const main = this.shadowRoot?.querySelector?.("[data-eoc-main]")
        || this.shadowRoot?.querySelector?.("main");
      if (main && !this._busy) {
        main.removeAttribute("aria-busy");
        main.inert = false;
        main.classList.remove("eoc-loading-in-background");
      }
      syncAgentPicker(this);
      finishMeasure(measure, {view, busy});
      this._markColdLifecycle("first-render-complete");
    }
  }

  _renderContent() {
    const view = this._viewKey();
    const ownsPageDraft = ["capabilities/guest-mode", "capabilities/quiet-hours", "capabilities/request-rules"].includes(view);
    if (ownsPageDraft) initializePageDraft(this);
    renderManagement(this);
    this._eocRenderedFeatureReady = routeFeaturesReady(view);
    const main = this.shadowRoot?.querySelector?.("[data-eoc-main]") || this.shadowRoot?.querySelector?.("main");
    const routeTitle = main?.querySelector?.(".page-intro h1");
    if (routeTitle && this._markColdLifecycle("route-title-present", {view: this._viewKey()})) {
      requestAnimationFrame(() => this._markColdLifecycle("route-title-next-frame", {view: this._viewKey()}));
    }
    if (this._viewKey() === "overview" && main?.querySelector?.(".dashboard-grid")
        && this._markColdLifecycle("overview-content-present")) {
      requestAnimationFrame(() => this._markColdLifecycle("overview-content-next-frame"));
    }
    if (view === "capabilities/request-rules") {
      bindRequestRuleSearch(this);
      applyRequestRuleSearch(this);
    }
    if (ownsPageDraft) refreshPageSaveBar(this);

    // Keep the former decorator ordering explicit without mutating class methods at runtime.
    this._enhanceSubsectionNavigation();
    navigationSearchModule?.enhanceNavigationSearch(this);
    enhanceConfigurationClarity(this);
    const ownsConfigurationGuidance = (
      (routeAssetKind(view) === "agent-config"
        && view !== "capabilities/functions"
        && (view !== "data-memory/conversations" || this._data?.is_admin))
      || view === "data-memory/memory-settings"
    );
    if (ownsConfigurationGuidance) {
      getRouteFeature("configuration")?.enhanceConfigurationGuidance(this);
    }
    if (this._page === "overview") queueMicrotask(() => enhanceOverviewHealthClarity(this));
  }

  _renderShell() {
    this._markColdLifecycle("shell-start");
    this._eocShellRevision = (this._eocShellRevision || 0) + 1;
    const agent = this._selectedAgent();
    const navigation = NAVIGATION.filter((item) => this._canAccessView(item.id));
    const local = this._visibleSubsections();
    const currentSection = local.find((item) => item.id === this._subsection);
    this._eocMainMarkup = !agent ? this._empty("No conversation agents configured.") : this._busy ? this._loadingContent(agent) : this._error ? `<div class="error" role="alert">${this._e(this._error)}</div>` : this._content(agent);
    const configurationActions = agent && !this._busy && !this._error ? this._configurationActions() : "";
    this._eocDialogMarkup = this._dialogs();
    this._eocRenderedRoute = `${this._agentId}|${this._viewKey()}`;
    const shell = document.createElement("template");
    shell.innerHTML = `
      <div class="page-shell" data-eoc-persistent-shell>
        <header>
          <div class="page-heading"><h1>Extended OpenAI</h1><p>Configure your assistant, capabilities, retained data, and maintenance.</p></div>
          ${settingsSearchShellMarkup(this)}
        </header>
        <label class="mobile-nav"><span>Page</span><select id="top-section-mobile" tabindex="0">${navigation.map((item) => `<option value="${item.id}" ${item.id === this._page ? "selected" : ""}>${item.label}</option>`).join("")}</select></label>
        <div class="eoc-agent-context-row" aria-label="Assistant context">
          <label class="agent-picker"><span>Conversation agent</span><select id="agent">${(this._data?.agents || []).map((a) => `<option value="${this._e(a.subentry_id)}" ${a.subentry_id === this._agentId ? "selected" : ""}>${this._e(a.title)}</option>`).join("")}</select>${agent ? `<small>${this._e(agent.provider)} · ${this._e(agent.model)}</small>` : ""}</label>
          <div id="eoc-agent-actions-host" class="eoc-agent-actions" ${configurationActions ? "" : "hidden"}>${configurationActions}</div>
        </div>
        <nav class="top-nav" aria-label="Management sections">${navigation.map((item) => `<button type="button" data-page="${item.id}" class="${item.id === this._page ? "active" : ""}" ${item.id === this._page ? 'aria-current="page"' : ""}>${item.label}</button>`).join("")}</nav>
        <nav class="subsection-nav" aria-label="${this._e(pageMetadata(this._page).label)} sections" ${local.length > 1 ? "" : "hidden"}>${local.length > 1 ? local.map((item) => `<button type="button" data-subsection="${this._e(item.id)}" class="${item.id === this._subsection ? "active" : ""}" ${item.id === this._subsection ? 'aria-current="page"' : ""}>${this._e(item.label)}</button>`).join("") : ""}</nav>
        <div id="eoc-scope-host">${["data-memory/conversations", "data-memory/memories"].includes(this._viewKey()) ? this._scopePicker() : ""}</div>
        <div id="eoc-section-host">${local.length > 1 ? `<div class="section-selector"><label><span>${this._e(pageMetadata(this._page).label)} section</span><select id="local-section" tabindex="0" aria-description="${this._e(currentSection?.description || "")}">${local.map((item) => `<option value="${this._e(item.id)}" ${item.id === this._subsection ? "selected" : ""}>${this._e(item.label)}</option>`).join("")}</select></label></div>` : ""}</div>
        <div id="eoc-assistant-intro-host">${agent && this._page === "assistant" ? ASSISTANT_INTRO_MARKUP : ""}</div>
        <div class="section-layout">
          <main data-eoc-main ${this._page === "guide" ? 'data-eoc-guide-layout=""' : ""}>${this._eocMainMarkup}</main>
        </div>
      </div>
      <div id="eoc-dialog-host">${this._eocDialogMarkup}</div>
      <div id="toast" class="toast" role="status" aria-live="polite"></div>`;
    for (const child of [...this.shadowRoot.children]) {
      if (!child.matches("style[data-eoc-critical-styles],link[data-eoc-persistent-styles]")) child.remove();
    }
    this.shadowRoot.append(shell.content);
    this._bindActions();
    this._markColdLifecycle("shell-complete");
    if (this.shadowRoot.querySelector(".page-heading h1")
        && this._markColdLifecycle("shell-title-present")) {
      requestAnimationFrame(() => this._markColdLifecycle("shell-next-frame"));
    }
  }

  _reconcileCollectionView() {
    const view = this._viewKey();
    if (view === "data-memory/knowledge") return getRouteFeature(view)?.reconcileKnowledge(this) || false;
    if (view === "data-memory/memories") return getRouteFeature(view)?.reconcileMemories(this) || false;
    if (view === "capabilities/request-rules") return getRouteFeature(view)?.reconcileRequestRules?.(this) || false;
    if (view !== "capabilities/functions") return false;
    const repair = getRouteFeature(view);
    if (repair?.repairIssue(this) && repair.repairMetadata(this)?.isolatable === false) return false;
    return getConfigurationTools()?.reconcileTools?.(this, {repairCards: repair?.renderFunctionRepairCards(this) || ""}) || false;
  }

  _loadingContent(agent) {
    if (this._viewKey() === "capabilities/home-assistant") {
      return `${this._homeAssistantIntro()}${this._loading()}`;
    }
    return this._loading();
  }

  _content(agent) {
    const view = this._viewKey();
    if (!routeFeaturesReady(view)) return this._loadingContent(agent);
    if (view === "data-memory/memory-settings") return getRouteFeature(view)?.renderMemorySettings(this) || this._loading();
    if (view === "capabilities/home-assistant") {
      this._configSections = ["local"];
      return `${this._homeAssistant(agent)}${(getConfigurationEditor()?.renderConfiguration(this) || this._loading())}`;
    }
    if (view === "capabilities/web-skills") {
      this._configSections = ["capabilities"];
      return (getConfigurationEditor()?.renderConfiguration(this) || this._loading());
    }
    if (!this._canAccessView(this._page, this._subsection)) return this._empty("Administrator permission is required for this section.");
    if (view === "overview") return renderOverview(this, agent);
    if (view === "guide") return renderGuide(this);
    if (this._page === "assistant") {
      this._configSections = this._configSectionsForView();
      const voiceIdentity = view === "assistant/voice" ? getRouteFeature(view)?.renderVoiceIdentityCore : null;
      const specialized = getRouteFeature(view);
      return (getConfigurationEditor()?.renderConfiguration(this, {
        voiceIdentity,
        renderExposedAttributes: view === "assistant/prompt-context" ? () => `<div class="exposed-attribute-settings" data-exposed-feature style="min-height:96px;padding:16px 0 4px;border-top:1px solid var(--divider-color)"><h3>Additional entity attributes</h3><p class="help">Loading Assist-exposed entity choices…</p></div>` : specialized?.renderExposedAttributeSettings,
      }) || this._loading());
    }
    if (view === "capabilities/request-rules") return getRouteFeature("capabilities/request-rules")?.renderRequestRules(this) || this._loading();
    if (view === "capabilities/functions") {
      const repair = getRouteFeature(view);
      const issue = repair?.repairIssue(this);
      if (issue && repair.repairMetadata(this)?.isolatable === false) return repair.renderFallbackRepair(this, issue);
      const repairCards = issue ? repair.renderFunctionRepairCards(this) : "";
      return getConfigurationTools()?.renderTools(this, {repairCards}) || this._loading();
    }
    if (view === "capabilities/quiet-hours") return getRouteFeature(view)?.renderQuietHours(this) || this._loading();
    if (view === "usage-maintenance/request-debug") return getRouteFeature(view)?.renderManagementDebug(this) || this._loading();
    if (view === "capabilities/guest-mode") return this._guestMode();
    if (view === "data-memory/memories") return this._memories();
    if (view === "data-memory/knowledge") return `<section class="page-intro"><h1>Knowledge Library</h1><p>Manage reference sources the assistant can search when needed. <button type="button" class="guide-topic-link guide-link" data-guide-topic="knowledge">Learn more</button></p></section>${this._knowledge()}`;
    if (view === "data-memory/conversations") {
      this._configSections = ["archive"];
      return `${this._conversations()}<div data-eoc-history-config>${this._historySettingsMarkup()}</div>`;
    }
    if (view === "usage-maintenance/usage") return this._usage();
    if (view === "usage-maintenance/backup-restore") return getRouteFeature(view)?.renderBackupTransferPanel(Boolean(this._configDirty)) || this._loading();
    if (view === "usage-maintenance/diagnostics") return this._diagnostics(agent);
    if (view === "usage-maintenance/retention") {
      return getRouteFeature(view)?.renderRetentionSettings(this) || this._loading();
    }
    return this._empty("This section is not available.");
  }

  _homeAssistantIntro() {
    return `<section class="page-intro"><h1>Home Assistant access</h1><p>Home Assistant controls which entities this assistant is allowed to access through Assist. Extended OpenAI can also automatically include exposed entity names and current states in the context sent to the model.</p></section>`;
  }

  _homeAssistant(agent) {
    const contextIncluded = this._draft?.exposed_entities_enabled === true;
    return `${this._homeAssistantIntro()}<section class="content-card access-explainer"><div><h2>Entity access</h2><p>Home Assistant's Assist exposure settings decide which entities may be used by the assistant. Manage exposure in Home Assistant's voice assistant settings.</p></div><div class="compact-status"><span><strong>Include exposed entity states in the prompt</strong><small>Adds exposed entity names and current states to the context sent with each request. Turning this off does not necessarily prevent the assistant from using exposed entities through Home Assistant tools.</small></span><strong class="status-value ${contextIncluded ? "on" : ""}">${contextIncluded ? "On" : "Off"}</strong></div><button type="button" class="secondary inline-route" data-page="assistant" data-subsection="prompt-context">Configure exposed entity context</button></section><section class="home-assistant-boundary"><div><strong>Guest Mode</strong><p>Adds extra restrictions to the assistant's normal Home Assistant access when active.</p></div><button type="button" class="secondary inline-route compact-button" data-page="capabilities" data-subsection="guest-mode">Configure Guest Mode</button></section>`;
  }
  _usage() {
    const usage = getRouteFeature("usage-maintenance/usage");
    return usage ? usage.renderUsagePage(this, this._result || {}) : this._loading();
  }

  _conversations() {
    return getRouteFeature("data-memory/conversations")?.renderConversations(this) || this._loading();
  }

  _historySettingsMarkup() {
    if (!this._data?.is_admin) return "";
    const error = this._contentData?.load_errors?.find((issue) => issue.key === "config")?.message
      || this._eocHistoryEditorError;
    if (error) return `<section class="content-card"><div class="error" role="alert">Archive settings unavailable: ${this._e(error)}</div></section>`;
    if (this._configData && this._draftAgentId === this._agentId && !this._eocHistoryEditorLoading) {
      const markup = getConfigurationEditor()?.renderConfiguration(this);
      if (markup) return markup;
    }
    return `<section class="content-card"><div class="loading" role="status">Loading archive settings…</div></section>`;
  }

  _patchHistorySettings() {
    if (this._viewKey?.() !== "data-memory/conversations" || !this._contentData) return false;
    if (!reconcileHistoryConfiguration(this, this._historySettingsMarkup())) return false;
    if (this._configData && this._draftAgentId === this._agentId && !this._eocHistoryEditorLoading
        && !this._eocHistoryEditorError) getConfigurationEditor()?.bindConfiguration(this);
    return true;
  }

  _memories() {
    if (this._memoryKind === "temporary") return getRouteFeature("data-memory/memories")?.renderTemporaryMemories(this);
    return getRouteFeature("memory-browser")?.renderPersistentMemories(this);
  }

  _knowledge() {
    return getRouteFeature("data-memory/knowledge")?.renderKnowledge(this) || this._loading();
  }

  _guestMode() {
    return getRouteFeature("capabilities/guest-mode")?.guestMode(this) || this._loading();
  }

  _guestPolicyView(status, policy, state) {
    return getRouteFeature("capabilities/guest-mode")?.guestPolicyView(this, status, policy, state) || this._loading();
  }

  _setupGuestSelectors() {
    const config = this._guestDraft || {};
    const result = this._result || {};
    const select = (items, value, label) => ({select: {multiple: true, custom_value: false, options: items.map((item) => ({value: value(item), label: label(item)}))}});
    this.shadowRoot.querySelectorAll("ha-selector[data-guest-key]").forEach((element) => {
      const type = element.dataset.guestSelector;
      element.hass = this.hass;
      element.value = config[element.dataset.guestKey] || [];
      element.selector = type === "entity" ? {entity: {multiple: true}} : type === "area" ? {area: {multiple: true}} : type === "label" ? {label: {multiple: true}} : type === "domain" ? select(result.domains || [], (item) => item, (item) => item) : type === "knowledge" ? select(result.knowledge_sources || [], (item) => item.source_id, (item) => `${item.title} — ${item.description || "No description"}`) : type === "group" ? select(result.function_groups || [], (item) => item.id, (item) => `${item.name} — ${item.description}`) : select((result.functions || []).filter((item) => !item.unsafe_in_guest_mode), (item) => item.name, (item) => `${item.name}${item.enabled ? "" : " (disabled)"} — ${item.description || "No description"}`);
      element.addEventListener("value-changed", (event) => {
        config[element.dataset.guestKey] = event.detail.value || [];
        queueMicrotask(() => refreshPageSaveBar(this));
      });
    });
  }

  async _saveGuestPolicy() {
    return savePageChanges(this);
  }

  async _startFreshGuestPolicy() {
    if (!await this._confirm(
      "Start a fresh Guest policy?",
      "Starting fresh means guests will be able to use all Home Assistant entities normally available to this assistant unless you add exclusions. The existing policy remains enforced until you save.",
      "Start fresh",
    )) return;
    this._guestDraft = getRouteFeature("memory-browser")?.freshGuestPolicyDraft(this._guestDraft);
    this._guestMigrationReview = true;
    this._guestStartingFresh = true;
    this._render();
  }

  _dateTimeLocal(value) {
    if (!value) return "";
    const date = new Date(value);
    const shifted = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
    return shifted.toISOString().slice(0, 16);
  }

  _runGuestOperation(operation) {
    if (this._guestOperation) return this._guestOperation;
    const pending = Promise.resolve().then(operation);
    this._guestOperation = pending;
    syncAgentPicker(this);
    return pending.finally(() => {
      this._guestOperation = null;
      syncAgentPicker(this);
    });
  }

  _patchGuestModeStatus(agentId, status) {
    if (!agentId || !status) return;
    const agent = this._data?.agents?.find((item) => item.subentry_id === agentId);
    if (agent) agent.guest_mode = {...(agent.guest_mode || {}), ...status};
    if (this._agentId === agentId && this._viewKey() === "capabilities/guest-mode" && this._result) {
      this._result = {...this._result, status: {...(this._result.status || {}), ...status}};
    }
  }

  async _refreshGuestModeMutation(agentId, mutationResult) {
    if (this._agentId !== agentId || this._viewKey() !== "capabilities/guest-mode") return;
    const loadToken = ++this._loadToken;
    this._patchGuestModeStatus(agentId, mutationResult?.status);
    this._result = {...this._result, loading: {...(this._result?.loading || {}), details: true}};
    this._error = null;
    this._render();
    try {
      const details = await this._call("guest_mode", "details");
      if (this._agentId !== agentId || this._viewKey() !== "capabilities/guest-mode" || this._loadToken !== loadToken) return;
      this._result = {...this._result, ...details, loading: {...this._result.loading, details: false},
        load_errors: (this._result.load_errors || []).filter(issue => issue.key !== "details")};
    } catch (err) {
      if (this._agentId !== agentId || this._viewKey() !== "capabilities/guest-mode" || this._loadToken !== loadToken) return;
      this._result = {...this._result, loading: {...this._result.loading, details: false},
        load_errors: [...(this._result.load_errors || []).filter(issue => issue.key !== "details"),
          {key: "details", label: "Guest Mode capabilities", message: err.message || String(err)}]};
    }
    this._render();
  }

  _updateGuestMode(now = false) {
    return this._runGuestOperation(async () => {
      const root = this.shadowRoot;
      const agentId = this._agentId;
      const indefinite = root.querySelector("#guest-indefinite")?.checked ?? true;
      const start = now ? new Date().toISOString() : root.querySelector("#guest-start")?.value;
      const end = root.querySelector("#guest-end")?.value;
      try {
        const result = await this._call("guest_mode", "update", {
          ...(start ? {active_from: start} : {}),
          ...(!indefinite && end ? {active_until: end} : {}),
          indefinite: indefinite || !end,
        });
        await this._refreshGuestModeMutation(agentId, result);
        this._toast("Guest Mode updated");
      } catch (err) {
        this._toast(`Unable to update Guest Mode: ${err.message || String(err)}`, true);
      }
    });
  }

  _disableGuestMode() {
    return this._runGuestOperation(async () => {
      if (!await this._confirm("End Guest Mode?", "This immediately ends an active interval or cancels a future schedule.", "End Guest Mode")) return;
      const agentId = this._agentId;
      try {
        const result = await this._call("guest_mode", "disable");
        await this._refreshGuestModeMutation(agentId, result);
        this._toast("Guest Mode ended");
      } catch (err) {
        this._toast(`Unable to end Guest Mode: ${err.message || String(err)}`, true);
      }
    });
  }

  _diagnostics(agent) {
    return getRouteFeature("status")?.diagnosticsMarkup(this, agent);
  }

  _dialogOwnership() {
    const owner = `${this._data?.entry_id || ""}|${this._agentId}|${this._viewKey()}|${this._memoryKind || ""}`;
    if (this._eocDialogOwner !== owner) {
      this._eocDialogOwner = owner;
      this._eocOwnedDialogs = new Set();
    }
    return this._eocOwnedDialogs;
  }

  _dialogs() {
    const owned = this._dialogOwnership();
    const view = this._viewKey();
    const content = `${view === "data-memory/knowledge" && owned.has("knowledge-dialog") ? `<dialog id="knowledge-dialog" class="editor-dialog wide" aria-labelledby="knowledge-dialog-title"><form id="knowledge-form"><div class="dialog-header"><h2 id="knowledge-dialog-title">Add Knowledge source</h2><button type="button" class="icon close-editor" aria-label="Close">×</button></div><div class="dialog-body"><label>Title<input id="knowledge-title" maxlength="${KNOWLEDGE_TITLE_LIMIT}" required></label><label>Description<textarea id="knowledge-description" class="short-textarea" maxlength="${KNOWLEDGE_DESCRIPTION_LIMIT}" spellcheck="true"></textarea></label>${knowledgeSourceAvailabilityControl()}<label>Content<textarea id="knowledge-content" class="knowledge-editor" maxlength="${KNOWLEDGE_LIMIT}" required spellcheck="true"></textarea></label><div id="knowledge-counter" class="counter">0 / ${KNOWLEDGE_LIMIT.toLocaleString()} characters</div><div id="knowledge-error" class="inline-error" role="alert"></div></div><div class="dialog-actions"><button type="button" id="knowledge-delete" class="danger" hidden>Delete</button><button type="button" class="secondary close-editor">Cancel</button><button type="submit" id="knowledge-save">Save</button></div></form></dialog>` : ""}
      ${view === "data-memory/memories" && this._memoryKind !== "temporary" && owned.has("memory-dialog") ? `<dialog id="memory-dialog" class="editor-dialog" aria-labelledby="memory-dialog-title"><form id="memory-form"><div class="dialog-header"><h2 id="memory-dialog-title">Add memory</h2><button type="button" class="icon close-editor" aria-label="Close">×</button></div><div class="dialog-body"><label>Memory<textarea id="memory-content" required spellcheck="true" placeholder="What should the agent remember?"></textarea></label><label>Category<input id="memory-category" value="general" required></label><div id="memory-metadata"></div><p id="memory-meta" class="meta"></p><div id="memory-error" class="inline-error" role="alert"></div></div><div class="dialog-actions"><button type="button" id="memory-delete" class="danger" hidden>Delete</button><button type="button" class="secondary close-editor">Cancel</button><button type="submit" id="memory-save">Save</button></div></form></dialog>` : ""}
      ${view === "data-memory/conversations" && owned.has("session-dialog") ? `<dialog id="session-dialog" class="editor-dialog wide" aria-labelledby="session-title"><div class="dialog-header"><h2 id="session-title">Conversation</h2><button type="button" class="icon close-session" aria-label="Close">×</button></div><div id="session-body" class="dialog-body session-body"></div><div class="dialog-actions"><button type="button" class="secondary close-session">Close</button></div></dialog>` : ""}
      ${view === "data-memory/memories" && this._memoryKind !== "temporary" && owned.has("reassign-dialog") ? `<dialog id="reassign-dialog" class="editor-dialog" aria-labelledby="reassign-title"><div class="dialog-header"><h2 id="reassign-title">Assign unowned memory</h2></div><div class="dialog-body"><p class="help">Choose the user or household that should be able to use this older memory.</p><label>Assign to<select id="reassign-scope"></select></label></div><div class="dialog-actions"><button type="button" class="secondary" id="reassign-cancel">Cancel</button><button type="button" id="reassign-save">Assign memory</button></div></dialog>` : ""}
      <dialog id="confirm-dialog" class="editor-dialog confirm-dialog" aria-labelledby="confirm-title"><div class="dialog-header"><h2 id="confirm-title">Confirm</h2></div><div class="dialog-body"><p id="confirm-message"></p></div><div class="dialog-actions"><button type="button" class="secondary" id="confirm-cancel">Cancel</button><button type="button" class="danger" id="confirm-accept">Confirm</button></div></dialog>
      ${this._viewKey() === "capabilities/request-rules" ? (getRouteFeature("capabilities/request-rules")?.requestRulesDialog(this) || "") : ""}${routeAssetKind(this._viewKey()) === "agent-config" ? getConfigurationEditor()?.configurationDialogs(this) || "" : this._viewKey() === "capabilities/functions" ? getConfigurationTools()?.configurationDialogs(this) || "" : ""}${this._viewKey() === "usage-maintenance/backup-restore" ? getRouteFeature("usage-maintenance/backup-restore")?.renderRestoreTransferDialog(this) || "" : ""}`;
    const usageDialog = this._viewKey() === "usage-maintenance/usage"
      ? getRouteFeature("usage-maintenance/usage")?.requestDetailsDialog() || "" : "";
    return `${content}${view === "data-memory/memories" && this._memoryKind === "temporary" && owned.has("temporary-memory-dialog") ? getRouteFeature(view)?.temporaryDialog(this) || "" : ""}${usageDialog}`;
  }

  _ensureOwnedDialog(id) {
    this._dialogOwnership().add(id);
    const markup = this._dialogs();
    if (markup !== this._eocDialogMarkup || !this.shadowRoot.querySelector(`#${id}`)) {
      updateDialogs(this, markup, {preserveEditors: true});
      this._eocDialogMarkup = markup;
    }
  }

  _bindActiveConversationActions() {
    this.shadowRoot.querySelectorAll(".end-active").forEach((button) => button.addEventListener("click", async () => {
      const agentId = this._agentId;
      const loadToken = this._loadToken;
      if (!await this._confirm("End active conversation?", "The next matching Assist request will start with fresh model context.", "End conversation")) return;
      if (agentId !== this._agentId || loadToken !== this._loadToken
          || this._viewKey() !== "data-memory/conversations") return;
      try {
        const response = await this._call("conversations", "end_active", { continuity_key: button.dataset.key });
        if (response?.ended && agentId === this._agentId && loadToken === this._loadToken
            && this._viewKey() === "data-memory/conversations" && this._contentData?.active?.active) {
          this._contentData = {
            ...this._contentData,
            active: {
              ...this._contentData.active,
              active: this._contentData.active.active.filter((item) => item.key !== button.dataset.key),
            },
          };
          getRouteFeature("data-memory/conversations")?.reconcileActiveConversations(this);
        }
        this._toast("Conversation will start fresh next time");
      } catch (err) {
        this._toast(`Unable to end conversation: ${err.message || String(err)}`, true);
      }
    }));
  }

  _bindActions() {
    const root = this.shadowRoot;
    const q = (selector) => root.querySelector(selector);
    if (!["data-memory/knowledge", "data-memory/memories"].includes(this._viewKey())) {
      q("#list-search")?.addEventListener("input", (event) => { this._query = event.target.value; this._updateVisibleList(); });
    }
    const view = this._viewKey();
    if (view === "data-memory/knowledge") getRouteFeature(view)?.bindKnowledge(this);
    if (view === "data-memory/conversations") getRouteFeature(view)?.bindConversationActions(this);
    if (view === "data-memory/conversations") {
      this._bindActiveConversationActions();
      root.querySelectorAll(".delete-session").forEach((button) => button.addEventListener("click", (event) => { event.stopPropagation(); this._deleteSession(button.dataset.id); }));
    }
    if (view === "usage-maintenance/usage") q("#clear-details")?.addEventListener("click", () => this._clearUsageDetails());
    q("#test-agent")?.addEventListener("click", () => this._testAgent());
    if (view === "capabilities/guest-mode") {
      q("#guest-indefinite")?.addEventListener("change", (event) => { const end = q("#guest-end"); if (end) end.disabled = event.target.checked; });
      q("#guest-update")?.addEventListener("click", () => this._updateGuestMode(false));
      q("#guest-now")?.addEventListener("click", () => this._updateGuestMode(true));
      q("#guest-disable")?.addEventListener("click", () => this._disableGuestMode());
      q("#guest-review-converted")?.addEventListener("click", () => { this._guestMigrationReview = true; this._render(); });
      q("#guest-start-fresh")?.addEventListener("click", () => this._startFreshGuestPolicy());
      q("#guest-separate-control")?.addEventListener("change", (event) => { this._guestDraft.guest_separate_control_restrictions = event.target.checked; this._render(); });
      q("#guest-controls-enabled")?.addEventListener("change", (event) => { this._guestDraft.guest_mode_enabled = event.target.checked; });
      root.querySelectorAll("[data-guest-mode]").forEach((element) => element.addEventListener("change", () => { this._guestDraft[element.dataset.guestMode] = element.value; this._render(); }));
      this._setupGuestSelectors();
    }
    if (this._page === "assistant" || view === "data-memory/conversations") getConfigurationEditor()?.bindConfiguration(this);
    if (view === "usage-maintenance/retention") getRouteFeature(view)?.bindRetentionSettings(this);
    if (view === "assistant/prompt-context") this._hydrateExposedAttributes();
    if (view === "usage-maintenance/backup-restore") getRouteFeature(view)?.bindBackupTransfer(this);
    if (view === "capabilities/functions") getConfigurationTools()?.bindTools(this);
    if (view === "capabilities/request-rules") getRouteFeature(view)?.bindRequestRules(this);
    if (view === "overview") bindOverview(this);
    if (view === "guide") bindGuide(this);
    if (this._pendingSettingFocus && root.querySelector(`#${CSS.escape(this._pendingSettingFocus)}`)) {
      const target = this._pendingSettingFocus;
      this._pendingSettingFocus = null;
      requestAnimationFrame(() => { const element = this.shadowRoot.querySelector(`#${target}`); element?.scrollIntoView({behavior:"smooth", block:"start"}); (element?.querySelector("input,select,textarea,button") || element)?.focus?.(); });
    }
    if (["data-memory/memories", "capabilities/guest-mode"].includes(view)) {
      getRouteFeature("memory-browser")?.bindMemoryBrowser(this);
    }
    if (view === "data-memory/memories") getRouteFeature(view)?.bindTemporaryMemory(this);
    if (view === "data-memory/memory-settings") getRouteFeature(view)?.bindMemorySettings(this);
    if (["capabilities/home-assistant", "capabilities/web-skills"].includes(view)) {
      getRouteFeature("capabilities")?.bindCapabilities(this);
    }
    if (view === "assistant/voice") getRouteFeature(view)?.bindVoiceIdentityCore(this);
    if (view === "capabilities/quiet-hours") getRouteFeature(view)?.bindQuietHours(this);
    if (view === "usage-maintenance/usage") {
      getRouteFeature(view)?.bindUsageDiagnostics(this);
    }
  }

  async _hydrateExposedAttributes() {
    const target = this.shadowRoot?.querySelector("[data-exposed-feature]");
    if (!target) return;
    const agentId = this._agentId;
    const revision = this._configData?.revision;
    try {
      const [feature] = await Promise.all([
        import("./exposed-attributes-ui.js"),
        this._loadConfigurationLiveMetadata("assistant/prompt-context"),
      ]);
      if (!target.isConnected || agentId !== this._agentId
          || revision !== this._configData?.revision
          || this._viewKey() !== "assistant/prompt-context") return;
      target.outerHTML = feature.renderExposedAttributeSettings(this);
      feature.bindExposedAttributeSettings(this);
    } catch (error) {
      if (target.isConnected && agentId === this._agentId
          && this._viewKey() === "assistant/prompt-context") {
        target.querySelector(".help").textContent = `Unable to load entity choices: ${error.message || String(error)}`;
      }
    }
  }

  _activate(element, callback) {
    element.addEventListener("click", (event) => { if (!event.target.closest("button") || event.currentTarget === event.target) callback(); });
    element.addEventListener("keydown", (event) => { if ((event.key === "Enter" || event.key === " ") && event.target === element) { event.preventDefault(); callback(); } });
  }

  _updateVisibleList() {
    if (this._viewKey() === "data-memory/knowledge") return getRouteFeature("data-memory/knowledge")?.filterKnowledge(this);
    if (this._viewKey() === "data-memory/memories") return this._memoryKind === "persistent"
      ? getRouteFeature("memory-browser")?.filterPersistentMemories(this)
      : getRouteFeature("data-memory/memories")?.filterTemporaryMemories(this);
    const query = this._query.trim().toLocaleLowerCase();
    this.shadowRoot.querySelectorAll(".list-card").forEach((card) => {
      card.hidden = query && !card.textContent.toLocaleLowerCase().includes(query);
    });
  }

  async _openKnowledge(sourceId = null) {
    this._ensureOwnedDialog("knowledge-dialog");
    const availability = this.shadowRoot?.querySelector("#knowledge-source-enabled");
    if (availability) availability.checked = true;
    const root = this.shadowRoot;
    const dialog = root.querySelector("#knowledge-dialog");
    const loadToken = (this._knowledgeLoadToken || 0) + 1;
    this._knowledgeLoadToken = loadToken;
    this._editingSource = null;
    this._knowledgeMode = sourceId ? "edit-loading" : "create";
    this._editorInitial = null;
    this._setDialogError("knowledge", "");
    root.querySelector("#knowledge-title").value = "";
    root.querySelector("#knowledge-description").value = "";
    root.querySelector("#knowledge-content").value = "";
    root.querySelector("#knowledge-delete").hidden = true;
    root.querySelector("#knowledge-dialog-title").textContent = sourceId ? "Loading source…" : "Add Knowledge source";
    this._setKnowledgeEditorDisabled(Boolean(sourceId));
    this._editorKind = "knowledge";
    dialog.showModal();
    if (sourceId) {
      try {
        const response = await this._call("knowledge", "get", { source_id: sourceId });
        if (this._knowledgeLoadToken !== loadToken || !dialog.open) return;
        this._editingSource = response.source;
        if (availability) availability.checked = response.source.enabled !== false;
        this._knowledgeMode = "edit";
        root.querySelector("#knowledge-title").value = response.source.title || "";
        root.querySelector("#knowledge-description").value = response.source.description || "";
        root.querySelector("#knowledge-content").value = response.source.content || "";
        root.querySelector("#knowledge-delete").hidden = false;
        root.querySelector("#knowledge-dialog-title").textContent = "Edit Knowledge source";
        this._setKnowledgeEditorDisabled(false);
      } catch (err) {
        if (this._knowledgeLoadToken !== loadToken || !dialog.open) return;
        this._knowledgeMode = "edit-error";
        this._setDialogError("knowledge", `Unable to load source: ${err.message || String(err)}`);
        root.querySelector("#knowledge-dialog-title").textContent = "Unable to load source";
      }
    }
    if (["create", "edit"].includes(this._knowledgeMode)) {
      this._editorInitial = this._knowledgeValues();
    }
    this._updateKnowledgeCounter();
    // The dialog and fields are ready; a deferred focus can steal later input.
    (this._knowledgeMode === "edit-error" ? dialog.querySelector(".close-editor") : root.querySelector("#knowledge-title")).focus();
  }

  async _openMemory(memoryId = null) {
    this._ensureOwnedDialog("memory-dialog");
    const root = this.shadowRoot;
    const memory = getRouteFeature("memory-browser")?.findPersistentMemory(this, memoryId) || null;
    this._editingMemory = memory ? {...memory} : null;
    this._memoryEditorScope = this._scopeId;
    this._memoryEditorAgent = this._agentId;
    this._editorKind = "memory";
    root.querySelector("#memory-dialog-title").textContent = memory ? "Edit memory" : "Add memory";
    root.querySelector("#memory-content").value = memory?.content || "";
    root.querySelector("#memory-category").value = memory?.category || "general";
    getRouteFeature("data-memory/memories").populateMemoryMetadata(this, memory);
    root.querySelector("#memory-delete").hidden = !memory;
    root.querySelector("#memory-meta").textContent = memory ? [memory.source, memory.created_at ? `Created ${this._formatDate(memory.created_at)}` : "", memory.updated_at ? `Updated ${this._formatDate(memory.updated_at)}` : ""].filter(Boolean).join(" · ") : "Categories help organise memories.";
    this._setDialogError("memory", "");
    this._editorInitial = this._memoryValues();
    root.querySelector("#memory-dialog").showModal();
    root.querySelector("#memory-content").focus();
  }

  async _requestEditorClose() {
    return this._confirmEditorClose(this.shadowRoot.querySelector(`#${this._editorKind}-dialog`));
  }

  _setKnowledgeEditorDisabled(disabled) {
    const availability = this.shadowRoot?.querySelector("#knowledge-source-enabled");
    if (availability) availability.disabled = disabled;
    const root = this.shadowRoot;
    ["#knowledge-title", "#knowledge-description", "#knowledge-content", "#knowledge-save"].forEach((selector) => {
      root.querySelector(selector).disabled = disabled;
    });
  }

  _knowledgeValues() {
    const root = this.shadowRoot;
    return { enabled: root.querySelector("#knowledge-source-enabled")?.checked ?? true, title: root.querySelector("#knowledge-title").value, description: root.querySelector("#knowledge-description").value, content: root.querySelector("#knowledge-content").value };
  }

  _memoryValues() {
    const root = this.shadowRoot;
    return { content: root.querySelector("#memory-content").value, category: root.querySelector("#memory-category").value, ...getRouteFeature("data-memory/memories").memoryMetadataValues(this) };
  }

  _retainedMutationOwner() {
    return {agent: this._agentId, entry: this._selectedAgent()?.entry_id,
      user: this._hass?.user?.id, route: this._viewKey(), scope: this._scopeId,
      kind: this._memoryKind, loadToken: this._loadToken};
  }

  _ownsRetainedMutation(owner) {
    return this._agentId === owner.agent && this._selectedAgent()?.entry_id === owner.entry
      && this._hass?.user?.id === owner.user && this._viewKey() === owner.route
      && this._scopeId === owner.scope && this._memoryKind === owner.kind
      && this._loadToken === owner.loadToken;
  }

  _patchScopeCount(scopeId, field, delta) {
    const scopes = this._data?.scopes;
    if (!Array.isArray(scopes)) return;
    this._data.scopes = scopes.map(scope => scope.scope_id === scopeId
      ? {...scope, [field]: Math.max(0, Number(scope[field] || 0) + delta)} : scope);
    const key = this._scopeCatalogKey();
    if (key) {
      this._scopeCatalogCache.set(key, this._data.scopes);
      this._eocScopeCatalogTimes.set(key, Date.now());
      this._scopeCatalogVisitKey = key;
    }
  }

  async _saveKnowledge() {
    const button = this.shadowRoot.querySelector("#knowledge-save");
    if (button.disabled || !["create", "edit"].includes(this._knowledgeMode)) return;
    const values = this._knowledgeValues();
    const editing = this._knowledgeMode === "edit";
    const owner = this._retainedMutationOwner();
    this._setSaving(button, true);
    try {
      const response = await this._call("knowledge", editing ? "update" : "create", { ...(editing ? { source_id: this._editingSource.source_id, expected_revision: this._editingSource.updated_at } : {}), ...values });
      if (!this._ownsRetainedMutation(owner)) return;
      if (!getRouteFeature("data-memory/knowledge")?.applyKnowledgeMutation(this, response)) throw new Error("The saved source response was incomplete.");
      this.shadowRoot.querySelector("#knowledge-dialog").close();
      writeSectionCache(this, this._sectionCacheKey(), this._result);
      const agent = this._selectedAgent();
      if (agent && Number.isFinite(response?.feature_status?.source_count)) agent.knowledge_source_count = response.feature_status.source_count;
      this._render();
      this._toast(editing ? "Knowledge source updated" : "Knowledge source saved");
    } catch (err) {
      this._setDialogError("knowledge", `Unable to save source: ${err.message || String(err)}`);
    } finally { this._setSaving(button, false); }
  }

  async _saveMemory() {
    const button = this.shadowRoot.querySelector("#memory-save");
    if (button.disabled) return;
    const values = this._memoryValues();
    const owner = this._retainedMutationOwner();
    this._setSaving(button, true);
    try {
      if (this._memoryEditorAgent !== this._agentId || this._memoryEditorScope !== this._scopeId) throw new Error("The selected agent or scope changed. Close and reopen this editor.");
      if (this._editingMemory && !this._editingMemory.revision) throw new Error("Refresh the Memory list and reopen this editor before saving.");
      const response = await this._call("memories", this._editingMemory ? "update" : "add", {
        scope_id: this._memoryEditorScope,
        ...(this._editingMemory ? {memory_id: this._editingMemory.memory_id, expected_revision: this._editingMemory.revision} : {}),
        ...getRouteFeature("data-memory/memories").memoryMutationValues(values, this._editingMemory),
      });
      if (!this._ownsRetainedMutation(owner)) return;
      if (!getRouteFeature("data-memory/memories")?.applyPersistentMemoryMutation(this, response, {sourceScope: owner.scope})) throw new Error("The saved memory response was incomplete.");
      this.shadowRoot.querySelector("#memory-dialog").close();
      if (response.status === "created") this._patchScopeCount(response.scope_id, "memory_count", 1);
      else if (this._editingMemory && response.scope_id !== owner.scope) {
        this._patchScopeCount(owner.scope, "memory_count", -1);
        this._patchScopeCount(response.scope_id, "memory_count", 1);
      }
      this._patchScopeCount(owner.scope, "memory_count", 0);
      if (response.status === "created") this._selectedAgent().memory_count = Number(this._selectedAgent().memory_count || 0) + 1;
      this._render();
      this._toast(this._editingMemory ? "Memory updated" : "Memory added");
    } catch (err) {
      this._setDialogError("memory", `Unable to save memory: ${err.message || String(err)}`);
    } finally { this._setSaving(button, false); }
  }

  async _deleteSource(sourceId, fromDialog = false) {
    if (!sourceId || !await this._confirm("Delete Knowledge source?", "This permanently removes the selected source from this agent's local Knowledge Library.", "Delete")) return;
    const owner = this._retainedMutationOwner();
    try {
      const response = await this._call("knowledge", "delete", { source_id: sourceId, confirm: true });
      if (!this._ownsRetainedMutation(owner)) return;
      if (!getRouteFeature("data-memory/knowledge")?.applyKnowledgeMutation(this, response, sourceId)) throw new Error("The deleted source response was incomplete.");
      if (fromDialog) this.shadowRoot.querySelector("#knowledge-dialog").close();
      writeSectionCache(this, this._sectionCacheKey(), this._result);
      const agent = this._selectedAgent();
      if (agent && Number.isFinite(response?.feature_status?.source_count)) agent.knowledge_source_count = response.feature_status.source_count;
      this._render();
      this._toast("Knowledge source deleted");
    } catch (err) { this._toast(`Unable to delete source: ${err.message || String(err)}`, true); }
  }

  async _deleteMemory(memoryId, fromDialog = false) {
    if (!memoryId || !await this._confirm("Delete memory?", "This memory will be permanently removed from the selected scope.", "Delete")) return;
    const owner = this._retainedMutationOwner();
    try {
      const response = await this._call("memories", "delete", { scope_id: owner.scope, memory_id: memoryId });
      if (!this._ownsRetainedMutation(owner)) return;
      if (!getRouteFeature("data-memory/memories")?.applyPersistentMemoryMutation(this, response, {deletedId: memoryId, sourceScope: owner.scope})) throw new Error("The deleted memory response was incomplete.");
      if (fromDialog) this.shadowRoot.querySelector("#memory-dialog").close();
      this._patchScopeCount(owner.scope, "memory_count", -1);
      this._selectedAgent().memory_count = Math.max(0, Number(this._selectedAgent().memory_count || 0) - 1);
      this._render();
      this._toast("Memory deleted");
      this.shadowRoot.querySelector("#add-memory")?.focus({preventScroll: true});
    } catch (err) { this._toast(`Unable to delete memory: ${err.message || String(err)}`, true); }
  }

  _adjustConversationScopeCount(delta) {
    const patch = (scopes) => (scopes || []).map((scope) => scope.scope_id === this._scopeId
      ? {...scope, conversation_count: Math.max(0, Number(scope.conversation_count || 0) + delta)}
      : scope);
    if (this._data?.scopes) this._data.scopes = patch(this._data.scopes);
    const key = this._scopeCatalogKey("data-memory/conversations");
    if (key && this._scopeCatalogCache.has(key)) {
      this._scopeCatalogCache.set(key, patch(this._scopeCatalogCache.get(key)));
    }
  }

  async _deleteSession(sessionId) {
    const agentId = this._agentId;
    const scopeId = this._scopeId;
    const loadToken = this._loadToken;
    if (!await this._confirm("Delete conversation?", "This retained conversation and its turns will be permanently removed.", "Delete")) return;
    if (agentId !== this._agentId || scopeId !== this._scopeId || loadToken !== this._loadToken
        || this._viewKey() !== "data-memory/conversations") return;
    try {
      const response = await this._call("conversations", "delete", { scope_id: scopeId, session_id: sessionId });
      if (response?.deleted_sessions && agentId === this._agentId && scopeId === this._scopeId
          && loadToken === this._loadToken && this._viewKey() === "data-memory/conversations"
          && this._contentData?.sessions) {
        const current = this._contentData.sessions;
        const sessions = (current.sessions || []).filter((item) => item.session_id !== sessionId);
        const removed = (current.sessions || []).length - sessions.length;
        this._contentData = {
          ...this._contentData,
          sessions: {
            ...current,
            sessions,
            returned: Math.max(0, Number(current.returned ?? current.sessions?.length ?? 0) - removed),
            ...(this._eocHistoryMode !== "search" && Number.isFinite(Number(current.total))
              ? {total: Math.max(0, Number(current.total) - removed)}
              : {}),
          },
        };
        if (removed) this._adjustConversationScopeCount(-1);
        this._render();
      }
      this._toast("Conversation deleted");
    } catch (err) {
      this._toast(`Unable to delete conversation: ${err.message || String(err)}`, true);
    }
  }

  _openReassign(memoryId) {
    this._ensureOwnedDialog("reassign-dialog");
    this.shadowRoot.querySelector("#reassign-scope").innerHTML = this._scopeOptions("memories", true, true);
    this._reassignMemoryId = memoryId;
    this.shadowRoot.querySelector("#reassign-dialog").showModal();
  }

  async _saveReassign() {
    const target = this.shadowRoot.querySelector("#reassign-scope").value;
    if (!target) return;
    try {
      const result = await this._call("memories", "reassign_legacy", { scope_id: "__anonymous__", target_scope_id: target, memory_ids: [this._reassignMemoryId] });
      this.shadowRoot.querySelector("#reassign-dialog").close();
      await this._refreshAfterMutation();
      this._toast(`Reassigned ${formatUsageNumber(result.reassigned)} memory record${result.reassigned === 1 ? "" : "s"}`);
    } catch (err) { this._toast(`Unable to reassign memory: ${err.message || String(err)}`, true); }
  }

  async _clearUsageDetails() {
    if (!await this._confirm("Clear recent usage details?", "Request and run details will be removed. Daily, monthly, and lifetime totals remain.", "Clear details")) return;
    const owner = this._retainedMutationOwner();
    try {
      await this._call("usage", "clear_details", { confirm: true });
      if (!this._ownsRetainedMutation(owner)) return;
      this._usageDetailGeneration = (this._usageDetailGeneration || 0) + 1;
      this._result = {...this._result, summary: {...this._result.summary, latest: null},
        runs: {...this._result.runs, runs: [], total: 0},
        loading: {...this._result.loading, runs: false}};
      writeSectionCache(this, this._sectionCacheKey(), {
        summary: this._result.summary,
        days: this._result.days,
      });
      this.shadowRoot.querySelector("#usage-request-dialog")?.close();
      this._render();
      this._toast("Recent usage details cleared");
    }
    catch (err) { this._toast(`Unable to clear details: ${err.message || String(err)}`, true); }
  }

  _testAgent() {
    return getRouteFeature("status")?.testAgent(this);
  }

  async _refreshAfterMutation() {
    await this._loadSection(true);
  }

  _confirm(title, message, confirmLabel = "Confirm") {
    const subject = this._eocDecisionConfirmSubject || "";
    this._eocDecisionConfirmSubject = "";
    const root = this.shadowRoot;
    root.querySelector("#confirm-title").textContent = title;
    root.querySelector("#confirm-message").textContent = message;
    root.querySelector("#confirm-accept").textContent = confirmLabel;
    root.querySelector("#confirm-dialog").showModal();
    enhanceConfirmationScope(this, subject);
    return new Promise((resolve) => { this._confirmResolver = resolve; });
  }

  _resolveConfirm(value) {
    const dialog = this.shadowRoot.querySelector("#confirm-dialog");
    if (dialog?.open) dialog.close();
    const resolver = this._confirmResolver;
    this._confirmResolver = null;
    if (resolver) resolver(value);
  }

  _setSaving(button, saving, label = "Saving…") {
    if (!button) return;
    if (saving && !button.dataset.label) button.dataset.label = button.textContent;
    button.disabled = saving;
    button.textContent = saving ? label : button.dataset.label || "Save";
  }

  _setDialogError(kind, message) {
    const element = this.shadowRoot.querySelector(`#${kind}-error`);
    if (element) element.textContent = message;
  }

  _updateKnowledgeCounter() {
    const length = this.shadowRoot.querySelector("#knowledge-content")?.value.length || 0;
    this.shadowRoot.querySelector("#knowledge-counter").textContent = `${formatUsageNumber(length)} / ${formatUsageNumber(KNOWLEDGE_LIMIT)} characters`;
  }

  _toast(message, error = false) {
    const toast = this.shadowRoot.querySelector("#toast");
    if (!toast) return;
    toast.textContent = message;
    toast.className = `toast visible${error ? " toast-error" : ""}`;
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => { toast.className = "toast"; }, 5000);
  }

  _scopePicker() {
    if (this._viewKey() === "data-memory/memories" && this._memoryKind === "temporary") return getRouteFeature("data-memory/memories")?.renderTemporaryScopePicker(this);
    const memories = this._viewKey() === "data-memory/memories";
    const hasEmpty = (this._data?.scopes || []).some((scope) => (memories ? scope.memory_count : scope.conversation_count) === 0 && scope.scope_type === "user" && !scope.is_current_user);
    return `<section class="scope-bar" aria-label="${memories ? "Memory scope" : "Conversation scope"}"><span class="scope-title">${memories ? "Memory scope" : "Conversation scope"}</span><label><span>${memories ? "Show memories available to" : "Show conversations belonging to"}</span><select id="scope">${this._scopeOptions(memories ? "memories" : "conversations")}</select></label>${hasEmpty ? `<label class="show-empty"><input id="show-empty-scopes" type="checkbox" ${this._showEmptyScopes ? "checked" : ""}> Show users with no ${memories ? "memories" : "conversations"}</label>` : ""}${this._data?.is_admin ? `<small>You can view data for all users because you are an administrator.</small>` : ""}</section>`;
  }

  _scopeOptions(section, includeEmpty = this._showEmptyScopes, excludeLegacy = false) {
    const key = section === "memories" ? "memory_count" : "conversation_count";
    const scopes = [...(this._data?.scopes || [])].filter((scope) =>
      (!excludeLegacy || scope.scope_type !== "anonymous_legacy")
      && (scope.scope_id === this._scopeId || scope.is_current_user || scope[key] > 0 || scope.scope_type !== "user" || includeEmpty)
    );
    scopes.sort((a, b) => {
      if (a.scope_type === "anonymous_legacy") return 1;
      if (b.scope_type === "anonymous_legacy") return -1;
      if (a.is_current_user !== b.is_current_user) return a.is_current_user ? -1 : 1;
      const populated = Number(b[key] > 0) - Number(a[key] > 0);
      return populated || a.display_name.localeCompare(b.display_name);
    });
    return scopes.map((scope) => `<option value="${this._e(scope.scope_id)}" ${scope.scope_id === this._scopeId ? "selected" : ""}>${this._e(scope.display_name)} (${formatUsageNumber(scope[key] || 0)})${scope.is_current_user ? " · You" : ""}</option>`).join("");
  }

  _filtered(items, value) {
    const query = this._query.trim().toLocaleLowerCase();
    return query ? items.filter((item) => value(item).toLocaleLowerCase().includes(query)) : items;
  }

  _retentionOptions(selected) { return [0,7,30,90,180,365].map((value) => `<option value="${value}" ${value === selected ? "selected" : ""}>${value ? `${formatUsageNumber(value)} days` : "Disabled"}</option>`).join(""); }
  _metric(title, value, detail = "") { const display = typeof value === "number" && Number.isFinite(value) ? formatUsageNumber(value) : String(value); return `<article class="metric"><span>${this._e(title)}</span><strong>${this._e(display)}</strong>${detail ? `<small>${this._e(String(detail))}</small>` : ""}</article>`; }
  _toggle(id, label, checked) { return `<label class="toggle"><span>${this._e(label)}</span><input id="${id}" type="checkbox" role="switch" ${checked ? "checked" : ""}></label>`; }
  _table(headers, rows) { return `<div class="table"><table><thead><tr>${headers.map((header) => `<th>${this._e(header)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((value) => `<td>${this._e(typeof value === "number" && Number.isFinite(value) ? formatUsageNumber(value) : String(value))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`; }
  _loading() { return `<div class="loading" role="status"><span class="spinner"></span>Loading…</div>`; }
  _empty(message) { return `<div class="empty">${this._e(message)}</div>`; }
  _label(value) { return value[0].toUpperCase() + value.slice(1); }
  _titleCase(value) { return String(value || "").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
  _temporaryOwner(scopeId) {
    const known = (this._data?.scopes || []).find((scope) => scope.scope_id === scopeId);
    if (known) return known.display_name;
    if (String(scopeId || "").startsWith("device:")) return "Assist device";
    if (String(scopeId || "").startsWith("conversation:")) return "Current Assist conversation";
    return "Current user";
  }

  _formatDate(value) {
    return formatManagementTimestamp(value, this._hass?.config?.time_zone);
  }
  _e(value) { return String(value ?? "").replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]); }

}

if (!customElements.get("extended-openai-management-panel")) {
  customElements.define("extended-openai-management-panel", ExtendedOpenAIManagementPanel);
}
