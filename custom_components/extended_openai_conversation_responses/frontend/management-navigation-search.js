import {enhancementChanged} from "./management-enhancement-state.js";
import {friendlySettingLabel, friendlySettingValue, settingSearchAliases} from "./management-setting-metadata.js";
import {pageMetadata, focusManagementSetting} from "./frontend-navigation.js";
import {SETTINGS_INDEX} from "./management-settings-index.js";

export function buildSettingsSearchProjection(settings = SETTINGS_INDEX) {
  return settings.map((item, index) => {
    const label = String(item.label || "");
    const description = String(item.description || "");
    const terms = String(item.terms || "");
    const configKey = String(item.configKey || "");
    return {
      item,
      index,
      label: label.toLowerCase(),
      haystack: `${label} ${description} ${terms} ${configKey} ${settingSearchAliases(configKey)}`.toLowerCase(),
    };
  });
}

export const SETTINGS_SEARCH_PROJECTION = buildSettingsSearchProjection();

export function searchProjectedSettings(query, projection = SETTINGS_SEARCH_PROJECTION) {
  const normalized = String(query || "").trim().toLowerCase();
  const terms = normalized.split(/\s+/).filter(Boolean);
  if (!terms.length) return [];
  return projection
    .map((entry) => {
      if (!terms.every((term) => entry.haystack.includes(term))) return null;
      const score = entry.label === normalized ? 0 : entry.label.startsWith(normalized) ? 1 : entry.label.includes(normalized) ? 2 : 3;
      return {item: entry.item, index: entry.index, score};
    })
    .filter(Boolean)
    .sort((a, b) => a.score - b.score || a.index - b.index)
    .map(({item}) => item);
}


function titleCase(value) {
  return String(value || "").replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function activeConfiguration(panel, explicitData = null) {
  if (panel._draft && panel._draftAgentId === panel._agentId) {
    return {data: panel._configData || explicitData || {}, config: panel._draft, title: panel._draftTitle, draft: true};
  }
  const cached = panel._settingsSearchConfigAgentId === panel._agentId ? panel._settingsSearchConfig : null;
  const data = explicitData || cached || panel._configData;
  if (!data?.config) return null;
  return {data, config: data.config, title: data.title, draft: false};
}

function optionLabel(data, key, value) {
  const option = data?.options?.[key]?.find((item) => String(typeof item === "string" ? item : item.value) === String(value));
  if (!option) return null;
  return typeof option === "string" ? titleCase(option) : option.label;
}

function countLabel(value, singular, plural) {
  let count = 0;
  if (Array.isArray(value)) count = value.length;
  else if (typeof value === "string") count = value.split(",").map((item) => item.trim()).filter(Boolean).length;
  else if (value && typeof value === "object") count = Object.keys(value).length;
  return `${count} ${count === 1 ? singular : plural}`;
}

export function settingCurrentState(item, panel, explicitData = null) {
  if (item.source === "guest-mode") {
    const state = panel?._selectedAgent?.()?.guest_mode?.state;
    return state ? {label:"Current", value:titleCase(state)} : null;
  }
  if (!item.configKey) return null;

  const active = activeConfiguration(panel, explicitData);
  if (!active) return null;
  if (item.capability && active.data?.model_capabilities?.[item.capability] === false) {
    return {label:"Current", value:"Not supported by current model"};
  }

  const value = item.configKey === "__title" ? active.title : active.config?.[item.configKey];
  if (value === undefined) return null;
  let display;
  const choice = friendlySettingValue(item.configKey, value) || optionLabel(active.data, item.configKey, value);
  if (choice) display = choice;
  else if (item.format === "boolean") display = value ? "On" : "Off";
  else if (item.format === "template") display = String(value || "").trim() ? "Custom" : "Default";
  else if (item.format === "prompt") display = `${String(value || "").length.toLocaleString()} characters`;
  else if (item.format === "mapping") display = countLabel(value, "assignment", "assignments");
  else if (item.format === "count") display = countLabel(value, item.singular || "item", item.plural || "items");
  else if (item.format === "text") display = String(value || "").trim() || "Not set";
  else if (typeof value === "boolean") display = value ? "On" : "Off";
  else if (Array.isArray(value)) display = countLabel(value, "item", "items");
  else display = `${value ?? "Not set"}${item.suffix || ""}`;
  return {label:active.draft ? "Current draft" : "Current", value:String(display)};
}

function visibleSettings(panel) {
  return searchProjectedSettings(panel._settingsSearchQuery).filter((item) => {
    if (!panel._canAccessView(item.page, item.section)) return false;
    if (item.configKey && panel._data?.is_admin === false) return false;
    return true;
  });
}

function pathLabel(item) {
  const page = pageMetadata(item.page);
  const section = page.sections.find((candidate) => candidate.id === item.section);
  return `${page.label}${section ? ` › ${section.label}` : ""}`;
}

export const SEARCH_DEBOUNCE_MS = 80;

function settingsResultsMarkup(panel) {
  const query = String(panel._settingsSearchQuery || "");
  const results = visibleSettings(panel);
  const configUnavailable = Boolean(
    query
      && panel._data?.is_admin !== false
      && !activeConfiguration(panel)
      && panel._settingsSearchConfigError
      && panel._settingsSearchConfigErrorAgentId === panel._agentId
  );
  const configLoading = Boolean(query && panel._data?.is_admin !== false && !activeConfiguration(panel) && panel._settingsSearchConfigLoading);
  const loadError = configUnavailable
    ? '<div class="settings-load-error" role="status"><span>Current setting values couldn’t be loaded.</span><button id="settings-search-retry" type="button">Retry</button></div>'
    : "";
  return query ? `${loadError}${results.map((item) => {
    const state = settingCurrentState(item, panel);
    if (state?.value === "Not supported by current model" && ["temperature", "top_p", "reasoning_effort", "service_tier"].includes(item.configKey)) {
      state.value = "Not used for current model";
    }
    const current = state
      ? `<span class="settings-current">${panel._e(state.label)}: ${panel._e(state.value)}</span>`
      : item.configKey && configLoading ? '<span class="settings-current settings-loading">Current value loading…</span>' : "";
    return `<button type="button" class="settings-result" role="option" data-page="${panel._e(item.page)}" data-subsection="${panel._e(item.section)}" data-target="${panel._e(item.target || "")}"><strong>${panel._e(friendlySettingLabel(item.configKey) || item.label)}</strong><span class="setting-path">${panel._e(pathLabel(item))}</span><small>${panel._e(item.description)}</small>${current}</button>`;
  }).join("") || '<p class="empty">No settings match.</p>'}` : "";
}

function searchMarkup(panel) {
  return `<div class="global-search eoc-global-search"><label><span class="search-label">Find a setting</span><input id="settings-search" type="search" value="${panel._e(panel._settingsSearchQuery || "")}" placeholder="Search settings by name or purpose" aria-label="Search all settings" autocomplete="off"></label><div class="search-results" role="listbox" aria-label="Settings search results" ${panel._settingsSearchQuery ? "" : "hidden"}>${settingsResultsMarkup(panel)}</div></div>`;
}

export function updateSettingsResults(panel) {
  const results = panel.shadowRoot?.querySelector(".eoc-global-search .search-results");
  if (!results) return;
  const query = String(panel._settingsSearchQuery || "");
  const markup = settingsResultsMarkup(panel);
  if (results._eocMarkup !== markup) {
    results.innerHTML = markup;
    results._eocMarkup = markup;
    panel._eocSearchResultsRevision = (panel._eocSearchResultsRevision || 0) + 1;
  }
  results.dataset.searchQuery = query;
  const hidden = !query;
  if (results.hidden !== hidden) results.hidden = hidden;
}

async function ensureSearchConfiguration(panel, {retry = false} = {}) {
  if (panel._data?.is_admin === false || !panel._settingsSearchQuery || activeConfiguration(panel)) return;
  if (!visibleSettings(panel).some((item) => item.configKey)) return;
  const agentId = panel._agentId;
  if (!agentId) return;
  if (panel._settingsSearchConfigAgentId === agentId && panel._settingsSearchConfig?.config) return;
  if (panel._settingsSearchConfigPromise && panel._settingsSearchConfigPromiseAgentId === agentId) return panel._settingsSearchConfigPromise;
  if (panel._settingsSearchConfigError && panel._settingsSearchConfigErrorAgentId === agentId && !retry) return;

  if (retry && panel._settingsSearchConfigErrorAgentId === agentId) {
    panel._settingsSearchConfigError = null;
    panel._settingsSearchConfigErrorAgentId = null;
  }
  panel._settingsSearchConfigLoading = true;
  panel._settingsSearchConfigPromiseAgentId = agentId;
  panel._settingsSearchConfigPromise = panel._call("configuration", "get")
    .then((data) => {
      if (panel._agentId !== agentId) return;
      if (!data?.config) throw new Error("Configuration response did not include config");
      panel._settingsSearchConfig = data;
      panel._settingsSearchConfigAgentId = agentId;
      panel._settingsSearchConfigError = null;
      panel._settingsSearchConfigErrorAgentId = null;
    })
    .catch(() => {
      if (panel._agentId !== agentId) return;
      panel._settingsSearchConfig = null;
      panel._settingsSearchConfigAgentId = null;
      panel._settingsSearchConfigError = true;
      panel._settingsSearchConfigErrorAgentId = agentId;
    })
    .finally(() => {
      if (panel._settingsSearchConfigPromiseAgentId === agentId) {
        panel._settingsSearchConfigLoading = false;
        panel._settingsSearchConfigPromise = null;
        panel._settingsSearchConfigPromiseAgentId = null;
        if (panel._settingsSearchQuery) {
          updateSettingsResults(panel);
        }
      }
    });
  if (panel._settingsSearchQuery) {
    updateSettingsResults(panel);
  }
  return panel._settingsSearchConfigPromise;
}

function bindSearch(panel, search) {
  if (!search || search.__eocSearchBound) return;
  search.__eocSearchBound = true;
  const input = search.querySelector("#settings-search");
  input.addEventListener("input", () => {
    panel._settingsSearchQuery = input.value;
    clearTimeout(panel._eocSettingsSearchTimer);
    panel._eocSettingsSearchTimer = setTimeout(() => {
      updateSettingsResults(panel);
      void ensureSearchConfiguration(panel);
    }, SEARCH_DEBOUNCE_MS);
  });
  search.addEventListener("click", async (event) => {
    if (event.target.closest("#settings-search-retry")) {
      void ensureSearchConfiguration(panel, {retry:true});
      return;
    }
    const button = event.target.closest(".settings-result");
    if (!button) return;
    const results = search.querySelector(".search-results");
    const currentQuery = String(panel._settingsSearchQuery || "");
    const {page, subsection, target} = button.dataset;
    if (results?.dataset.searchQuery !== currentQuery) {
      updateSettingsResults(panel);
    }
    panel._pendingSettingFocus = target;
    panel._settingsSearchQuery = "";
    input.value = "";
    updateSettingsResults(panel);
    await panel._navigate(page, subsection);
    // Cached same-page navigation does not rebind main content. Complete the
    // focus handoff here when the normal page binding did not consume it.
    if (target && panel._pendingSettingFocus === target
        && panel._page === page && panel._subsection === subsection) {
      const element = panel.shadowRoot.getElementById(target);
      if (element) {
        panel._pendingSettingFocus = null;
        focusManagementSetting(element);
      }
    }
  });
}

export function enhanceNavigationSearch(panel) {
  const root = panel.shadowRoot;
  if (!root) return;
  // Only values displayed by the current search can invalidate its results.
  // The input/debounce and asynchronous fetch paths still update independently.
  const searchState = panel._settingsSearchQuery ? JSON.stringify([
    visibleSettings(panel).map((item) => [item.target, settingCurrentState(item, panel)]),
    panel._settingsSearchConfigLoading, panel._settingsSearchConfigError,
    panel._settingsSearchConfigErrorAgentId,
  ]) : "";
  if (!enhancementChanged(panel, "navigation-search", [panel._page, panel._subsection, panel._agentId, panel._data?.is_admin, panel._settingsSearchQuery || "", searchState])) return;
  let search = root.querySelector(".eoc-global-search");
  const header = root.querySelector("header");
  if (!search && header) {
    header.insertAdjacentHTML("beforeend", searchMarkup(panel));
    search = header.querySelector(".eoc-global-search");
  }
  bindSearch(panel, search);
  const input = search?.querySelector("#settings-search");
  if (input && input.value !== (panel._settingsSearchQuery || "")) input.value = panel._settingsSearchQuery || "";
  updateSettingsResults(panel);

  if (panel._settingsSearchQuery) void ensureSearchConfiguration(panel);
}

export {ensureSearchConfiguration, searchMarkup, settingsResultsMarkup};
