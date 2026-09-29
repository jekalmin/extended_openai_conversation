import {saveConfiguration} from "./management-actions.js";
import {enhancementChanged} from "./management-enhancement-state.js";
import {UnsavedState, clone, same, draftScope, saveBarMarkup} from "./unsaved-state.js";
import {NAVIGATION} from "./frontend-navigation.js";

const GUEST = "capabilities/guest-mode";
const QUIET = "capabilities/quiet-hours";
const RULES = "capabilities/request-rules";
const settings = (result) => ({defaults: clone(result.defaults || {}), wording_groups: clone(result.wording_groups || [])});

export function pageCoordinator(panel) {
  if (!panel._unsavedState) panel._unsavedState = new UnsavedState();
  const state = panel._unsavedState;
  if (!state.scopes.has("configuration")) state.register("configuration", {
    dirty: () => Boolean(panel._configDirty),
    get pending() { return Boolean(panel._configurationSaving); },
    save: () => saveConfiguration(panel, panel.shadowRoot?.querySelector("#save-config")),
    destinations: () => [...(panel._configurationDirtyDestinations?.() || [])],
    // The selected agent owns this draft. A view change within the management
    // frontend does not change that owner; null represents leaving the frontend
    // or switching agents and must still pass the discard guard.
    owns: (destination) => {
      const [page, section] = destination?.split("/") || [];
      const route = NAVIGATION.find((item) => item.id === page);
      return Boolean(route && (!section || route.sections.some((item) => item.id === section))
        && panel._draftAgentId === panel._agentId);
    },
    discard: () => {
      if (panel._configData) {
        panel._draft = clone(panel._configData.config);
        panel._draftTitle = panel._configData.title;
      }
      panel._setConfigDirty(false);
    },
  });
  return state;
}

export function currentPageScope(panel) {
  return pageCoordinator(panel).scopes.get(panel._viewKey?.());
}

export function initializePageDraft(panel) {
  const state = pageCoordinator(panel), view = panel._viewKey?.(), result = panel._result;
  if (![GUEST, QUIET, RULES].includes(view) || !result || panel._busy || panel._error) return;
  const existing = state.scopes.get(view);
  if (existing?.agent === panel._agentId) {
    if (view === RULES && existing.result !== result && same(existing.baseline, settings(result))) existing.revision = result.revision;
    if (existing.result === result || existing.dirty() || existing.pending) return;
  }
  const field = view === GUEST ? "_guestDraft" : view === QUIET ? "_quietHoursDraft" : "_rulesSettingsDraft";
  const baseline = view === RULES ? settings(result) : result.config || {};
  panel[field] = clone(baseline);
  const scope = draftScope({
    baseline, read: () => panel[field], write: (value) => { panel[field] = value; },
    owns: (destination) => destination === view,
    destinations: () => [view],
    save: async (submitted, current) => {
      if (view === RULES) return saveRuleSettings(panel, submitted, current);
      const saved = await panel._call(view === GUEST ? "guest_mode" : "quiet_hours", view === GUEST ? "save_policy" : "update", {
        config: submitted, ...(current.revision ? {revision: current.revision} : {}),
      });
      panel._result = {...panel._result, ...saved, ...(view === GUEST ? {legacy_policy: false, migration_notice: null} : {})};
      if (view === GUEST) {
        const agentId = panel._agentId;
        const details = await panel._call("guest_mode", "details");
        if (panel._agentId === agentId && panel._viewKey?.() === GUEST && details?.policy) {
          panel._result = {...panel._result, policy: details.policy};
          const metrics = panel.shadowRoot?.querySelector(".metric-grid");
          if (metrics) metrics.innerHTML = [
            panel._metric("Guest-visible entities", details.policy.readable_entity_count ?? "—"),
            panel._metric("Guest-controllable entities", details.policy.controllable_entity_count ?? "—"),
            panel._metric("Guest functions", details.policy.configured_tool_count ?? "—"),
          ].join("");
        }
      }
      current.result = panel._result;
      current.revision = saved.revision;
      if (view === GUEST) { panel._guestMigrationReview = false; panel._guestStartingFresh = false; }
      return saved.config;
    },
  });
  Object.assign(scope, {agent: panel._agentId, result, revision: result.revision});
  state.register(view, scope);
}

async function saveRuleSettings(panel, submitted, scope) {
  const result = await panel._call("request_rules", "settings", {
    defaults: submitted.defaults,
    wording_groups: submitted.wording_groups,
    revision: scope.revision,
  });
  scope.baseline = settings(result);
  scope.revision = result.revision;
  panel._result = {
    ...panel._result,
    defaults: clone(result.defaults),
    wording_groups: clone(result.wording_groups),
    revision: result.revision,
  };
  scope.result = panel._result;
  return clone(scope.baseline);
}

export function readRuleSettings(panel) {
  if (panel._viewKey?.() !== RULES || !panel._rulesSettingsDraft) return;
  const root = panel.shadowRoot, q = (selector) => root.querySelector(selector);
  if (!q("#rules-default-word-forms")) return;
  panel._rulesSettingsDraft = {
    defaults: {
      word_forms: q("#rules-default-word-forms").checked,
      wording_alternatives: q("#rules-default-wording").checked,
      fuzzy: q("#rules-default-fuzzy").checked,
      fuzzy_threshold: Number(q("#rules-default-threshold").value),
    },
    wording_groups: [...root.querySelectorAll(".wording-group")].map((row) => ({
      canonical: row.querySelector(".wording-canonical").value.trim(),
      alternatives: row.querySelector(".wording-alternatives").value.split(",").map((value) => value.trim()).filter(Boolean),
    })),
  };
}

export function refreshPageSaveBar(panel) {
  const scope = currentPageScope(panel), root = panel.shadowRoot;
  if (!scope || !root?.querySelector) return;
  const dirty = scope.dirty();
  if (!enhancementChanged(panel, "page-save-bar", [scope, dirty, scope.pending])) return;
  let bar = root.querySelector(".save-bar");
  if (!dirty) { bar?.remove(); root.dispatchEvent?.(new Event("eoc-config-dirty-changed")); return; }
  if (!bar) {
    root.querySelector("main")?.insertAdjacentHTML("beforeend", saveBarMarkup(scope));
    bar = root.querySelector(".save-bar");
  }
  const save = bar?.querySelector("#save-page"), discard = bar?.querySelector("#discard-page");
  if (save) { save.disabled = scope.pending; save.textContent = scope.pending ? "Saving…" : "Save changes"; }
  if (discard) discard.disabled = scope.pending;
  root.dispatchEvent?.(new Event("eoc-config-dirty-changed"));
}

function syncRequestRulesSavedDom(panel) {
  const root = panel.shadowRoot;
  const draft = panel._rulesSettingsDraft;
  if (!root || !draft) return true;
  const q = (selector) => root.querySelector(selector);
  const defaults = draft.defaults || {};
  const controls = [
    ["#rules-default-word-forms", "checked", Boolean(defaults.word_forms)],
    ["#rules-default-wording", "checked", Boolean(defaults.wording_alternatives)],
    ["#rules-default-fuzzy", "checked", Boolean(defaults.fuzzy)],
    ["#rules-default-threshold", "value", String(defaults.fuzzy_threshold ?? 90)],
  ];
  for (const [selector, property, value] of controls) {
    const control = q(selector);
    if (control) control[property] = value;
  }
  const threshold = q("#rules-default-threshold");
  if (threshold) threshold.disabled = !defaults.fuzzy;

  const rows = [...root.querySelectorAll(".wording-group")];
  const groups = draft.wording_groups || [];
  if (rows.length !== groups.length) return false;
  rows.forEach((row, index) => {
    const group = groups[index] || {};
    const canonical = row.querySelector(".wording-canonical");
    const alternatives = row.querySelector(".wording-alternatives");
    if (canonical) canonical.value = group.canonical || "";
    if (alternatives) alternatives.value = (group.alternatives || []).join(", ");
  });
  return true;
}

function syncQuietHoursSavedDom(panel) {
  const root = panel.shadowRoot;
  const result = panel._result || {};
  const config = panel._quietHoursDraft || result.config || {};
  if (!root) return;
  const enabled = root.querySelector("#qh-enabled");
  const start = root.querySelector("#qh-start");
  const end = root.querySelector("#qh-end");
  const volume = root.querySelector("#qh-volume");
  const volumeValue = root.querySelector("#qh-volume-value");
  const wake = root.querySelector("#qh-wake");
  const maxPercent = Math.round(Number(config.max_volume ?? 0.2) * 100);
  if (enabled) enabled.checked = Boolean(config.enabled);
  if (start) start.value = config.start || "22:00";
  if (end) end.value = config.end || "07:00";
  if (volume) volume.value = String(maxPercent);
  if (volumeValue) volumeValue.textContent = `${maxPercent}%`;
  if (wake) wake.value = config.wake_sound || "off";

  const status = root.querySelector(".qh-status");
  if (status) {
    const title = result.active
      ? "Quiet Hours active now"
      : config.enabled ? "Outside Quiet Hours" : "Quiet Hours schedule disabled";
    const detail = config.enabled
      ? `${config.start || "22:00"}–${config.end || "07:00"} every day`
      : "The saved schedule is currently turned off.";
    const copy = status.querySelector("div");
    const strong = copy?.querySelector("strong");
    const small = copy?.querySelector("small");
    const badge = status.querySelector(":scope > span");
    if (strong) strong.textContent = title;
    if (small) small.textContent = detail;
    if (badge) {
      badge.textContent = result.active ? "Active" : "Inactive";
      badge.className = result.active ? "availability-badge" : "disabled-badge";
    }
  }
}

function syncGuestSavedDom(panel) {
  const root = panel.shadowRoot;
  root?.querySelector(".legacy-migration")?.remove();
}

function syncSavedPageDom(panel) {
  const view = panel._viewKey?.();
  if (view === RULES) return syncRequestRulesSavedDom(panel);
  if (view === QUIET) syncQuietHoursSavedDom(panel);
  if (view === GUEST) syncGuestSavedDom(panel);
  return true;
}

function renderDiscardedDraft(panel) {
  panel._eocMainMarkup = null;
  panel._render();
}

export async function savePageChanges(panel) {
  const scope = currentPageScope(panel);
  if (!scope || scope.pending || !scope.dirty()) return false;
  const operation = scope.save();
  refreshPageSaveBar(panel);
  try {
    await operation;
    // Keep the live route DOM when the saved structure is unchanged. The focused
    // control, details state, selection, and scroll position then remain native.
    // Fall back only if a future backend normalization changes editor structure.
    if (!syncSavedPageDom(panel)) panel._render();
    panel._toast("Changes saved");
    return true;
  } catch (err) {
    panel._toast(`Unable to save changes: ${err.message || String(err)}`, true);
    return false;
  } finally { refreshPageSaveBar(panel); }
}

export function bindPageDrafts(panel) {
  const root = panel.shadowRoot;
  if (!root || root.__eocPageDraftsBound) return;
  root.__eocPageDraftsBound = true;
  const sync = () => {
    if (![GUEST, QUIET, RULES].includes(panel._viewKey?.())) return;
    queueMicrotask(() => {
      if (![GUEST, QUIET, RULES].includes(panel._viewKey?.())) return;
      readRuleSettings(panel);
      refreshPageSaveBar(panel);
    });
  };
  for (const type of ["input", "change", "value-changed"]) root.addEventListener(type, sync);
  root.addEventListener("click", (event) => {
    const button = event.target?.closest?.("button");
    if (button?.id === "save-page") void savePageChanges(panel);
    if (button?.id === "discard-page") {
      const scope = currentPageScope(panel);
      if (scope?.pending) return;
      scope?.discard();
      renderDiscardedDraft(panel);
    }
    if (button?.matches("#wording-add,.wording-remove")) sync();
  });
}
