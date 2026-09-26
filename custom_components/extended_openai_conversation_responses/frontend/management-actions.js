import {same} from "./unsaved-state.js";
const clone = (value) => JSON.parse(JSON.stringify(value));

function fieldErrorKey(key) {
  return key === "title" ? "__title" : key;
}

function showErrors(panel, errors = {}) {
  const root = panel.shadowRoot;
  root.querySelectorAll(".field-error").forEach((item) => { item.textContent = ""; });
  Object.entries(errors).forEach(([key, message]) => {
    const mappedKey = fieldErrorKey(key);
    const escaped = CSS.escape(mappedKey);
    const fallback = CSS.escape(fieldErrorKey(key.split("[")[0]));
    const target = root.querySelector(`[data-error="${escaped}"]`) || root.querySelector(`[data-error="${fallback}"]`);
    if (target) target.textContent = message;
  });
}

function normalizeGuestModeTimestamp(value) {
  if (typeof value !== "string" || !value) return value;
  if (/(?:z|[+-]\d{2}:\d{2})$/i.test(value)) return value;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toISOString();
}

function validatedImportMatches(validatedDocument, currentDocument) {
  return typeof validatedDocument === "string" && validatedDocument === currentDocument;
}

function setControlPending(panel, control, pending) {
  if (!control) return;
  if (control.tagName === "BUTTON" && typeof panel._setSaving === "function") {
    panel._setSaving(control, pending);
    return;
  }
  control.disabled = pending;
}

async function runFrontendMutation(panel, control, label, operation) {
  if (!control || control.dataset?.eocMutationPending === "true") return false;
  control.dataset.eocMutationPending = "true";
  setControlPending(panel, control, true);
  try {
    await operation();
    return true;
  } catch (err) {
    panel._toast(`Unable to ${label}: ${err.message || String(err)}`, true);
    return false;
  } finally {
    delete control.dataset.eocMutationPending;
    setControlPending(panel, control, false);
  }
}

export async function saveConfigurationAcrossRestart(panel, payload, submitted, submittedTitle) {
  const baseline = panel._configData;
  const agentId = panel._agentId;
  try {
    return await panel._call("configuration", "save", payload);
  } catch (error) {
    if (!String(error?.message || error).includes("Configuration changed in another tab") ||
        !baseline?.server_epoch || panel._configData !== baseline) throw error;
    const latest = await panel._call("configuration", "get");
    if (latest?.server_epoch === baseline.server_epoch || !latest?.server_epoch ||
        latest.title !== baseline.title || !same(latest.config, baseline.config) ||
        panel._agentId !== agentId || !same(panel._draft, submitted) ||
        panel._draftTitle !== submittedTitle) throw error;
    // A new HA process lost the in-memory revision lineage. The persisted
    // baseline is unchanged, so the draft can safely use its fresh revision.
    return panel._call("configuration", "save", {...payload, revision: latest.revision});
  }
}

async function saveConfiguration(panel, button) {
  if (!panel._draft || !panel._selectedAgent?.() || panel._configurationSaving) return;
  const submitted = clone(panel._draft), submittedTitle = panel._draftTitle;
  panel._syncConfigDirty?.();
  const dirty = panel._eocDirtyConfigKeys || new Set();
  const changed = Object.fromEntries([...dirty].filter(key => key !== "__title" && Object.hasOwn(submitted, key)).map(key => [key, submitted[key]]));
  panel._configurationSaving = true;
  panel._setSaving(button, true);
  try {
    const result = await saveConfigurationAcrossRestart(panel, {
      config: changed,
      ...(dirty.has("__title") ? {title: submittedTitle} : {}),
      revision: panel._configData?.revision,
    }, submitted, submittedTitle);
    showErrors(panel, result.errors || {});
    if (!result.valid) {
      panel._toast("Fix the highlighted configuration errors", true);
      return;
    }

    const {valid: _valid, errors: _errors, agent, ...saved} = result;
    panel._configData = {...panel._configData, ...saved};
    panel._configDataStale = false;
    panel._result = panel._configData;
    panel._rememberCleanConfiguration?.(panel._configData);
    if (same(panel._draft, submitted)) panel._draft = clone(saved.config);
    if (panel._draftTitle === submittedTitle) panel._draftTitle = saved.title;
    panel._draftAgentId = panel._agentId;
    if (agent) Object.assign(panel._selectedAgent(), agent);
    panel._syncConfigDirty();
    panel._toast("Changes saved");
    panel._render();
  } catch (err) {
    panel._toast(`Unable to save configuration: ${err.message || String(err)}`, true);
  } finally {
    panel._configurationSaving = false;
    panel._setSaving(panel.shadowRoot.querySelector("#save-config") || button, false);
    const discard = panel.shadowRoot.querySelector("#revert-config");
    if (discard) discard.disabled = false;
  }
}

export function bindSingleRequestSave(panel) {
  const root = panel.shadowRoot;
  if (root.__eocSingleRequestSaveBound) return;
  root.__eocSingleRequestSaveBound = true;
  root.addEventListener("click", (event) => {
    if (event.target?.closest?.("#revert-config") && panel._configurationSaving) { event.preventDefault(); event.stopImmediatePropagation(); return; }
    const button = event.target?.closest?.("#save-config");
    if (!button) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const scope = panel._unsavedState?.scopes.get("configuration");
    void (scope ? scope.save() : saveConfiguration(panel, button));
  }, true);
}

export function bindFrontendCorrectness(panel) {
  const root = panel.shadowRoot;
  if (root.__eocFrontendCorrectnessBound) return;
  root.__eocFrontendCorrectnessBound = true;

  root.addEventListener("input", (event) => {
    const input = event.target;
    if (input?.id !== "import-document") return;
    const hadPreview = typeof panel._importDocument === "string";
    panel._importDocument = null;
    const apply = root.querySelector("#import-apply");
    if (apply) apply.disabled = true;
    if (hadPreview) {
      const summary = root.querySelector("#import-summary");
      if (summary) summary.textContent = "Document changed. Validate & preview again before importing.";
    }
  }, true);

  root.addEventListener("click", (event) => {
    const button = event.target?.closest?.("button");
    if (!button) return;

    if (button.id === "import-preview") {
      panel._importDocument = null;
      const apply = root.querySelector("#import-apply");
      if (apply) apply.disabled = true;
      return;
    }

    if (button.id === "import-apply") {
      const current = root.querySelector("#import-document")?.value ?? "";
      if (validatedImportMatches(panel._importDocument, current)) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      button.disabled = true;
      panel._toast("Validate & preview the current import document before importing it", true);
      return;
    }




  }, true);



  root.addEventListener("submit", (event) => {
    if (event.target?.id !== "rule-form" || !panel._eocRuleSavePromise) return;
    event.preventDefault();
    event.stopImmediatePropagation();
  }, true);
}


export {fieldErrorKey, normalizeGuestModeTimestamp, runFrontendMutation, validatedImportMatches, setControlPending, saveConfiguration};


export const MUTATIONS = new Map([
  ["backup", new Set(["restore"])],
  ["quiet_hours", new Set(["update"])],
  ["configuration", new Set(["update", "save"])],
  ["conversations", new Set(["delete", "end_active"])],
  ["guest_mode", new Set(["save_policy", "update", "disable"])],
  ["knowledge", new Set(["create", "update", "delete"])],
  ["memories", new Set(["add", "update", "delete", "temporary_delete", "temporary_update", "reassign_legacy"])],
  ["request_rules", new Set(["settings", "defaults", "wording_groups", "groups", "create", "update", "delete", "duplicate", "move"])],
  ["tools", new Set(["save", "delete", "set_enabled", "save_group", "delete_group"])],
  ["usage", new Set(["clear_details"])],
]);

export function isAgentMutation(section, action) {
  return Boolean(MUTATIONS.get(section)?.has(action));
}

export function syncAgentPicker(panel) {
  for (const id of ["guest-now", "guest-update", "guest-disable"]) {
    const button = panel.shadowRoot?.querySelector?.(`#${id}`);
    if (button) button.disabled = Boolean(panel._guestOperation);
  }
  const picker = panel.shadowRoot?.querySelector?.("#agent");
  if (!picker) return;
  const pending = Number(panel._eocAgentMutations || 0);
  picker.disabled = pending > 0;
  if (pending > 0) picker.setAttribute("aria-busy", "true");
  else picker.removeAttribute("aria-busy");
}
