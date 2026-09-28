import {adoptKeyedElements, reconcileKeyedChildren, delegateCollectionActions, setText} from "./keyed-collection.js";
import {ensureTemporaryScope, memoryCollectionIdentity} from "./management-data-state.js";
import {embeddedFeatureStatusMarkup} from "./management-feature-status-core.js";
import {friendlySettingValue} from "./management-setting-metadata.js";
const CONTENT_LIMIT = 500;
const CATEGORY_LIMIT = 64;

function validOwnerScope(scope) {
  return scope?.scope_type === "user" || scope?.scope_type === "shared";
}

function ownerLabel(panel, ownerScopeId) {
  const known = (panel._data?.scopes || []).find((scope) => scope.scope_id === ownerScopeId);
  if (known) return known.display_name;
  if (ownerScopeId === "shared:household") return "Shared household";
  return String(ownerScopeId || "").startsWith("user:") ? "Personal" : "Unavailable";
}


function temporaryScopeOptions(panel) {
  return (panel._data?.scopes || [])
    .filter(validOwnerScope)
    .map((scope) => `<option value="${panel._e(scope.scope_id)}" ${scope.scope_id === panel._scopeId ? "selected" : ""}>${panel._e(scope.display_name)}</option>`)
    .join("");
}

const collections = new WeakMap();

function temporaryMemoryCard(panel, memory) {
  return `<article class="list-card" data-memory-id="${panel._e(memory.memory_id)}">
    <div class="card-main clickable edit-temporary-memory" tabindex="0" role="button" data-id="${panel._e(memory.memory_id)}">
      <p class="primary-copy">${panel._e(memory.content)}</p>
      <p class="meta">${panel._e(memory.category)} · ${panel._e(ownerLabel(panel, memory.owner_scope_id))} · Expires ${panel._e(panel._formatDate(memory.expires_at))}</p>
    </div>
    <div class="actions"><button type="button" class="secondary edit-temporary-memory" data-id="${panel._e(memory.memory_id)}">Edit</button><button type="button" class="danger delete-temporary" data-id="${panel._e(memory.memory_id)}">Delete</button></div>
  </article>`;
}

function temporaryDiagnostics(panel) {
  const stats = panel._result?.stats || {};
  const pruned = Number(stats.invalid_owner_records_pruned || 0);
  const overflow = Number(stats.startup_overflow_records_pruned || 0);
  return pruned || overflow
    ? `<p class="help">Startup cleanup removed ${panel._e(String(pruned))} record(s) with invalid legacy ownership and ${panel._e(String(overflow))} record(s) above the 100-record ceiling.</p>` : "";
}

function temporaryMemoryStatus(panel) {
  const mode = panel._selectedAgent()?.temporary_memory || "off";
  const descriptions = {
    off: "The assistant does not create new short-term memories automatically.",
    balanced: "The assistant saves useful temporary details when they are clearly relevant.",
    eager: "The assistant saves useful temporary details more readily.",
  };
  return embeddedFeatureStatusMarkup(panel, "Short-term memory", {
    state: mode === "off" ? "disabled" : "enabled",
    label: friendlySettingValue("temporary_memory", mode) || mode,
    summary: descriptions[mode] || "Controls how readily the assistant saves useful temporary details.",
  }, {page:"data-memory", subsection:"memory-settings", label:"Configure"});
}

function renderTemporaryMemories(panel) {
  const items = panel._result?.memories || [];
  return `<section class="content-card" data-temporary-memories data-collection-identity="${panel._e(memoryCollectionIdentity(panel))}">
    <div class="section-heading"><div><h2>Memories</h2><p>Review short-term details saved for a Personal or Shared scope. <button type="button" class="guide-topic-link guide-link" data-guide-topic="memory">Learn more</button></p></div><button type="button" class="danger" id="clear-temporary">Clear short-term memories</button></div>
    <div data-temporary-feature-status>${temporaryMemoryStatus(panel)}</div>
    <div class="config-jumps"><button type="button" class="secondary memory-kind" data-kind="persistent">Long-term</button><button type="button" class="secondary memory-kind" data-kind="temporary" disabled>Short-term</button></div>
    <p class="help">Once stored, short-term memories remain until their expiry time, even if short-term memory is later turned off. Conversation and device continuity do not determine ownership.</p>
    <div data-temporary-diagnostics>${temporaryDiagnostics(panel)}</div>
    <input id="list-search" class="search" type="search" value="${panel._e(panel._query)}" placeholder="Search memories" aria-label="Search memories">
    <div class="list memory-list">${items.map(memory => temporaryMemoryCard(panel, memory)).join("")}<div data-temporary-empty>${panel._empty("No short-term memories in this scope.")}</div></div>
  </section>`;
}

export function filterTemporaryMemories(panel) {
  const state = collections.get(panel);
  if (!state?.host.isConnected || state.identity !== memoryCollectionIdentity(panel)) return;
  const query = panel._query.trim().toLocaleLowerCase();
  let visible = 0;
  for (const memory of panel._result?.memories || []) {
    const node = state.cards.get(String(memory.memory_id))?.node;
    if (!node) continue;
    const show = `${memory.content || ""} ${memory.category || ""} ${memory.owner_scope_id || ""}`.toLocaleLowerCase().includes(query);
    if (node.hidden === show) node.hidden = !show;
    if (show) visible++;
  }
  state.empty.hidden = visible > 0;
  const message = query ? "No short-term memories match this filter." : "No short-term memories in this scope.";
  setText(state.empty.firstElementChild || state.empty, message);
}

export function hasTemporaryMemoryCollection(panel) {
  const state = collections.get(panel);
  return Boolean(state?.identity === memoryCollectionIdentity(panel) && state.host?.isConnected
    && state.host === panel.shadowRoot.querySelector("[data-temporary-memories]"));
}

export function reconcileTemporaryMemories(panel) {
  const state = collections.get(panel);
  if (!state || state.host !== panel.shadowRoot.querySelector("[data-temporary-memories]") || state.identity !== memoryCollectionIdentity(panel)) return false;
  reconcileKeyedChildren(state.list, state.cards, panel._result?.memories || [], memory => memory.memory_id,
    memory => JSON.stringify([memory.content, memory.category, memory.expires_at, ownerLabel(panel, memory.owner_scope_id)]),
    memory => temporaryMemoryCard(panel, memory), [state.empty]);
  const diagnostics = temporaryDiagnostics(panel);
  if (state.diagnostics !== diagnostics) { state.host.querySelector("[data-temporary-diagnostics]").innerHTML = diagnostics; state.diagnostics = diagnostics; }
  const status = temporaryMemoryStatus(panel);
  if (state.status !== status) { state.host.querySelector("[data-temporary-feature-status]").innerHTML = status; state.status = status; }
  filterTemporaryMemories(panel);
  return true;
}

function temporaryDialog(panel) {
  return `<dialog id="temporary-memory-dialog" class="editor-dialog" aria-labelledby="temporary-memory-dialog-title">
    <form id="temporary-memory-form">
      <div class="dialog-header"><h2 id="temporary-memory-dialog-title">Edit short-term memory</h2><button type="button" class="icon close-temporary-editor" aria-label="Close">×</button></div>
      <div class="dialog-body">
        <label>Memory<textarea id="temporary-memory-content" maxlength="${CONTENT_LIMIT}" required spellcheck="true"></textarea></label>
        <label>Category<input id="temporary-memory-category" maxlength="${CATEGORY_LIMIT}" required></label>
        <label>Expires at<input id="temporary-memory-expiry" type="text" required placeholder="2026-09-07T18:30:00+01:00"></label>
        <p class="help">Use an ISO 8601 date-time including its timezone. Expiry must remain in the future and within the existing one-year Temporary Memory limit.</p>
        <p id="temporary-memory-meta" class="meta"></p>
        <div id="temporary-memory-error" class="inline-error" role="alert"></div>
      </div>
      <div class="dialog-actions"><button type="button" id="temporary-memory-delete" class="danger">Delete</button><button type="button" class="secondary close-temporary-editor">Cancel</button><button type="submit" id="temporary-memory-save">Save</button></div>
    </form>
  </dialog>`;
}

function openTemporaryMemory(panel, memoryId) {
  const memory = (panel._result?.memories || []).find((item) => item.memory_id === memoryId);
  if (!memory) return;
  panel._ensureOwnedDialog("temporary-memory-dialog");
  bindTemporaryMemory(panel);
  panel._temporaryMemoryDraft = {
    memory_id: memory.memory_id,
    content: memory.content || "",
    category: memory.category || "general",
    expires_at: memory.expires_at || "",
    owner_scope_id: memory.owner_scope_id || panel._scopeId,
  };
  const dialog = panel.shadowRoot.querySelector("#temporary-memory-dialog");
  panel.shadowRoot.querySelector("#temporary-memory-content").value = panel._temporaryMemoryDraft.content;
  panel.shadowRoot.querySelector("#temporary-memory-category").value = panel._temporaryMemoryDraft.category;
  panel.shadowRoot.querySelector("#temporary-memory-expiry").value = panel._temporaryMemoryDraft.expires_at;
  panel.shadowRoot.querySelector("#temporary-memory-meta").textContent = `Owner: ${ownerLabel(panel, panel._temporaryMemoryDraft.owner_scope_id)}`;
  panel.shadowRoot.querySelector("#temporary-memory-error").textContent = "";
  dialog?.showModal();
}

export function temporaryMemoryDirty(panel) {
  const draft = panel._temporaryMemoryDraft;
  if (!draft) return false;
  return panel.shadowRoot.querySelector("#temporary-memory-content")?.value !== draft.content
    || panel.shadowRoot.querySelector("#temporary-memory-category")?.value !== draft.category
    || panel.shadowRoot.querySelector("#temporary-memory-expiry")?.value !== draft.expires_at;
}

async function closeTemporaryMemory(panel, force = false) {
  const dialog = panel.shadowRoot.querySelector("#temporary-memory-dialog");
  if (force) dialog?.close();
  else if (!await panel._confirmEditorClose(dialog)) return false;
  panel._temporaryMemoryDraft = null;
  return true;
}

async function saveTemporaryMemory(panel) {
  const draft = panel._temporaryMemoryDraft;
  if (!draft || panel._temporaryMemorySaving) return;
  const content = panel.shadowRoot.querySelector("#temporary-memory-content")?.value ?? "";
  const category = panel.shadowRoot.querySelector("#temporary-memory-category")?.value ?? "";
  const expiresAt = panel.shadowRoot.querySelector("#temporary-memory-expiry")?.value ?? "";
  const error = panel.shadowRoot.querySelector("#temporary-memory-error");
  const owner = panel._retainedMutationOwner();
  if (!content.trim() || !category.trim() || !expiresAt.trim()) {
    error.textContent = "Memory, category, and expiry are required.";
    return;
  }
  panel._temporaryMemorySaving = true;
  const save = panel.shadowRoot.querySelector("#temporary-memory-save");
  panel._setSaving(save, true);
  try {
    const response = await panel._call("memories", "temporary_update", {
      scope_id: owner.scope,
      memory_id: draft.memory_id,
      content,
      category,
      expires_at: expiresAt,
    });
    if (!panel._ownsRetainedMutation(owner)) return;
    if (!response?.memory?.memory_id) throw new Error("The saved memory response was incomplete.");
    panel._result = {...panel._result, memories: (panel._result.memories || []).map(item =>
      item.memory_id === response.memory.memory_id ? response.memory : item)};
    await closeTemporaryMemory(panel, true);
    panel._patchScopeCount(owner.scope, "temporary_memory_count", 0);
    panel._render();
    panel._toast("Short-term memory updated");
  } catch (err) {
    if (panel._ownsRetainedMutation(owner)) error.textContent = err.message || String(err);
  } finally { panel._temporaryMemorySaving = false; panel._setSaving(save, false); }
}

async function deleteTemporaryMemory(panel, memoryId, button = null) {
  if (!memoryId || !await panel._confirm(
    "Delete temporary memory?",
    "This short-lived fact will no longer be included in later requests.",
    "Delete",
  )) return false;
  const owner = panel._retainedMutationOwner();
  panel._setSaving(button, true, "Deleting…");
  try {
    const response = await panel._call("memories", "temporary_delete", {
      scope_id: owner.scope,
      memory_id: memoryId,
    });
    if (!panel._ownsRetainedMutation(owner)) return false;
    if (response?.deleted !== 1) return false;
    panel._result = {...panel._result, memories: (panel._result.memories || []).filter(item => item.memory_id !== memoryId)};
    if (panel._temporaryMemoryDraft?.memory_id === memoryId) {
      await closeTemporaryMemory(panel, true);
    }
    panel._patchScopeCount(owner.scope, "temporary_memory_count", -1);
    panel._render();
    panel._toast("Temporary memory deleted");
    return true;
  } catch (err) {
    panel._toast(`Unable to delete temporary memory: ${err.message || String(err)}`, true);
    return false;
  } finally {
    panel._setSaving(button, false);
  }
}

export function bindTemporaryMemory(panel) {
  if (panel._viewKey?.() !== "data-memory/memories" || panel._memoryKind !== "temporary") return;
  const host = panel.shadowRoot.querySelector("[data-temporary-memories]");
  if (!host) return;
  if (collections.get(panel)?.host !== host) {
    const list = host.querySelector(".memory-list");
    collections.set(panel, {host, list, identity: memoryCollectionIdentity(panel), cards: adoptKeyedElements(list, "[data-memory-id]", "memoryId"), empty: list.querySelector("[data-temporary-empty]"), diagnostics: temporaryDiagnostics(panel), status: temporaryMemoryStatus(panel)});
    reconcileTemporaryMemories(panel);
    host.querySelector("#list-search").addEventListener("input", event => { panel._query = event.target.value; filterTemporaryMemories(panel); });
  }
  delegateCollectionActions(host, ".edit-temporary-memory,.delete-temporary,.memory-kind,#clear-temporary", control => {
    if (control.matches("#clear-temporary")) { void clearTemporaryMemories(panel, control); return; }
    if (control.matches(".delete-temporary")) void deleteTemporaryMemory(panel, control.dataset.id, control);
    else if (control.matches(".memory-kind")) {
      panel._memoryKind = control.dataset.kind; panel._query = "";
      void panel._loadSection();
    } else openTemporaryMemory(panel, control.dataset.id);
  });
  const form = panel.shadowRoot.querySelector("#temporary-memory-form");
  if (!form || form.__eocTemporaryBound) return;
  form.__eocTemporaryBound = true;
  form.addEventListener("submit", event => { event.preventDefault(); void saveTemporaryMemory(panel); });
  form.addEventListener("click", event => {
    const control = event.target.closest("button");
    if (control?.matches(".close-temporary-editor")) void closeTemporaryMemory(panel, !control.classList.contains("icon"));
    if (control?.id === "temporary-memory-delete" && panel._temporaryMemoryDraft?.memory_id) void deleteTemporaryMemory(panel, panel._temporaryMemoryDraft.memory_id, control);
  });
}

export function renderTemporaryScopePicker(panel) {
  ensureTemporaryScope(panel);
  return `<section class="scope-bar"><label><span>Show short-term memories belonging to</span><select id="scope">${temporaryScopeOptions(panel)}</select></label>${panel._data?.is_admin ? `<small>Temporary Memory ownership is limited to Personal and Shared scopes.</small>` : ""}</section>`;
}

export {
  temporaryDialog,
  CATEGORY_LIMIT,
  CONTENT_LIMIT,
  ensureTemporaryScope,
  ownerLabel,
  renderTemporaryMemories,
  temporaryScopeOptions,
  validOwnerScope,
};

export async function clearTemporaryMemories(panel, button = null) {
  const scope = panel._scopeId, agent = panel._agentId;
  if (!await panel._confirm("Clear short-term memories?", `All short-term memories belonging to ${ownerLabel(panel, scope)} for the selected agent will be permanently removed. Search does not limit this action. Long-term memories are unchanged.`, "Clear memories")) return;
  if (scope !== panel._scopeId || agent !== panel._agentId || panel._memoryKind !== "temporary") return;
  panel._setSaving(button, true, "Clearing…");
  try {
    const owner = panel._retainedMutationOwner();
    const response = await panel._call("memories", "temporary_clear", {scope_id: scope, confirm: true});
    if (!panel._ownsRetainedMutation(owner)) return;
    if (!Number.isFinite(response?.deleted)) throw new Error("The clear response was incomplete.");
    panel._result = {...panel._result, memories: []};
    panel._patchScopeCount(scope, "temporary_memory_count", -response.deleted);
    panel._render();
    panel._toast("Short-term memories cleared");
  } catch (err) { panel._toast(`Unable to clear short-term memories: ${err.message || String(err)}`, true); }
  finally { panel._setSaving(button, false); }
}
