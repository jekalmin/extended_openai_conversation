import {adoptKeyedElements, reconcileKeyedChildren, delegateCollectionActions, setText} from "./keyed-collection.js";
import {embeddedFeatureStatusMarkup, selectedFeatureStatus} from "./management-feature-status.js";
import {formatManagementTimestamp, browserState, memoryCollectionIdentity} from "./management-data-state.js";
export {prepareMemoryBrowser} from "./management-data-state.js";
export {formatManagementTimestamp} from "./management-data-state.js";
export const GUEST_EXCLUSION_KEYS = [
  "guest_excluded_labels", "guest_excluded_areas", "guest_excluded_domains", "guest_excluded_entities",
  "guest_control_excluded_labels", "guest_control_excluded_areas", "guest_control_excluded_domains", "guest_control_excluded_entities",
];

const MEMORY_PAGE_SIZE = 100;
const MEMORY_SEARCH_DEBOUNCE_MS = 250;

export function freshGuestPolicyDraft(config = {}) {
  const draft = JSON.parse(JSON.stringify(config));
  [...GUEST_EXCLUSION_KEYS, "guest_knowledge_source_ids", "guest_allowed_function_names", "guest_allowed_group_ids"]
    .forEach((key) => { draft[key] = []; });
  return Object.assign(draft, {
    guest_mode_enabled: true,
    guest_web_search: false,
    guest_separate_control_restrictions: false,
    guest_knowledge_policy: "off",
    guest_function_policy: "off",
    guest_shared_memory_policy: "off",
  });
}

export function renderGuestWebSearchSetting(config = {}) {
  return `<div class="guest-hosted-capabilities"><label class="toggle"><span>Allow hosted Web Search</span><input id="guest-web-search" type="checkbox" ${config.guest_web_search ? "checked" : ""}></label><p class="help">Off by default. When enabled, Guest Mode may expose hosted Web Search only when this agent's provider, API mode, and Web Search configuration support it.</p></div>`;
}


export function memorySearchProjection(memory) {
  return `${memory?.content ?? ""} ${memory?.category ?? ""} ${memory?.source ?? ""}`.toLocaleLowerCase();
}

const memoryCollections = new WeakMap();

function collectionState(panel) {
  const identity = memoryCollectionIdentity(panel);
  let state = memoryCollections.get(panel);
  if (!state || state.identity !== identity) {
    state = {identity, items: new Map(), result: null, host: null};
    memoryCollections.set(panel, state);
  }
  if (state.result !== panel._result) {
    // A route/mutation refresh is authoritative. Search pages, by contrast,
    // merge into this collection so filtering alone never destroys cards.
    state.items = new Map((panel._result?.memories || []).map(memory => [String(memory.memory_id), memory]));
    state.result = panel._result;
  }
  return state;
}

function indexMemories(panel) {
  browserState(panel).projections = new Map([...collectionState(panel).items].map(([id, memory]) => [id, memorySearchProjection(memory)]));
}

export function findPersistentMemory(panel, id) {
  return collectionState(panel).items.get(String(id));
}

export function applyPersistentMemoryMutation(panel, response, {deletedId = null, sourceScope = panel._scopeId} = {}) {
  if (!panel._result || !Array.isArray(panel._result.memories)) return false;
  if (deletedId && response?.deleted !== 1) return false;
  const memory = response?.memory;
  if (!deletedId && !memory?.memory_id) return false;
  const id = String(deletedId || memory.memory_id);
  const collection = collectionState(panel);
  const search = browserState(panel);
  clearTimeout(search.memorySearchTimer);
  search.memorySearchSequence++;
  const remainsHere = !deletedId && response.scope_id === sourceScope;
  if (remainsHere) collection.items.set(id, memory);
  else collection.items.delete(id);
  const query = browserState(panel).memoryQuery;
  const matchesQuery = !query || memorySearchProjection(memory).includes(query);
  const oldPage = panel._result.memories || [];
  const wasInPage = oldPage.some(item => String(item.memory_id) === id);
  let page = oldPage.filter(item => String(item.memory_id) !== id);
  if (remainsHere && matchesQuery) page.push(memory);
  page.sort((left, right) => String(right.updated_at).localeCompare(String(left.updated_at)));
  const limit = Number(panel._result.limit) || MEMORY_PAGE_SIZE;
  const hasMore = Boolean(panel._result.has_more || page.length > limit);
  page = page.slice(0, Math.max(limit, oldPage.length));
  const pageDelta = Number(remainsHere && matchesQuery) - Number(wasInPage);
  panel._result = {
    ...panel._result, memories: page, has_more: hasMore,
    ...(Number.isFinite(panel._result.total) ? {total: Math.max(0, panel._result.total + pageDelta)} : {}),
  };
  collection.result = panel._result;
  if (panel._query.trim().toLocaleLowerCase() !== search.memoryQuery) scheduleMemorySearch(panel);
  return true;
}

function memoryCard(panel, memory) {
  return `<article class="list-card" data-memory-id="${panel._e(memory.memory_id)}" ${memorySearchProjection(memory).includes(panel._query.trim().toLocaleLowerCase()) ? "" : "hidden"}><div class="card-main clickable edit-memory" tabindex="0" role="button" data-id="${panel._e(memory.memory_id)}"><p class="primary-copy">${panel._e(memory.content)}</p><p class="meta">${panel._e(memory.category)} · ${panel._e(memory.source)} · Updated ${panel._e(panel._formatDate(memory.updated_at))}</p></div><div class="actions"><button type="button" class="secondary memory-edit-button" data-id="${panel._e(memory.memory_id)}">Edit</button>${panel._data?.is_admin && panel._scopeId === "__anonymous__" ? `<button type="button" class="secondary reassign-memory" data-id="${panel._e(memory.memory_id)}">Assign to user</button>` : ""}<button type="button" class="danger delete-memory" data-id="${panel._e(memory.memory_id)}">Delete</button></div></article>`;
}

function currentSearch(panel, identity, sequence, query) {
  return identity === memoryCollectionIdentity(panel)
    && sequence === browserState(panel).memorySearchSequence
    && panel._viewKey() === "data-memory/memories" && panel._memoryKind === "persistent"
    && panel._query.trim() === query;
}

function applySearchResult(panel, result, query) {
  const collection = collectionState(panel);
  for (const memory of result.memories || []) collection.items.set(String(memory.memory_id), memory);
  panel._result = result;
  collection.result = result;
  browserState(panel).memoryQuery = query.toLocaleLowerCase();
  indexMemories(panel);
  // Use the renderer's existing editor-deferral mechanism, not a second editor
  // lifecycle. Search completion never enters the whole-route render path.
  if (panel.shadowRoot.querySelector("dialog[open]")) panel._eocDeferredEditorRender = true;
  else reconcilePersistentMemories(panel);
}

async function runMemorySearch(panel) {
  const state = browserState(panel);
  const query = panel._query.trim();
  const sequence = ++state.memorySearchSequence;
  const identity = memoryCollectionIdentity(panel);
  const action = query ? "search" : "list";
  try {
    const result = await panel._call("memories", action, {
      scope_id: panel._scopeId, query, limit: MEMORY_PAGE_SIZE, offset: 0,
    });
    if (!currentSearch(panel, identity, sequence, query)) return;
    applySearchResult(panel, result, query);
  } catch (err) {
    if (currentSearch(panel, identity, sequence, query)) panel._toast(`Unable to search memories: ${err.message || String(err)}`, true);
  }
}

function scheduleMemorySearch(panel) {
  const state = browserState(panel);
  clearTimeout(state.memorySearchTimer);
  // Invalidate in-flight work immediately, including A -> B -> A typing.
  state.memorySearchSequence++;
  const identity = memoryCollectionIdentity(panel), query = panel._query.trim();
  const sequence = state.memorySearchSequence;
  if (query.toLocaleLowerCase() === state.memoryQuery) return;
  state.memorySearchTimer = setTimeout(() => {
    if (currentSearch(panel, identity, sequence, query)) void runMemorySearch(panel);
  }, MEMORY_SEARCH_DEBOUNCE_MS);
}

async function loadMoreMemories(panel, button) {
  if (button.disabled) return;
  const state = browserState(panel);
  const current = panel._result?.memories || [];
  const query = panel._query.trim();
  if (query.toLocaleLowerCase() !== state.memoryQuery) return;
  const sequence = state.memorySearchSequence, identity = memoryCollectionIdentity(panel);
  panel._setSaving(button, true, "Loading…");
  try {
    const result = await panel._call("memories", query ? "search" : "list", {
      scope_id: panel._scopeId, query, limit: MEMORY_PAGE_SIZE, offset: current.length,
    });
    if (!currentSearch(panel, identity, sequence, query)) return;
    const memories = [...new Map([...current, ...(result.memories || [])].map(memory => [String(memory.memory_id), memory])).values()];
    applySearchResult(panel, {...result, memories}, query);
  } catch (err) {
    if (currentSearch(panel, identity, sequence, query)) panel._toast(`Unable to load more memories: ${err.message || String(err)}`, true);
  } finally {
    panel._setSaving(button, false);
    if (hasPersistentMemoryCollection(panel)) applyMemoryFilter(panel);
  }
}

export async function finishMemoryBrowserLoad(panel) {
  if (panel._viewKey() === "data-memory/memories" && panel._memoryKind === "persistent" && !panel._busy && !panel._error) {
    indexMemories(panel);
    if (panel._query.trim() && panel._query.trim().toLocaleLowerCase() !== browserState(panel).memoryQuery) await runMemorySearch(panel);
  }
}

const memoryStatusMarkup = panel => {
  const current = selectedFeatureStatus(panel, "memory");
  const mode = String(current?.mode || panel._selectedAgent?.()?.memory_mode || "").toLowerCase();
  const status = {
    state: current?.state || (mode === "off" ? "disabled" : mode ? "enabled" : "unknown"),
    label: current?.label || ({automatic:"Automatic", manual:"Manual", off:"Off"}[mode] || "View settings"),
    detail: current?.enabled === false || mode === "off"
      ? "Stored memories remain available to manage, but this assistant cannot use them."
      : "The assistant can use long-term memories when the request scope permits it.",
  };
  return embeddedFeatureStatusMarkup(panel, "Long-term memory", status, {page:"data-memory", subsection:"memory-settings", label:"Configure"});
};

export function renderPersistentMemories(panel) {
  const items = [...collectionState(panel).items.values()];
  return `<section class="content-card" data-persistent-memories data-collection-identity="${panel._e(memoryCollectionIdentity(panel))}"><div class="section-heading"><div><h2>Memories</h2><p>Long-term facts the assistant can reuse in future conversations. <button type="button" class="guide-topic-link guide-link" data-guide-topic="memory">Learn more</button></p></div><button type="button" id="add-memory">+ Add memory</button></div><div data-memory-feature-status>${memoryStatusMarkup(panel)}</div><div class="config-jumps"><button type="button" class="secondary memory-kind" data-kind="persistent" disabled>Long-term</button><button type="button" class="secondary memory-kind" data-kind="temporary">Short-term</button></div><input id="list-search" class="search" type="search" value="${panel._e(panel._query)}" placeholder="Search memories" aria-label="Search memories"><div class="list memory-list">${items.map(memory => memoryCard(panel, memory)).join("")}<div data-memory-empty>${panel._empty(panel._query.trim() ? "No memories match this search." : "No long-term memories yet.")}</div></div><div class="section-actions" data-memory-pagination ${panel._result?.has_more ? "" : "hidden"}><button type="button" class="secondary" id="load-more-memories">Load more ${browserState(panel).memoryQuery ? "matches" : "memories"}</button></div></section>`;
}

function applyMemoryFilter(panel) {
  const collection = collectionState(panel);
  if (!collection.host?.isConnected) return;
  const query = panel._query.trim().toLocaleLowerCase();
  const backendQuery = browserState(panel).memoryQuery;
  const resultIds = new Set((panel._result?.memories || []).map(memory => String(memory.memory_id)));
  let visible = 0;
  for (const [id, record] of collection.cards) {
    const memory = collection.items.get(id);
    const show = query === backendQuery ? resultIds.has(id) : memorySearchProjection(memory).includes(query);
    if (record.node.hidden === show) record.node.hidden = !show;
    if (show) visible++;
  }
  collection.empty.hidden = visible > 0;
  const message = query ? "No memories match this search." : "No long-term memories yet.";
  setText(collection.empty.firstElementChild || collection.empty, message);
  collection.host.querySelector("[data-memory-pagination]").hidden = !panel._result?.has_more || query !== backendQuery;
  const button = collection.host.querySelector("#load-more-memories");
  if (!button.disabled) setText(button, `Load more ${backendQuery ? "matches" : "memories"}`);
}

export function hasPersistentMemoryCollection(panel) {
  const collection = memoryCollections.get(panel);
  return Boolean(collection?.identity === memoryCollectionIdentity(panel) && collection.host?.isConnected
    && collection.host === panel.shadowRoot.querySelector("[data-persistent-memories]"));
}

export function reconcilePersistentMemories(panel) {
  const collection = collectionState(panel);
  if (!collection.host || collection.host !== panel.shadowRoot.querySelector("[data-persistent-memories]")) return false;
  const items = [...collection.items.values()];
  indexMemories(panel);
  reconcileKeyedChildren(collection.list, collection.cards, items, memory => memory.memory_id,
    memory => JSON.stringify([memory.content, memory.category, memory.source, memory.updated_at, panel._data?.is_admin]),
    memory => memoryCard(panel, memory), [collection.empty]);
  // This small status region is independent of the large card collection.
  const markup = memoryStatusMarkup(panel);
  if (collection.statusMarkup !== markup) {
    const statusHost = panel.shadowRoot.querySelector("[data-memory-feature-status]");
    statusHost.innerHTML = markup;
    statusHost.querySelector(".inline-route")?.addEventListener("click", () => panel._navigate("data-memory", "memory-settings"));
    collection.statusMarkup = markup;
  }
  applyMemoryFilter(panel);
  return true;
}

function bindPersistentMemories(panel) {
  if (panel._viewKey() !== "data-memory/memories" || panel._memoryKind !== "persistent") return;
  const host = panel.shadowRoot.querySelector("[data-persistent-memories]");
  if (!host) return;
  const collection = collectionState(panel);
  if (collection.host !== host) {
    collection.host = host;
    collection.statusMarkup = memoryStatusMarkup(panel);
    collection.list = host.querySelector(".memory-list");
    collection.cards = adoptKeyedElements(collection.list, "[data-memory-id]", "memoryId");
    collection.empty = collection.list.querySelector("[data-memory-empty]");
    reconcilePersistentMemories(panel);
    host.querySelector("#list-search").addEventListener("input", event => { panel._query = event.target.value; filterPersistentMemories(panel); });
  }
  delegateCollectionActions(host, "#add-memory,.edit-memory,.memory-edit-button,.delete-memory,.reassign-memory,.memory-kind,#load-more-memories", control => {
    if (control.matches(".delete-memory")) void panel._deleteMemory(control.dataset.id, false, control);
    else if (control.matches(".reassign-memory")) panel._openReassign(control.dataset.id);
    else if (control.matches("#load-more-memories")) void loadMoreMemories(panel, control);
    else if (control.matches(".memory-kind")) {
      panel._memoryKind = control.dataset.kind; panel._query = "";
      void panel._loadSection();
    } else void panel._openMemory(control.dataset.id);
  });
}


export function filterPersistentMemories(panel) {
  applyMemoryFilter(panel);
  scheduleMemorySearch(panel);
}

export function bindMemoryBrowser(panel) {
  bindPersistentMemories(panel);
  const guest = panel.shadowRoot.querySelector("#guest-web-search");
  if (guest && !guest.__eocMemoryBrowserBound) {
    guest.__eocMemoryBrowserBound = true;
    guest.addEventListener("change", event => {
      if (panel._guestDraft) panel._guestDraft.guest_web_search = Boolean(event.currentTarget.checked);
    });
  }
}
