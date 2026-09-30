import {adoptKeyedElements, reconcileKeyedChildren, delegateCollectionActions, setText} from "./keyed-collection.js";
import {selectedFeatureStatus} from "./management-feature-status-core.js";
import {knowledgeSourceAvailabilityBadge} from "./knowledge-presentation.js";
import {formatUsageNumber} from "./usage-format.js";

const collections = new WeakMap();
const identity = panel => JSON.stringify([panel._selectedAgent?.()?.entry_id, panel._agentId, panel._hass?.user?.id]);
const signature = source => JSON.stringify([source.title, source.description, source.enabled, source.character_count, source.updated_at]);
const matches = (source, query) => `${source.title || ""} ${source.description || ""}`.toLocaleLowerCase().includes(query);
const countText = sources => `${formatUsageNumber(sources.length)} source${sources.length === 1 ? "" : "s"}`;
function knowledgeAccessState(panel) {
  const current = selectedFeatureStatus(panel, "knowledge") || {};
  const enabled = typeof current.enabled === "boolean" ? current.enabled : panel._selectedAgent?.()?.knowledge_enabled !== false;
  const sources = panel._result?.sources || [];
  const usable = sources.some((source) => source.enabled !== false);
  const exceptional = ["unknown", "unavailable", "error"].includes(current.state);
  const state = exceptional ? current.state : enabled ? usable ? "available" : "empty" : "disabled";
  const detail = !enabled ? "Stored sources stay in the library, but the assistant cannot use them."
    : ["unknown", "unavailable", "error"].includes(state) ? current.detail || current.summary || "Knowledge availability could not be confirmed."
    : usable ? "The assistant can search these sources when needed."
    : sources.length ? "Enable a source to make Knowledge available to the assistant."
    : "Add a source to make Knowledge available to the assistant.";
  return {
    enabled, state, detail,
    label: exceptional ? current.label || "Unknown" : {available:"Available", empty:"Needs sources", disabled:"Off"}[state],
  };
}
const statusMarkup = panel => {
  const status = knowledgeAccessState(panel);
  return `<div class="compact-status"><small>${panel._e(status.detail)}</small><strong class="status-value ${["enabled", "available"].includes(status.state) ? "on" : ""}">${panel._e(status.label)}</strong></div>`;
};

export function applyKnowledgeMutation(panel, response, deletedId = null) {
  if (!panel._result || !Array.isArray(panel._result.sources)) return false;
  if (deletedId && response?.deleted !== 1) return false;
  const summary = response?.summary;
  if (!deletedId && !summary?.source_id) return false;
  const sources = panel._result.sources.filter(source => source.source_id !== (deletedId || summary.source_id));
  if (summary) sources.push(summary);
  sources.sort((left, right) => String(right.updated_at).localeCompare(String(left.updated_at)));
  panel._result = {
    ...panel._result, sources,
    ...(response.stats ? {stats: response.stats} : {}),
    ...(response.feature_status ? {feature_status: response.feature_status} : {}),
  };
  return true;
}


export function knowledgeAvailabilityMarkup(panel) {
  const status = `<div id="knowledge-status" aria-live="polite">${statusMarkup(panel)}</div>`;
  if (panel._data?.is_admin === false) {
    return `<div class="knowledge-availability-setting"><strong>Assistant access</strong>${status}</div>`;
  }
  const enabled = knowledgeAccessState(panel).enabled;
  return `<div class="knowledge-availability-setting">
    <div class="config-toggle setting">
      <div class="setting-copy"><span class="setting-label-row"><label for="knowledge-enabled-toggle"><strong>Allow the assistant to use Knowledge</strong></label></span>${status}<small>Changes here save immediately.</small></div>
      <label class="switch-control" for="knowledge-enabled-toggle"><input id="knowledge-enabled-toggle" type="checkbox" role="switch" ${enabled ? "checked" : ""}><span class="switch-track" aria-hidden="true"></span></label>
    </div>
  </div>`;
}

async function saveKnowledgeAvailability(panel, input) {
  const desired = input.checked;
  const agentId = panel._agentId;
  const loadToken = panel._loadToken;
  input.disabled = true;
  try {
    const result = await panel._call("knowledge", "set_enabled", {enabled: desired});
    if (panel._agentId !== agentId || panel._viewKey?.() !== "data-memory/knowledge" || panel._loadToken !== loadToken) return;
    const agent = panel._selectedAgent?.();
    if (agent) {
      agent.knowledge_enabled = result.knowledge_enabled;
      agent.feature_status = {
        ...(agent.feature_status || {}),
        knowledge: result.feature_status,
      };
    }
    if (panel._viewKey?.() === "data-memory/knowledge" && panel._result) {
      panel._result = {
        ...panel._result,
        feature_status: result.feature_status,
      };
    }
    panel._clearConfigDraft?.();
    panel._render();
    panel._toast(`Knowledge ${result.knowledge_enabled ? "enabled" : "disabled"}`);
  } catch (err) {
    input.checked = !desired;
    input.disabled = false;
    panel._toast(`Unable to update Knowledge: ${err.message || String(err)}`, true);
  } finally {
    input.disabled = false;
  }
}

export function bindKnowledgeAvailability(panel) {
  const input = panel.shadowRoot?.querySelector("#knowledge-enabled-toggle");
  if (!input || input.__eocKnowledgeBound) return;
  input.__eocKnowledgeBound = true;
  input.addEventListener("change", () => saveKnowledgeAvailability(panel, input));
}

function sourceCard(panel, source) {
  return `<article class="list-card" data-source-id="${panel._e(source.source_id)}" ${matches(source, String(panel._query || "").trim().toLocaleLowerCase()) ? "" : "hidden"}><div class="card-main clickable edit-source" tabindex="0" role="button" data-id="${panel._e(source.source_id)}"><h3>${panel._e(source.title)}</h3>${knowledgeSourceAvailabilityBadge(source)}<p class="description">${panel._e(source.description || "No description")}</p><p class="meta">${formatUsageNumber(source.character_count || 0)} characters · Updated ${panel._e(panel._formatDate(source.updated_at))}</p></div><div class="actions"><button type="button" class="secondary source-edit-button" data-id="${panel._e(source.source_id)}">Edit</button><button type="button" class="danger delete-source" data-id="${panel._e(source.source_id)}">Delete</button></div></article>`;
}

export function renderKnowledge(panel) {
  const sources = panel._result?.sources || [];
  return `<section class="content-card" data-knowledge-collection data-collection-identity="${panel._e(identity(panel))}"><div class="section-heading"><div><h2>Sources</h2><p data-source-count>${countText(sources)}</p></div><button type="button" id="add-source">+ Add source</button></div>${knowledgeAvailabilityMarkup(panel)}<input id="list-search" class="search" type="search" value="${panel._e(panel._query)}" placeholder="Filter by title or description" aria-label="Filter Knowledge sources"><div class="list knowledge-list">${sources.map(source => sourceCard(panel, source)).join("")}<div data-source-empty>${panel._empty("No Knowledge sources yet. Add one to make reference information available on demand.")}</div></div></section>`;
}

export function filterKnowledge(panel) {
  const state = collections.get(panel);
  if (!state || !state.host.isConnected || state.identity !== identity(panel)) return;
  const query = String(panel._query || "").trim().toLocaleLowerCase();
  let visible = 0;
  for (const source of panel._result?.sources || []) {
    const node = state.cards.get(String(source.source_id))?.node;
    if (!node) continue;
    const show = matches(source, query);
    if (node.hidden === show) node.hidden = !show;
    if (show) visible++;
  }
  state.empty.hidden = visible > 0;
  // Empty copy has the same presentation as the original route's _empty().
  const message = query ? "No sources match this filter." : "No Knowledge sources yet. Add one to make reference information available on demand.";
  setText(state.empty.firstElementChild || state.empty, message);
}

export function reconcileKnowledge(panel) {
  const state = collections.get(panel);
  if (!state || state.host !== panel.shadowRoot.querySelector("[data-knowledge-collection]") || state.identity !== identity(panel)) return false;
  const sources = panel._result?.sources || [];
  reconcileKeyedChildren(state.list, state.cards, sources, source => source.source_id, signature, source => sourceCard(panel, source), [state.empty]);
  setText(state.host.querySelector("[data-source-count]"), countText(sources));
  const status = state.host.querySelector("#knowledge-status"), markup = statusMarkup(panel);
  if (state.status !== markup) { status.innerHTML = markup; state.status = markup; }
  const toggle = panel.shadowRoot.querySelector("#knowledge-enabled-toggle");
  const enabled = selectedFeatureStatus(panel, "knowledge")?.enabled;
  if (toggle && typeof enabled === "boolean" && !toggle.disabled) toggle.checked = enabled;
  filterKnowledge(panel);
  return true;
}

export function bindKnowledge(panel) {
  if (panel._viewKey() !== "data-memory/knowledge") return;
  bindKnowledgeAvailability(panel);
  const host = panel.shadowRoot.querySelector("[data-knowledge-collection]");
  if (!host) return;
  if (collections.get(panel)?.host !== host) {
    const list = host.querySelector(".knowledge-list");
    collections.set(panel, {host, list, identity: identity(panel), cards: adoptKeyedElements(list, "[data-source-id]", "sourceId"), empty: list.querySelector("[data-source-empty]"), status: statusMarkup(panel)});
    // Prime adopted signatures before the first asynchronous data change.
    reconcileKnowledge(panel);
    host.querySelector("#list-search").addEventListener("input", event => { panel._query = event.target.value; filterKnowledge(panel); });
  }
  delegateCollectionActions(host, "#add-source,.edit-source,.source-edit-button,.delete-source", control => {
    if (control.matches(".delete-source")) void panel._deleteSource(control.dataset.id, false, control);
    else void panel._openKnowledge(control.dataset.id);
  });
}
