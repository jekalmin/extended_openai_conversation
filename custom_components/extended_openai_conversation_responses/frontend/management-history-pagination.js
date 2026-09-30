const CONVERSATIONS_VIEW = "data-memory/conversations";
const LIST_PAGE_LIMIT = 50;
const SEARCH_PAGE_LIMIT = 20;
const TURN_PAGE_LIMIT = 20;

function integer(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.trunc(number)) : fallback;
}

function pageLabel(meta = {}, noun = "items") {
  const offset = integer(meta.offset);
  const returned = integer(meta.returned, Array.isArray(meta.sessions) ? meta.sessions.length : 0);
  const total = Number.isFinite(Number(meta.total)) ? integer(meta.total) : null;
  if (!returned) return total === null ? `No ${noun}` : `0 of ${total.toLocaleString()} ${noun}`;
  const range = `${(offset + 1).toLocaleString()}–${(offset + returned).toLocaleString()}`;
  return total === null ? `${range} ${noun}` : `${range} of ${total.toLocaleString()} ${noun}`;
}

function searchSessionRows(found) {
  return (found?.results || []).map((item) => ({
    ...item,
    last_message_at: item.timestamp,
    turn_count: "matching",
    scope_source: "search match",
  }));
}

function conversationResult(panel) {
  return panel._contentData || panel._result || {};
}

function setSessionResult(panel, response, mode) {
  const target = conversationResult(panel);
  target.sessions = mode === "search"
    ? {...response, sessions: searchSessionRows(response)}
    : response;
}

async function loadConversationPage(panel, offset) {
  if (panel._eocHistoryPagePending) return;
  panel._eocHistoryPagePending = true;
  try {
    const mode = panel._eocHistoryMode === "search" ? "search" : "list";
    const extra = {
      scope_id: panel._scopeId,
      offset: integer(offset),
      limit: mode === "search" ? SEARCH_PAGE_LIMIT : LIST_PAGE_LIMIT,
    };
    if (mode === "search") extra.query = panel._eocHistoryQuery || "";
    const response = await panel._call("conversations", mode, extra);
    setSessionResult(panel, response, mode);
  } catch (err) {
    panel._toast(`Unable to load conversation history: ${err.message || String(err)}`, true);
  } finally {
    panel._eocHistoryPagePending = false;
    panel._render();
  }
}

function decorateConversationPager(panel) {
  if (panel._viewKey?.() !== CONVERSATIONS_VIEW) return;
  const input = panel.shadowRoot?.querySelector("#archive-query");
  const card = input?.closest(".content-card");
  if (!input || !card) return;
  input.value = panel._eocHistoryMode === "search" ? (panel._eocHistoryQuery || "") : "";

  const sessions = conversationResult(panel).sessions || {};
  const existing = card.querySelector(".eoc-history-pager");
  existing?.remove();
  const pager = document.createElement("div");
  pager.className = "section-actions eoc-history-pager";

  const status = document.createElement("small");
  status.className = "meta";
  status.style.marginRight = "auto";
  status.textContent = pageLabel(
    sessions,
    panel._eocHistoryMode === "search" ? "matching turns" : "conversations",
  );
  pager.append(status);

  if (panel._eocHistoryMode === "search") {
    const clear = document.createElement("button");
    clear.type = "button";
    clear.className = "secondary";
    clear.textContent = "Clear search";
    clear.addEventListener("click", () => {
      panel._eocHistoryMode = "list";
      panel._eocHistoryQuery = "";
      void loadConversationPage(panel, 0);
    });
    pager.append(clear);
  }

  const previous = document.createElement("button");
  previous.type = "button";
  previous.className = "secondary";
  previous.textContent = "Previous";
  const offset = integer(sessions.offset);
  const limit = Math.max(1, integer(sessions.limit, panel._eocHistoryMode === "search" ? SEARCH_PAGE_LIMIT : LIST_PAGE_LIMIT));
  previous.disabled = offset === 0 || panel._eocHistoryPagePending;
  previous.addEventListener("click", () => void loadConversationPage(panel, Math.max(0, offset - limit)));
  pager.append(previous);

  const next = document.createElement("button");
  next.type = "button";
  next.className = "secondary";
  next.textContent = "Next";
  next.disabled = !sessions.has_more || panel._eocHistoryPagePending;
  const nextOffset = sessions.next_offset == null
    ? offset + integer(sessions.returned, (sessions.sessions || []).length)
    : integer(sessions.next_offset);
  next.addEventListener("click", () => void loadConversationPage(panel, nextOffset));
  pager.append(next);

  card.append(pager);
}

function renderTurns(panel, data) {
  return (data.turns || []).map((turn) => `<article class="turn"><div class="message user"><strong>You</strong><p>${panel._e(turn.user_text)}</p></div><div class="message assistant"><strong>Assistant</strong><p>${panel._e(turn.assistant_text)}</p></div><small>${panel._e(panel._formatDate(turn.timestamp))}</small></article>`).join("") || panel._empty("No turns retained on this page.");
}

function appendTurnPager(panel, body, sessionId, data) {
  const pager = document.createElement("div");
  pager.className = "section-actions eoc-turn-pager";
  const status = document.createElement("small");
  status.className = "meta";
  status.style.marginRight = "auto";
  status.textContent = pageLabel(data, "turns");
  pager.append(status);

  const offset = integer(data.offset, integer(data.start_turn));
  const limit = Math.max(1, integer(data.limit, TURN_PAGE_LIMIT));
  const previous = document.createElement("button");
  previous.type = "button";
  previous.className = "secondary";
  previous.textContent = "Previous turns";
  previous.disabled = offset === 0;
  previous.addEventListener("click", () => void openSession(panel, sessionId, Math.max(0, offset - limit)));
  pager.append(previous);

  const next = document.createElement("button");
  next.type = "button";
  next.className = "secondary";
  next.textContent = "Next turns";
  next.disabled = !data.has_more;
  const nextOffset = data.next_offset == null
    ? offset + integer(data.returned, (data.turns || []).length)
    : integer(data.next_offset);
  next.addEventListener("click", () => void openSession(panel, sessionId, nextOffset));
  pager.append(next);
  body.append(pager);
}

export function bindConversationActions(panel) {
  const root = panel.shadowRoot;
  const host = root.querySelector("#archive-query")?.closest(".content-card");
  if (!host || host.__eocHistoryActionsBound) return;
  host.__eocHistoryActionsBound = true;
  root.querySelectorAll(".open-session, .view-session").forEach(element => panel._activate(element, () => openSession(panel, element.dataset.id)));
  root.querySelector("#archive-search")?.addEventListener("click", () => searchArchive(panel));
  root.querySelector("#archive-query")?.addEventListener("keydown", event => { if (event.key === "Enter") void searchArchive(panel); });
}

async function searchArchive(panel, offset = 0) {
  const input = panel.shadowRoot.querySelector("#archive-query");
  const query = (offset ? panel._eocHistoryQuery : input?.value || "").trim();
  if (!query) {
    panel._eocHistoryMode = "list";
    panel._eocHistoryQuery = "";
    await loadConversationPage(panel, 0);
    return;
  }
  panel._eocHistoryMode = "search";
  panel._eocHistoryQuery = query;
  await loadConversationPage(panel, offset);
}

async function openSession(panel, sessionId, startTurn = 0) {
  panel._ensureOwnedDialog("session-dialog");
  const root = panel.shadowRoot;
  const dialog = root.querySelector("#session-dialog");
  const title = root.querySelector("#session-title");
  const body = root.querySelector("#session-body");
  const token = (panel._eocSessionLoadToken || 0) + 1;
  panel._eocSessionLoadToken = token;
  title.textContent = "Loading conversation…";
  body.innerHTML = panel._loading();
  if (!dialog.open) dialog.showModal();
  try {
    const data = await panel._call("conversations", "get", {
      scope_id: panel._scopeId,
      session_id: sessionId,
      start_turn: integer(startTurn),
      limit: TURN_PAGE_LIMIT,
    });
    if (panel._eocSessionLoadToken !== token || !dialog.open) return;
    title.textContent = data.session?.title || "Untitled conversation";
    body.innerHTML = renderTurns(panel, data);
    appendTurnPager(panel, body, sessionId, data);
  } catch (err) {
    if (panel._eocSessionLoadToken !== token || !dialog.open) return;
    title.textContent = "Unable to load conversation";
    body.innerHTML = `<div class="error" role="alert">${panel._e(err.message || String(err))}</div>`;
  }
}

export {
  decorateConversationPager,
  LIST_PAGE_LIMIT,
  SEARCH_PAGE_LIMIT,
  TURN_PAGE_LIMIT,
  pageLabel,
  searchSessionRows,
};

function activeConversationMarkup(panel, result) {
  const active = result.active?.active || [];
  const loading = result.loading || {};
  const errors = result.load_errors || [];
  return `${loading.active ? '<p class="help">Loading active conversations…</p>' : ""}${errors.map((issue) => `<div class="notice"><strong>${panel._e(issue.label)} unavailable</strong><p>${panel._e(issue.message)}</p></div>`).join("")}${panel._data?.is_admin && active.length ? `<section class="content-card"><div class="section-heading"><div><h2>Active conversations</h2><p>Recent conversations that can continue when the same user or voice device speaks again.</p></div></div><div class="list">${active.map((item) => `<article class="list-card"><div class="card-main"><h3>${panel._e(item.label)}</h3><p class="meta">Last active ${panel._e(panel._formatDate(item.last_active))} · Expires ${panel._e(panel._formatDate(item.expires_at))}</p></div><div class="actions"><button type="button" class="danger end-active" data-key="${panel._e(item.key)}">Start fresh next time</button></div></article>`).join("")}</div></section>` : ""}`;
}

export function reconcileActiveConversations(panel) {
  if (panel._viewKey?.() !== "data-memory/conversations" || panel._busy) return false;
  const host = panel.shadowRoot?.querySelector?.("[data-eoc-active-conversations]");
  if (!host) return false;
  host.innerHTML = activeConversationMarkup(panel, panel._contentData || {});
  panel._bindActiveConversationActions?.();
  return true;
}

export function renderConversations(panel) {
  const result = panel._contentData || panel._result || {};
  const content = `<section class="page-intro"><h1>Conversation history</h1><p>Recent conversations can continue when the same user or device speaks again. Review or search retained conversations below.</p></section><div data-eoc-active-conversations>${activeConversationMarkup(panel, result)}</div>
    <section class="content-card"><div class="section-heading"><div><h2>Retained conversations</h2><p>Search and review conversations for the selected scope.</p></div></div><div class="search-row"><input id="archive-query" type="search" placeholder="Search retained discussions" aria-label="Search retained discussions"><button type="button" id="archive-search">Search</button></div><div class="list">${(result.sessions?.sessions || []).map((item) => `<article class="list-card"><div class="card-main clickable open-session" tabindex="0" role="button" data-id="${panel._e(item.session_id)}"><h3>${panel._e(item.title || "Untitled conversation")}</h3><p class="meta">${panel._e(panel._formatDate(item.last_message_at))} · ${panel._e(String(item.turn_count))} turns · ${panel._e(item.scope_source)}</p></div><div class="actions"><button type="button" class="secondary view-session" data-id="${panel._e(item.session_id)}">View</button><button type="button" class="danger delete-session" data-id="${panel._e(item.session_id)}">Delete</button></div></article>`).join("") || panel._empty(panel._eocHistoryMode === "search" ? "No conversations match this search." : "No retained conversations in this scope.")}</div></section>`;
  return content;
}
