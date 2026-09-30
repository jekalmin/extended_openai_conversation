const watches = new WeakMap();

export function stopLiveStatus(panel, key = null) {
  const entries = watches.get(panel);
  if (!entries) return;
  for (const [name, watch] of entries) {
    if (key && name !== key) continue;
    clearTimeout(watch.timer);
    watch.stopped = true;
    entries.delete(name);
  }
}

export function watchLiveStatus(panel, key, {view, delay, refresh, maxRefreshes = Infinity}) {
  let entries = watches.get(panel);
  if (!entries) watches.set(panel, entries = new Map());
  const identity = `${panel._data?.entry_id || ""}|${panel._agentId || ""}|${view}`;
  const existing = entries.get(key);
  if (existing?.identity === identity && !existing.stopped) return;
  stopLiveStatus(panel, key);
  const watch = {identity, stopped:false, timer:null, attempts:0};
  entries.set(key, watch);
  const stop = () => {
    if (entries.get(key) === watch) stopLiveStatus(panel, key);
    else {clearTimeout(watch.timer);watch.stopped = true;}
  };
  const current = () => !watch.stopped && panel.isConnected !== false
    && panel._viewKey?.() === view
    && `${panel._data?.entry_id || ""}|${panel._agentId || ""}|${view}` === identity;
  const schedule = () => {
    if (!current() || watch.attempts >= maxRefreshes) return stop();
    const milliseconds = typeof delay === "function" ? delay() : delay;
    if (milliseconds == null) return stop();
    watch.timer = setTimeout(async () => {
      if (!current()) return stop();
      if (globalThis.document?.hidden) return schedule();
      watch.attempts++;
      try { await refresh(current); }
      catch (_err) { /* A transient read failure must not invalidate the user's draft. */ }
      schedule();
    }, Math.max(200, milliseconds));
  };
  schedule();
}

function boundaryDelay(value) {
  const boundary = Date.parse(value || "");
  return Number.isFinite(boundary) && boundary > Date.now()
    ? Math.min(60000, boundary - Date.now() + 200) : 60000;
}

export function watchTimedFeatureStatus(panel) {
  const view = panel._viewKey?.();
  if (!["capabilities/guest-mode", "capabilities/quiet-hours"].includes(view)) return;
  const guest = view === "capabilities/guest-mode";
  watchLiveStatus(panel, "timed-feature", {
    view,
    delay: () => guest
      ? boundaryDelay(panel._result?.status?.scheduled ? panel._result.status.active_from : panel._result?.status?.active_until)
      : boundaryDelay(panel._result?.period_ends_at),
    refresh: async (current) => {
      const loadToken = panel._loadToken;
      const result = await panel._call(guest ? "guest_mode" : "quiet_hours", "get");
      if (!current() || panel._loadToken !== loadToken || panel._guestOperation) return;
      if (guest) {
        panel._patchGuestModeStatus(panel._agentId, result.status);
        const status = panel._result?.status || {};
        const node = panel.shadowRoot.querySelector("[data-guest-live-status]");
        if (node) node.textContent = `${panel._titleCase(String(status.state || "inactive").replaceAll("_", " "))} · ${status.active_from ? `Starts ${panel._formatDate(status.active_from)}${status.active_until ? ` · Ends ${panel._formatDate(status.active_until)}` : " · No expiry"}` : "No interval configured"}`;
      } else {
        panel._result = {...panel._result, active:result.active, period_started_at:result.period_started_at, period_ends_at:result.period_ends_at, owned_controls:result.owned_controls};
        const title = panel.shadowRoot.querySelector("[data-qh-live-title]");
        const badge = panel.shadowRoot.querySelector("[data-qh-live-badge]");
        if (title) title.textContent = result.active ? "Quiet Hours active now" : panel._result.config?.enabled ? "Outside Quiet Hours" : "Quiet Hours schedule disabled";
        if (badge) { badge.textContent = result.active ? "Active" : "Inactive"; badge.className = result.active ? "availability-badge" : "disabled-badge"; }
      }
    },
  });
}
