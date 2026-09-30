import {watchLiveStatus, stopLiveStatus} from "./management-live-status.js";
const ROUTE_STYLE = `/* Overview Broadcast */
#broadcast-card{background:color-mix(in srgb,var(--secondary-background-color) 20%,var(--card-background-color));border-color:color-mix(in srgb,var(--divider-color) 72%,var(--secondary-text-color))}
.broadcast-toggle-row{display:flex;justify-content:space-between;gap:18px;align-items:center;padding:16px 0;border-bottom:1px solid var(--divider-color)}
.broadcast-toggle-row p{margin:4px 0 0;color:var(--secondary-text-color);line-height:1.45}
.status-pill{padding:5px 10px;border-radius:999px;background:var(--secondary-background-color);font-size:13px}.status-on{color:var(--success-color,#0f9d58)}
.broadcast-compose{display:grid;gap:18px;margin-top:18px}
.broadcast-message>span,.broadcast-destination legend{display:block;font-weight:600;margin-bottom:8px}
.broadcast-message textarea{box-sizing:border-box;width:100%;resize:vertical;padding:10px 12px;border:1px solid var(--divider-color);border-radius:8px;background:var(--card-background-color);color:var(--primary-text-color);font:inherit}
.broadcast-destination{border:0;padding:0;margin:0;min-width:0}
.broadcast-mode-options{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
.broadcast-mode-option,.broadcast-target{display:flex;align-items:flex-start;gap:10px;padding:12px 14px;border:1px solid var(--divider-color);border-radius:10px;background:var(--card-background-color);cursor:pointer}
.broadcast-mode-option.is-selected,.broadcast-target.is-selected{border-color:var(--primary-color);background:color-mix(in srgb,var(--primary-color) 6%,var(--card-background-color))}
.broadcast-mode-option input,.broadcast-target input{width:18px!important;height:18px!important;min-width:18px;flex:0 0 18px;margin:2px 0 0;padding:0}
.broadcast-mode-option span,.broadcast-target span{display:grid;gap:3px;min-width:0}
.broadcast-mode-option small,.broadcast-target small,.broadcast-target-heading small{color:var(--secondary-text-color);font-weight:400}
.broadcast-target-section{display:grid;gap:8px;margin-top:14px}
.broadcast-target-heading{display:flex;align-items:center;justify-content:space-between;gap:12px}
.broadcast-targets{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:9px}
.broadcast-actions{justify-content:flex-end}
.broadcast-history{margin-top:24px}.broadcast-history h3{margin:0 0 8px}
.broadcast-history-item{display:grid;gap:8px;padding:12px 0;border-top:1px solid var(--divider-color)}
.broadcast-history-item>div{display:flex;justify-content:space-between;gap:14px}.broadcast-history-item small{color:var(--secondary-text-color)}
.broadcast-history-item ul{list-style:none;padding:0;margin:0;display:grid;gap:4px}.broadcast-history-item li{display:flex;justify-content:space-between;gap:12px}
@media (max-width:700px){.broadcast-mode-options{grid-template-columns:1fr}.broadcast-actions{justify-content:stretch}.broadcast-actions button{flex:1}}`;
function ensureRouteStyle(panel) {
  if (typeof document === "undefined") return;
  if (panel.shadowRoot.querySelector('style[data-eoc-feature-style="broadcast"]')) return;
  const style = document.createElement("style");
  style.dataset.eocFeatureStyle = "broadcast";
  style.textContent = ROUTE_STYLE;
  panel.shadowRoot.append(style);
}

const WS_BROADCAST = "extended_openai_conversation_responses/broadcast";

export function hasPendingBroadcast(snapshot) {
  const pending = new Set(["pending", "queued", "queued_idle", "queued_busy", "waiting_idle", "delivering"]);
  return (snapshot?.history || []).some(item => pending.has(item.status)
    || Object.values(item.deliveries || {}).some(delivery => pending.has(delivery.status)));
}

function statusLabel(status) {
  return {
    pending: "Pending",
    queued_idle: "Queued",
    queued_busy: "Waiting for idle",
    waiting_idle: "Waiting for idle",
    delivering: "Delivering",
    delivered: "Delivered",
    failed: "Failed",
    expired: "Expired",
  }[status] || status;
}

function broadcastMarkup(panel, snapshot) {
  if (!snapshot) return `<p class="empty">Loading Broadcast…</p>`;
  const satellites = snapshot.catalog?.satellites || [];
  const satellitesById = new Map(satellites.map((satellite) => [satellite.id, satellite]));
  const areas = new Map((snapshot.catalog?.areas || []).map((area) => [area.id, area.name]));
  const selected = panel._broadcastSelected || new Set();
  const wholeHome = Boolean(panel._broadcastWholeHome);
  const message = panel._broadcastMessage || "";
  const enabled = snapshot.enabled === true;
  const canManage = snapshot.can_manage === true;
  const history = snapshot.history || [];
  return `
    <div class="section-heading broadcast-heading"><div><span class="section-kicker"><ha-icon icon="mdi:bullhorn-outline"></ha-icon> Home messaging</span><h2>Broadcast</h2><p>Send a spoken message to selected Assist satellites or the whole home. Busy satellites wait until they are free.</p></div></div>
    <div class="broadcast-toggle-row">
      <div><strong>Enable Broadcast</strong>${enabled ? "" : "<p>Broadcast is currently off.</p>"}</div>
      ${canManage ? `<label class="switch-control" for="broadcast-enabled"><input id="broadcast-enabled" type="checkbox" role="switch" aria-label="Enable Broadcast" ${enabled ? "checked" : ""}><span class="switch-track" aria-hidden="true"></span></label>` : `<strong class="status-pill ${enabled ? "status-on" : "status-off"}">${enabled ? "On" : "Off"}</strong>`}
    </div>
    ${enabled ? `
      <div class="broadcast-compose">
        <label class="broadcast-message"><span>Message</span><textarea id="broadcast-message" rows="3" placeholder="Dinner is ready">${panel._e(message)}</textarea></label>
        <fieldset class="broadcast-destination">
          <legend>Send to</legend>
          <div class="broadcast-mode-options" role="radiogroup" aria-label="Broadcast destination">
            <label class="broadcast-mode-option ${wholeHome ? "is-selected" : ""}"><input type="radio" name="broadcast-destination" value="whole" ${wholeHome ? "checked" : ""}><span><strong>Whole home</strong><small>Send to every available Assist satellite.</small></span></label>
            <label class="broadcast-mode-option ${wholeHome ? "" : "is-selected"}"><input type="radio" name="broadcast-destination" value="selected" ${wholeHome ? "" : "checked"}><span><strong>Selected satellites</strong><small>Choose one or more destinations below.</small></span></label>
          </div>
          ${wholeHome ? "" : `<div class="broadcast-target-section"><div class="broadcast-target-heading"><strong>Satellites</strong><small>${satellites.length} available</small></div><div class="broadcast-targets">
            ${satellites.map((sat) => {
              const area = areas.get(sat.area_id);
              const detail = [area, panel._titleCase(sat.state || "unknown")].filter(Boolean).join(" · ");
              return `<label class="broadcast-target ${selected.has(sat.id) ? "is-selected" : ""}"><input type="checkbox" data-broadcast-entity="${panel._e(sat.id)}" ${selected.has(sat.id) ? "checked" : ""}><span><strong>${panel._e(sat.name)}</strong><small>${panel._e(detail || "Assist satellite")}</small></span></label>`;
            }).join("") || `<p class="empty">No announcement-capable Assist satellites are available.</p>`}
          </div></div>`}
        </fieldset>
        <div class="actions broadcast-actions"><button id="broadcast-refresh" type="button" class="secondary compact-button">Refresh devices</button><button id="broadcast-send" type="button" ${!satellites.length ? "disabled" : ""}>Send broadcast</button></div>
      </div>` : ""}
    <div class="broadcast-history">
      <h3>Recent broadcasts</h3>
      ${history.length ? history.slice(0, 10).map((item) => {
        const deliveries = Object.entries(item.deliveries || {}).map(([entityId, delivery]) => {
          const satellite = satellitesById.get(entityId);
          return `<li><span>${panel._e(satellite?.name || entityId)}</span><strong>${panel._e(statusLabel(delivery.status))}</strong></li>`;
        }).join("");
        return `<article class="broadcast-history-item"><div><strong>${panel._e(item.message)}</strong><small>${panel._e(new Date(item.created_at).toLocaleString())}</small></div><ul>${deliveries}</ul></article>`;
      }).join("") : `<p class="empty">No broadcasts sent yet.</p>`}
    </div>`;
}

function bindBroadcastControls(panel, snapshot) {
  const root = panel.shadowRoot;
  root.querySelector("#broadcast-enabled")?.addEventListener("change", async (event) => {
    const enabled = event.target.checked;
    event.target.disabled = true;
    try {
      await panel._hass.callWS({type: WS_BROADCAST, action: "set_enabled", enabled});
      await loadBroadcast(panel);
      panel._toast(`Broadcast ${enabled ? "enabled" : "disabled"}`);
    } catch (err) {
      panel._toast(`Unable to change Broadcast: ${err.message || String(err)}`, true);
      await loadBroadcast(panel);
    }
  });
  root.querySelector("#broadcast-message")?.addEventListener("input", (event) => { panel._broadcastMessage = event.target.value; });
  root.querySelectorAll('input[name="broadcast-destination"]').forEach((input) => input.addEventListener("change", (event) => {
    if (!event.target.checked) return;
    panel._broadcastWholeHome = event.target.value === "whole";
    const host = root.querySelector("#broadcast-card");
    if (host) host.innerHTML = broadcastMarkup(panel, snapshot);
    bindBroadcastControls(panel, snapshot);
  }));
  root.querySelectorAll("[data-broadcast-entity]").forEach((box) => box.addEventListener("change", (event) => {
    panel._broadcastSelected ||= new Set();
    if (event.target.checked) panel._broadcastSelected.add(event.target.dataset.broadcastEntity);
    else panel._broadcastSelected.delete(event.target.dataset.broadcastEntity);
    event.target.closest(".broadcast-target")?.classList.toggle("is-selected", event.target.checked);
  }));
  root.querySelector("#broadcast-refresh")?.addEventListener("click", () => loadBroadcast(panel));
  root.querySelector("#broadcast-send")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    const message = String(panel._broadcastMessage || "").trim();
    const selected = [...(panel._broadcastSelected || new Set())];
    const wholeHome = Boolean(panel._broadcastWholeHome);
    if (!message) return panel._toast("Enter a message to broadcast.", true);
    if (!wholeHome && !selected.length) return panel._toast("Choose at least one Assist satellite or Whole home.", true);
    button.disabled = true;
    try {
      await panel._hass.callWS({
        type: WS_BROADCAST,
        action: "send",
        message,
        whole_home: wholeHome,
        entity_ids: selected,
      });
      panel._broadcastMessage = "";
      await loadBroadcast(panel);
      panel._toast("Broadcast queued");
    } catch (err) {
      panel._toast(`Unable to send Broadcast: ${err.message || String(err)}`, true);
      button.disabled = false;
    }
  });
}

function applyBroadcastSnapshot(panel, snapshot) {
  if (panel._viewKey?.() !== "overview") return;
  const host = panel.shadowRoot.querySelector("#broadcast-card");
  if (!host) return;
  const available = new Set((snapshot.catalog?.satellites || []).map((satellite) => satellite.id));
  panel._broadcastSelected = new Set([...(panel._broadcastSelected || new Set())].filter((entityId) => available.has(entityId)));
  host.innerHTML = broadcastMarkup(panel, snapshot);
  bindBroadcastControls(panel, snapshot);
  panel._eocBroadcastEpoch = (panel._eocBroadcastEpoch || 0) + 1;
  panel._eocBroadcastSnapshot = snapshot;
  if (hasPendingBroadcast(snapshot)) {
    const started = Date.now();
    watchLiveStatus(panel, "broadcast", {
      // Fast completion feedback, then back off while a satellite is busy.
      // The bounded budget covers the backend's maximum one-hour delivery TTL.
      view:"overview", delay:() => Date.now() - started < 30000 ? 1000 : 5000, maxRefreshes:800,
      refresh: async current => {
        const epoch = panel._eocBroadcastEpoch;
        const result = await panel._hass.callWS({type:WS_BROADCAST, action:"snapshot"});
        if (!current() || panel._eocBroadcastEpoch !== epoch) return;
        panel._eocBroadcastSnapshot = result;
        // Preserve the message draft, selection and focus while status settles.
        const currentHost = panel.shadowRoot.querySelector("#broadcast-card");
        if (!currentHost) return;
        const history = currentHost.querySelector(".broadcast-history");
        const template = currentHost.ownerDocument.createElement("template");
        template.innerHTML = broadcastMarkup(panel, result);
        history?.replaceWith(template.content.querySelector(".broadcast-history"));
        if (!hasPendingBroadcast(result)) stopLiveStatus(panel, "broadcast");
      },
    });
  } else stopLiveStatus(panel, "broadcast");
}

function applyBroadcastError(panel, err) {
  if (panel._viewKey?.() !== "overview") return;
  const host = panel.shadowRoot.querySelector("#broadcast-card");
  if (!host) return;
  host.innerHTML = `<div class="error" role="alert">Unable to load Broadcast: ${panel._e(err.message || String(err))}</div>`;
}

async function loadBroadcast(panel) {
  try {
    const snapshot = await panel._hass.callWS({type: WS_BROADCAST, action: "snapshot"});
    applyBroadcastSnapshot(panel, snapshot);
  } catch (err) {
    applyBroadcastError(panel, err);
  }
}

export function bindBroadcast(panel, broadcastPromise) {
  ensureRouteStyle(panel);
  return Promise.resolve(broadcastPromise)
    .then((snapshot) => applyBroadcastSnapshot(panel, snapshot))
    .catch((err) => applyBroadcastError(panel, err))
    .finally(() => {
      if (panel._eocOverviewBroadcastPromise === broadcastPromise) panel._eocOverviewBroadcastPromise = null;
    });
}
