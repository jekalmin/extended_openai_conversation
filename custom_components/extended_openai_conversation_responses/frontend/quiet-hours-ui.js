export async function loadQuietHours(panel, silent = false) {
  const token = ++panel._loadToken;
  if (!silent) { panel._busy = true; panel._render(); }
  try {
    const result = await panel._call("quiet_hours", "get");
    if (token !== panel._loadToken) return;
    panel._result = result;
    if (panel._unsavedState?.scopes.get(VIEW)?.agent !== panel._agentId) panel._quietHoursDraft = JSON.parse(JSON.stringify(result.config || {}));
    panel._error = null;
  } catch (err) {
    if (token === panel._loadToken) panel._error = err.message || String(err);
  } finally {
    if (token === panel._loadToken) { panel._busy = false; panel._render(); }
  }
}

const VIEW = "capabilities/quiet-hours";
const QUIET_HOURS_STYLES = `
      .qh-satellite-summary{display:grid;gap:5px;overflow-wrap:anywhere;color:var(--secondary-text-color)}.qh-satellite-config{margin-top:12px;padding-top:6px}.qh-satellite-config>summary{min-height:42px}

      .qh-status,.qh-satellite-heading{display:flex;justify-content:space-between;gap:16px;align-items:center}.qh-status{padding-bottom:18px;border-bottom:1px solid var(--divider-color);margin-bottom:8px}.qh-status div,.qh-satellite-heading div{display:flex;flex-direction:column;gap:3px}.qh-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.qh-grid label{display:flex;flex-direction:column;gap:7px}.qh-grid label>span{font-weight:600}.qh-override{display:block;width:100%;min-width:0}.qh-grid small,.qh-entity-note small,.qh-satellite small{color:var(--secondary-text-color);line-height:1.45}.qh-policy{margin-top:20px}.qh-volume{display:flex;align-items:center;gap:12px}.qh-volume input{flex:1}.qh-volume output{min-width:44px;text-align:right;font-variant-numeric:tabular-nums}.qh-entity-note{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:20px;padding:14px;border-radius:10px;background:var(--secondary-background-color)}.qh-entity-note small{flex-basis:100%}.qh-satellites{display:grid;gap:14px}.qh-satellite{border:1px solid var(--divider-color);border-radius:12px;padding:16px}.qh-satellite-heading{margin-bottom:14px}.config-actions{display:flex;gap:10px;justify-content:flex-end;margin-top:16px}@media(max-width:760px){.qh-grid{grid-template-columns:1fr}.qh-status,.qh-satellite-heading{align-items:flex-start}}`;

function manualOverride(config, satelliteId) {
  return config?.overrides?.[satelliteId] || {};
}

function volumeStatus(panel, satellite) {
  const entityId = satellite.media_player_entity_id;
  if (!entityId) return {label: "No speaker found", ready: false};
  const state = panel._hass?.states?.[entityId];
  const volume = state?.attributes?.volume_level;
  if (!state || state.state === "unavailable" || state.state === "unknown" || typeof volume !== "number" || !Number.isFinite(volume)) {
    return {label: "Speaker unavailable", ready: false};
  }
  return {label: "Speaker ready", ready: true};
}

// Summaries use the draft's selected entities. Clearing a saved manual override
// does not guess what the backend will detect automatically after Save.
export function quietHoursSatellitePresentation(panel, satellite, config) {
  const override = manualOverride(config, satellite.satellite_entity_id);
  const saved = manualOverride(panel._result?.config, satellite.satellite_entity_id);
  const selected = (key, source) => {
    if (override[key]) return {id: override[key], manual: true, pending: false};
    if (saved[key]) return {id: "", manual: false, pending: true};
    return {id: satellite[key] || "", manual: source === "manual", pending: false};
  };
  const media = selected("media_player_entity_id", satellite.media_player_source);
  const wake = selected("wake_sound_entity_id", satellite.wake_sound_source);
  const label = (choice) => {
    if (choice.pending) return "Automatic — resolved after saving";
    const friendly = panel._hass?.states?.[choice.id]?.attributes?.friendly_name;
    const entity = friendly ? `${friendly} (${choice.id})` : choice.id;
    return `${choice.manual ? "Manual" : "Automatic"} · ${entity}`;
  };
  const speaker = media.pending ? {label: "Speaker selection pending save", ready: false}
    : volumeStatus(panel, {media_player_entity_id: media.id});
  const wakeState = panel._hass?.states?.[wake.id]?.state;
  const wakeUnavailable = Boolean(wake.id) && (!wakeState || ["unknown", "unavailable"].includes(wakeState));
  const candidates = satellite.wake_sound_candidates || [];
  const wakeReview = !wake.id && !wake.pending && candidates.length > 1;
  const wakeUsed = config.wake_sound !== "unchanged";
  return {
    media, wake, speaker,
    mediaSummary: media.id || media.pending ? label(media) : "No speaker selected",
    wakeSummary: wake.id ? `${label(wake)}${wakeUnavailable ? " — unavailable" : ""}`
      : wake.pending ? label(wake)
      : wakeReview ? `No automatic selection · ${candidates.length} switches available to review`
      : "No compatible wake-word switch detected (optional)",
    needsAttention: (!media.pending && !speaker.ready) || (wakeUsed && (wakeUnavailable || wakeReview)),
  };
}

function satelliteCard(panel, satellite, config) {
  const autoMedia = satellite.media_player_source === "auto" ? satellite.media_player_entity_id : null;
  const autoWake = satellite.wake_sound_source === "auto" ? satellite.wake_sound_entity_id : null;
  const mediaCandidates = satellite.media_player_candidates || [];
  const wakeCandidates = satellite.wake_sound_candidates || [];
  const presentation = quietHoursSatellitePresentation(panel, satellite, config);
  const speakerStatus = presentation.speaker;
  const mediaHelp = satellite.media_player_source === "manual"
    ? "Manually selected for this satellite."
    : autoMedia
      ? `Found automatically: ${panel._e(autoMedia)}`
      : mediaCandidates.length
        ? "No speaker was selected automatically. Choose one of the media players attached to this satellite's device."
        : "No media player entity is attached to this satellite's Home Assistant device.";
  const wakeHelp = satellite.wake_sound_source === "manual"
    ? "Manually selected for this satellite. Make sure this switch controls the chime played when the wake word is heard."
    : autoWake
      ? `Found automatically: ${panel._e(autoWake)}`
      : wakeCandidates.length
        ? "No wake-word switch was selected automatically. Choose one of the switches attached to this satellite's device."
        : "No switch entity is attached to this satellite's Home Assistant device. That is normal for many satellites.";
  return `<article class="qh-satellite" data-qh-satellite="${panel._e(satellite.satellite_entity_id)}">
    <div class="qh-satellite-heading">
      <div><strong>${panel._e(satellite.name)}</strong><small>${panel._e(satellite.satellite_entity_id)}</small></div>
      <span data-qh-speaker-status class="${speakerStatus.ready ? "availability-badge" : "disabled-badge"}">${speakerStatus.label}</span>
    </div>
    <div class="qh-satellite-summary"><span>Speaker: <span data-qh-media-summary>${panel._e(presentation.mediaSummary)}</span></span><span>Wake-word switch: <span data-qh-wake-summary>${panel._e(presentation.wakeSummary)}</span></span></div>
    <details class="qh-satellite-config eoc-details-base" ${presentation.needsAttention ? "open" : ""}><summary>Configure</summary><div class="qh-grid">
      <label><span>Speaker volume entity</span><ha-entity-picker class="qh-override" data-satellite="${panel._e(satellite.satellite_entity_id)}" data-kind="media_player_entity_id" data-domain="media_player" data-auto-entity="${panel._e(autoMedia || "")}"></ha-entity-picker><small>${mediaHelp}</small></label>
      <label><span>Wake-word sound switch</span><ha-entity-picker class="qh-override" data-satellite="${panel._e(satellite.satellite_entity_id)}" data-kind="wake_sound_entity_id" data-domain="switch" data-auto-entity="${panel._e(autoWake || "")}"></ha-entity-picker><small>${wakeHelp}</small></label>
    </div></details>
  </article>`;
}

export function renderQuietHours(panel) {
  const result = panel._result || {};
  const savedConfig = result.config || {};
  const config = panel._quietHoursDraft || savedConfig;
  const satellites = result.satellites || [];
  const maxPercent = Math.round(Number(config.max_volume ?? 0.2) * 100);
  const statusTitle = result.active ? "Quiet Hours active now" : savedConfig.enabled ? "Outside Quiet Hours" : "Quiet Hours schedule disabled";
  const statusDetail = savedConfig.enabled ? `${panel._e(savedConfig.start || "22:00")}–${panel._e(savedConfig.end || "07:00")} every day` : "The saved schedule is currently turned off.";
  return `<style>${QUIET_HOURS_STYLES}</style><section class="page-intro"><h1>Quiet Hours</h1><p>Make your Assist satellites quieter at set times each day. Quiet Hours can lower speaker volume and, where supported, change the sound played when the wake word is heard. For LEDs or other device-specific settings, use the Quiet Hours entity in a normal Home Assistant automation.</p></section>
    <section class="content-card">
      <div class="qh-status"><div><strong data-qh-live-title>${statusTitle}</strong><small>${statusDetail}</small></div><span data-qh-live-badge class="${result.active ? "availability-badge" : "disabled-badge"}">${result.active ? "Active" : "Inactive"}</span></div>
      <div class="config-toggle setting"><span class="setting-copy"><span class="setting-label-row"><label for="qh-enabled"><strong>Enable daily schedule</strong></label></span><small>When enabled, Quiet Hours starts and ends automatically at the times below. If you save while the current time is inside that period, it starts immediately.</small></span><label class="switch-control" for="qh-enabled"><input id="qh-enabled" type="checkbox" role="switch" ${config.enabled ? "checked" : ""}><span class="switch-track" aria-hidden="true"></span></label></div>
      <div class="qh-grid qh-policy">
        <label><span>Start</span><input id="qh-start" type="time" value="${panel._e(config.start || "22:00")}"></label>
        <label><span>End</span><input id="qh-end" type="time" value="${panel._e(config.end || "07:00")}"></label>
        <label><span>Maximum speaker volume</span><div class="qh-volume"><input id="qh-volume" type="range" min="0" max="100" step="1" value="${maxPercent}"><output id="qh-volume-value">${maxPercent}%</output></div><small>Quiet Hours only turns louder satellites down to this level. It never turns a quieter satellite up. This changes the satellite's media-player volume, so other audio from that speaker may also be quieter.</small></label>
        <label><span>Wake-word sound</span><select id="qh-wake"><option value="off" ${config.wake_sound === "off" ? "selected" : ""}>Off during Quiet Hours</option><option value="on" ${config.wake_sound === "on" ? "selected" : ""}>On during Quiet Hours</option><option value="unchanged" ${config.wake_sound === "unchanged" ? "selected" : ""}>Don't change it</option></select><small>This is the chime played when a satellite hears its wake word. Quiet Hours changes it only where a compatible switch is found or manually selected.</small></label>
      </div>
      <div class="qh-entity-note"><strong>Use Quiet Hours in automations</strong><code>${panel._e(result.state_entity_id || "binary_sensor.extended_openai_quiet_hours")}</code><small>This read-only entity is on for the whole Quiet Hours period, even if no speaker needs changing. You can use it to control LEDs or anything else in your own Home Assistant automations. The saved schedule can also be turned on or off with the Extended OpenAI actions <code>enable_quiet_hours</code> and <code>disable_quiet_hours</code>.</small></div>
    </section>
    <section class="content-card"><div class="section-heading"><div><h2>Assist satellites</h2><p>Quiet Hours tries to find each satellite's speaker and wake-word sound automatically. If it picks the wrong entity, or cannot find one, choose the correct entity below.</p></div></div>${satellites.length ? `<div class="qh-satellites">${satellites.map((satellite) => satelliteCard(panel, satellite, config)).join("")}</div>` : `<p class="empty">No Assist satellites were found. Quiet Hours will keep checking and will pick them up when they become available.</p>`}</section>
    `;
}

function setOverride(panel, satelliteId, kind, value) {
  const draft = panel._quietHoursDraft;
  if (!draft) return;
  draft.overrides ||= {};
  const current = {...(draft.overrides[satelliteId] || {})};
  if (value) current[kind] = value;
  else delete current[kind];
  if (current.media_player_entity_id || current.wake_sound_entity_id) draft.overrides[satelliteId] = current;
  else delete draft.overrides[satelliteId];
}

function refreshSatelliteSummary(panel, card, satellite) {
  if (!card || !satellite) return;
  const result = quietHoursSatellitePresentation(panel, satellite, panel._quietHoursDraft || {});
  card.querySelector("[data-qh-media-summary]").textContent = result.mediaSummary;
  card.querySelector("[data-qh-wake-summary]").textContent = result.wakeSummary;
  const status = card.querySelector("[data-qh-speaker-status]");
  status.textContent = result.speaker.label;
  status.className = result.speaker.ready ? "availability-badge" : "disabled-badge";
  if (result.needsAttention) card.querySelector(".qh-satellite-config").open = true;
}

export function bindQuietHours(panel) {
  const root = panel.shadowRoot;
  if (!root.__eocQuietSummaryBound) {
    root.__eocQuietSummaryBound = true;
    let snapshot = panel._result;
    // Page saves preserve the route DOM. Refresh resolved entity summaries when
    // the save coordinator announces dirty-state changes with a new snapshot.
    root.addEventListener("eoc-config-dirty-changed", () => {
      if (panel._viewKey?.() !== VIEW || snapshot === panel._result) return;
      snapshot = panel._result;
      for (const card of root.querySelectorAll("[data-qh-satellite]")) {
        const satellite = (snapshot?.satellites || []).find(item => item.satellite_entity_id === card.dataset.qhSatellite);
        if (!satellite) continue;
        refreshSatelliteSummary(panel, card, satellite);
        for (const picker of card.querySelectorAll(".qh-override")) {
          const source = picker.dataset.kind === "media_player_entity_id" ? satellite.media_player_source : satellite.wake_sound_source;
          const automatic = source === "auto" ? satellite[picker.dataset.kind] || "" : "";
          picker.dataset.autoEntity = automatic;
          picker.placeholder = automatic ? `Automatic · ${automatic}` : "Automatic";
        }
      }
    });
  }
  root.querySelector("#qh-enabled")?.addEventListener("change", (event) => { panel._quietHoursDraft.enabled = event.target.checked; });
  root.querySelector("#qh-start")?.addEventListener("input", (event) => { panel._quietHoursDraft.start = event.target.value; });
  root.querySelector("#qh-end")?.addEventListener("input", (event) => { panel._quietHoursDraft.end = event.target.value; });
  root.querySelector("#qh-wake")?.addEventListener("change", (event) => {
    panel._quietHoursDraft.wake_sound = event.target.value;
    root.querySelectorAll("[data-qh-satellite]").forEach((card) => {
      const satellite = (panel._result?.satellites || []).find((item) => item.satellite_entity_id === card.dataset.qhSatellite);
      refreshSatelliteSummary(panel, card, satellite);
    });
  });
  const volume = root.querySelector("#qh-volume");
  volume?.addEventListener("input", (event) => {
    const percent = Number(event.target.value);
    panel._quietHoursDraft.max_volume = percent / 100;
    const output = root.querySelector("#qh-volume-value");
    if (output) output.textContent = `${percent}%`;
  });
  const satellites = new Map((panel._result?.satellites || []).map((satellite) => [satellite.satellite_entity_id, satellite]));
  root.querySelectorAll(".qh-override").forEach((picker) => {
    const satellite = satellites.get(picker.dataset.satellite);
    const candidates = picker.dataset.kind === "media_player_entity_id"
      ? satellite?.media_player_candidates || []
      : satellite?.wake_sound_candidates || [];
    const selected = manualOverride(panel._quietHoursDraft, picker.dataset.satellite)?.[picker.dataset.kind] || "";
    picker.hass = panel._hass;
    picker.value = selected;
    picker.includeDomains = [picker.dataset.domain];
    picker.includeEntities = selected && !candidates.includes(selected)
      ? [...candidates, selected]
      : candidates;
    picker.allowCustomEntity = false;
    picker.placeholder = picker.dataset.autoEntity ? `Automatic · ${picker.dataset.autoEntity}` : "Automatic";
    picker.addEventListener("value-changed", (event) => {
      const value = String(event?.detail?.value || picker.value || "");
      picker.value = value;
      setOverride(panel, picker.dataset.satellite, picker.dataset.kind, value);
      refreshSatelliteSummary(panel, picker.closest(".qh-satellite"), (panel._result?.satellites || []).find(item => item.satellite_entity_id === picker.dataset.satellite));
    });
  });

}
