import {restoreScopeMarkup} from "./management-decision-guidance.js";
import {TRANSFER_SECTIONS} from "./backup-transfer-shell.js";
export {backupSummaryLines} from "./backup-summary.js";
export {TRANSFER_SECTIONS, renderBackupTransferPanel} from "./backup-transfer-shell.js";
export const WS_BACKUP_TRANSFER = "extended_openai_conversation_responses/management/backup_transfer";

const CREATE_ID = "create-backup-transfer";
const RESTORE_ID = "restore-backup-transfer";
const FILE_ID = "backup-file-transfer";
const CANCEL_ID = "restore-transfer-cancel";
const APPLY_ID = "restore-transfer-apply";
const EXPORT_MODE_ID = "transfer-export-mode";
const CUSTOM_OPTIONS_ID = "transfer-custom-options";
const RESTORE_SECTIONS_ID = "restore-transfer-sections";
const RESTORE_STATUS_ID = "restore-transfer-status";
const IMPORT_PREVIEW_DEBOUNCE_MS = 150;

const SECTION_LABELS = Object.freeze(Object.fromEntries(TRANSFER_SECTIONS));

function selectedAgent(panel) {
  const agent = panel?._selectedAgent?.();
  if (!agent?.entry_id || !agent?.subentry_id) throw new Error("Select a conversation agent first");
  return agent;
}

export async function callBackupTransfer(panel, action, data = {}) {
  if (!panel?._hass?.callWS) throw new Error("Home Assistant connection is unavailable");
  const agent = selectedAgent(panel);
  return panel._hass.callWS({
    type: WS_BACKUP_TRANSFER,
    action,
    entry_id: agent.entry_id,
    subentry_id: agent.subentry_id,
    data,
  });
}

export function bytesToBase64(bytes) {
  const source = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let binary = "";
  const block = 0x8000;
  for (let offset = 0; offset < source.length; offset += block) {
    binary += String.fromCharCode(...source.subarray(offset, offset + block));
  }
  return globalThis.btoa(binary);
}

export function base64ToBytes(value) {
  const binary = globalThis.atob(String(value || ""));
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

async function defaultSaveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  try {
    const link = document.createElement("a");
    link.href = url;
    link.download = filename || "conversation-agent-export";
    link.click();
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}

export async function downloadSetupExport(panel, saveBlob = defaultSaveBlob) {
  const result = await callBackupTransfer(panel, "setup_export");
  if (!result?.json || !result?.filename) throw new Error("The server returned an invalid setup export");
  const blob = new Blob([result.json], {type: "application/json"});
  await saveBlob(blob, result.filename, result);
  return result;
}

export async function downloadArchiveBackup(panel, {mode = "full", sections = null} = {}, saveBlob = defaultSaveBlob) {
  let sessionId = null;
  try {
    const data = {mode};
    if (sections != null) data.sections = sections;
    const manifest = await callBackupTransfer(panel, "export_start", data);
    sessionId = manifest.session_id;
    if (!sessionId || !Number.isInteger(manifest.chunk_count) || manifest.chunk_count < 1 || !Number.isInteger(manifest.size) || manifest.size < 1) {
      throw new Error("The backup server returned an invalid transfer manifest");
    }
    const parts = [];
    let received = 0;
    for (let index = 0; index < manifest.chunk_count; index += 1) {
      const chunk = await callBackupTransfer(panel, "export_chunk", {session_id: sessionId, index});
      if (chunk.session_id !== sessionId || chunk.index !== index || chunk.offset !== received || !Number.isInteger(chunk.bytes) || chunk.bytes < 1) {
        throw new Error("The backup server returned an out-of-order chunk");
      }
      const bytes = base64ToBytes(chunk.data);
      if (bytes.byteLength !== chunk.bytes || received + bytes.byteLength > manifest.size) {
        throw new Error("The backup server returned an incomplete or oversized chunk");
      }
      parts.push(bytes);
      received += bytes.byteLength;
    }
    if (received !== manifest.size) throw new Error("The backup download is incomplete");
    const blob = new Blob(parts, {type: manifest.content_type || "application/zip"});
    if (blob.size !== manifest.size) throw new Error("The backup download is incomplete");
    await saveBlob(blob, manifest.filename, manifest);
    return manifest;
  } finally {
    if (sessionId) {
      try {
        await callBackupTransfer(panel, "export_cancel", {session_id: sessionId});
      } catch (_err) {
        // The server expires abandoned sessions. Cleanup failure after a complete
        // browser download must not make the successful export appear to fail.
      }
    }
  }
}

export async function downloadFullBackup(panel, saveBlob = defaultSaveBlob) {
  return downloadArchiveBackup(panel, {mode: "full"}, saveBlob);
}

export async function downloadCustomBackup(panel, sections, saveBlob = defaultSaveBlob) {
  if (!Array.isArray(sections) || !sections.length) throw new Error("Select at least one section for the custom backup");
  return downloadArchiveBackup(panel, {mode: "custom", sections}, saveBlob);
}

export async function uploadFullBackup(panel, file) {
  let sessionId = null;
  try {
    const manifest = await callBackupTransfer(panel, "import_start", {
      filename: file.name || "backup",
      size: file.size,
    });
    sessionId = manifest.session_id;
    if (!sessionId || !Number.isInteger(manifest.chunk_size) || manifest.chunk_size < 1 || !Number.isInteger(manifest.chunk_count) || manifest.chunk_count < 1) {
      throw new Error("The backup server returned an invalid upload manifest");
    }
    let sent = 0;
    for (let index = 0; index < manifest.chunk_count; index += 1) {
      const end = Math.min(file.size, sent + manifest.chunk_size);
      const bytes = new Uint8Array(await file.slice(sent, end).arrayBuffer());
      if (!bytes.byteLength) throw new Error("The selected transfer file ended unexpectedly");
      const result = await callBackupTransfer(panel, "import_chunk", {
        session_id: sessionId,
        index,
        data: bytesToBase64(bytes),
      });
      sent += bytes.byteLength;
      if (result.session_id !== sessionId || result.next_index !== index + 1 || result.received !== sent) {
        throw new Error("The backup server rejected the upload sequence");
      }
    }
    if (sent !== file.size) throw new Error("The selected transfer file was not uploaded completely");
    const inspection = await callBackupTransfer(panel, "import_inspect", {session_id: sessionId});
    if (!inspection?.valid) throw new Error("The transfer could not be validated");
    return {session_id: sessionId, ...inspection};
  } catch (err) {
    if (sessionId) {
      try {
        await callBackupTransfer(panel, "import_cancel", {session_id: sessionId});
      } catch (_cleanupErr) {
        // Server-side expiry is the final cleanup fallback.
      }
    }
    throw err;
  }
}

export async function cancelBackupImport(panel, sessionId = panel?._backupTransferSession) {
  invalidateImportPreview(panel);
  if (!sessionId) return;
  if (panel?._backupTransferSession === sessionId) panel._backupTransferSession = null;
  try {
    await callBackupTransfer(panel, "import_cancel", {session_id: sessionId});
  } catch (_err) {
    // Closing the dialog should remain reliable; the server expires the session.
  }
}

export function openBackupPicker(panel, input) {
  invalidateImportPreview(panel);
  const previousSession = panel?._backupTransferSession;
  if (previousSession && panel) panel._backupTransferSession = null;
  input.value = "";
  // Keep the picker invocation in the original click task. Awaiting server cleanup
  // first can consume the browser's transient user-activation permission.
  input.click();
  if (previousSession) void cancelBackupImport(panel, previousSession);
}

function sectionChoices(panel, {className, checked = true, available = null} = {}) {
  const allowed = available ? new Set(available) : null;
  return TRANSFER_SECTIONS
    .filter(([key]) => !allowed || allowed.has(key))
    .map(([key, label]) => `<label class="group-function-choice"><input type="checkbox" class="${className}" value="${panel?._e ? panel._e(key) : key}" ${checked ? "checked" : ""}><span><strong>${panel?._e ? panel._e(label) : label}</strong></span></label>`)
    .join("");
}

export function renderRestoreTransferDialog(panel) {
  const dirtyWarning = panel?._configDirty
    ? '<p class="inline-error">Your unsaved configuration changes will be discarded if configuration is one of the restored sections.</p>'
    : "";
  return `<dialog id="restore-dialog" class="editor-dialog" aria-labelledby="restore-dialog-title"><div class="dialog-header"><h2 id="restore-dialog-title">Import / Restore</h2></div><div class="dialog-body">${restoreScopeMarkup(panel)}<div><strong id="restore-backup-name"></strong><p id="restore-backup-meta" class="meta"></p></div><ul id="restore-summary" class="restore-summary"></ul><fieldset class="setting-group"><legend><strong>Sections to replace</strong></legend><p class="help">Only sections contained in this file are shown. Unselected destination sections remain unchanged.</p><div id="${RESTORE_SECTIONS_ID}" class="group-function-choices"></div></fieldset><div id="${RESTORE_STATUS_ID}" class="validation" role="status" aria-live="polite"></div><div class="notice"><strong>Replacement, not merge</strong><p>Every selected section replaces that section on the current agent. The combined target is validated first, including Request Rule references to Function Tools. A final confirmation is required before applying it.</p></div>${dirtyWarning}</div><div class="dialog-actions"><button type="button" class="secondary" id="${CANCEL_ID}">Cancel</button><button type="button" class="danger" id="${APPLY_ID}" disabled>Restore selected sections</button></div></dialog>`;
}

function selectedValues(root, selector) {
  return [...root.querySelectorAll(selector)].filter((item) => item.checked).map((item) => item.value);
}

function formatSourceKind(value) {
  return {
    portable_transfer: "Shareable Setup",
    custom_backup: "Custom Backup",
    full_backup: "Full Backup",
    legacy_setup: "Legacy setup export",
  }[value] || "Extended OpenAI transfer";
}

function transferSummaryLines(result, summaryFormatter) {
  const summary = result?.summary || {};
  const lines = [formatSourceKind(result?.source_kind)];
  const configured = summaryFormatter(summary).filter(Boolean);
  for (const line of configured) if (!lines.includes(line)) lines.push(line);
  return lines;
}

function updateSensitiveStatus(panel, preview) {
  const root = panel.shadowRoot;
  const status = root.querySelector(`#${RESTORE_STATUS_ID}`);
  if (!status) return;
  const preserved = Number(preview?.preserved_sensitive_field_count || 0);
  const missing = Number(preview?.missing_sensitive_field_count || 0);
  if (missing) {
    status.className = "validation error";
    status.textContent = `${missing} redacted secret field${missing === 1 ? "" : "s"} cannot be preserved from the destination and will need to be entered after restore.${preserved ? ` ${preserved} existing secret field${preserved === 1 ? " was" : "s were"} preserved.` : ""}`;
  } else if (preserved) {
    status.className = "validation success";
    status.textContent = `${preserved} redacted secret field${preserved === 1 ? " was" : "s were"} preserved from the current destination configuration.`;
  } else {
    status.className = "validation success";
    status.textContent = "The selected sections passed validation.";
  }
}

function previewWork(panel) {
  return panel._backupTransferPreviewWork ||= {timer: null, inFlight: false, latest: null};
}

function invalidateImportPreview(panel) {
  if (!panel) return 0;
  const work = previewWork(panel);
  if (work.timer !== null) clearTimeout(work.timer);
  work.timer = null;
  work.latest = null;
  panel._backupTransferPreview = null;
  panel._backupTransferPreviewToken = null;
  const apply = panel.shadowRoot?.querySelector(`#${APPLY_ID}`);
  if (apply) apply.disabled = true;
  return panel._transferPreviewToken = (panel._transferPreviewToken || 0) + 1;
}

function sendLatestImportPreview(panel) {
  const work = previewWork(panel);
  if (work.inFlight || work.timer !== null || !work.latest) return;
  const request = work.latest;
  work.latest = null;
  if (panel._backupTransferSession !== request.sessionId || panel._transferPreviewToken !== request.token) return;
  work.inFlight = true;
  void (async () => {
    const status = panel.shadowRoot?.querySelector(`#${RESTORE_STATUS_ID}`);
    const apply = panel.shadowRoot?.querySelector(`#${APPLY_ID}`);
    try {
      const result = await callBackupTransfer(panel, "import_inspect", {session_id: request.sessionId, sections: request.sections});
      if (panel._transferPreviewToken !== request.token || panel._backupTransferSession !== request.sessionId) return;
      panel._backupTransferPreview = result.preview;
      panel._backupTransferPreviewToken = result.preview_token;
      updateSensitiveStatus(panel, result.preview);
      if (apply) apply.disabled = false;
    } catch (err) {
      if (panel._transferPreviewToken !== request.token || panel._backupTransferSession !== request.sessionId) return;
      panel._backupTransferPreview = null;
      if (status) {
        status.className = "validation error";
        status.textContent = err.message || String(err);
      }
      if (apply) apply.disabled = true;
    } finally {
      work.inFlight = false;
      sendLatestImportPreview(panel);
    }
  })();
}

export function refreshImportPreview(panel) {
  const root = panel.shadowRoot;
  const token = invalidateImportPreview(panel);
  const sessionId = panel._backupTransferSession;
  if (!sessionId) return false;
  const sections = selectedValues(root, ".transfer-restore-section");
  const status = root.querySelector(`#${RESTORE_STATUS_ID}`);
  if (!sections.length) {
    if (status) {
      status.className = "validation error";
      status.textContent = "Select at least one section to restore.";
    }
    return false;
  }
  if (status) {
    status.className = "validation";
    status.textContent = "Validating selected sections…";
  }
  const work = previewWork(panel);
  work.latest = {sessionId, sections, token};
  work.timer = setTimeout(() => {
    work.timer = null;
    sendLatestImportPreview(panel);
  }, IMPORT_PREVIEW_DEBOUNCE_MS);
  return true;
}

export function bindBackupTransfer(panel, summaryFormatter = () => []) {
  const root = panel?.shadowRoot;
  if (!root) return;

  const exportMode = root.querySelector(`#${EXPORT_MODE_ID}`);
  const customOptions = root.querySelector(`#${CUSTOM_OPTIONS_ID}`);
  const syncExportMode = () => {
    if (customOptions) customOptions.hidden = exportMode?.value !== "custom";
  };
  exportMode?.addEventListener("change", syncExportMode);
  syncExportMode();

  root.querySelector(`#${CREATE_ID}`)?.addEventListener("click", async () => {
    const button = root.querySelector(`#${CREATE_ID}`);
    const mode = exportMode?.value || "setup";
    const sections = selectedValues(root, ".transfer-custom-section");
    if (mode === "custom" && !sections.length) {
      panel._toast("Select at least one section for the custom backup", true);
      return;
    }
    panel._setSaving(button, true);
    try {
      if (mode === "setup") await downloadSetupExport(panel);
      else if (mode === "custom") await downloadCustomBackup(panel, sections);
      else await downloadFullBackup(panel);
      panel._toast(mode === "setup" ? "Shareable setup exported" : mode === "custom" ? "Custom backup created" : "Full backup created");
    } catch (err) {
      panel._toast(`Unable to create export: ${err.message || String(err)}`, true);
    } finally {
      panel._setSaving(button, false);
    }
  });

  root.querySelector(`#${RESTORE_ID}`)?.addEventListener("click", () => {
    openBackupPicker(panel, root.querySelector(`#${FILE_ID}`));
  });

  root.querySelector(`#${FILE_ID}`)?.addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const apply = root.querySelector(`#${APPLY_ID}`);
    apply.disabled = true;
    await cancelBackupImport(panel);
    try {
      const result = await uploadFullBackup(panel, file);
      panel._backupTransferSession = result.session_id;
      panel._backupTransferInspection = result;
      panel._backupTransferPreview = result.preview;
      panel._backupTransferPreviewToken = result.preview_token;
      root.querySelector("#restore-backup-name").textContent = result.title;
      const created = result.summary?.created_at ? new Date(result.summary.created_at).toLocaleString() : "legacy export";
      const version = result.summary?.integration_version ? ` · integration ${result.summary.integration_version}` : "";
      root.querySelector("#restore-backup-meta").textContent = `${formatSourceKind(result.source_kind)} · ${created}${version}`;
      root.querySelector("#restore-summary").innerHTML = transferSummaryLines(result, summaryFormatter).map((line) => `<li>${panel._e(line)}</li>`).join("");
      const sectionRoot = root.querySelector(`#${RESTORE_SECTIONS_ID}`);
      sectionRoot.innerHTML = sectionChoices(panel, {className: "transfer-restore-section", available: result.available_sections});
      sectionRoot.querySelectorAll(".transfer-restore-section").forEach((input) => input.addEventListener("change", () => void refreshImportPreview(panel)));
      updateSensitiveStatus(panel, result.preview);
      apply.disabled = false;
      root.querySelector("#restore-dialog").showModal();
    } catch (err) {
      panel._backupTransferSession = null;
      panel._backupTransferInspection = null;
      panel._backupTransferPreview = null;
      panel._backupTransferPreviewToken = null;
      apply.disabled = true;
      panel._toast(err.message || String(err), true);
    }
  });

  root.querySelector(`#${CANCEL_ID}`)?.addEventListener("click", async () => {
    await cancelBackupImport(panel);
    panel._backupTransferInspection = null;
    panel._backupTransferPreview = null;
    root.querySelector("#restore-dialog").close();
  });

  root.querySelector(`#${APPLY_ID}`)?.addEventListener("click", async () => {
    const sessionId = panel._backupTransferSession;
    if (!sessionId) return;
    const sections = selectedValues(root, ".transfer-restore-section");
    if (!sections.length) return;
    const labels = sections.map((key) => SECTION_LABELS[key] || key).join(", ");
    const previewToken = panel._transferPreviewToken;
    const serverPreviewToken = panel._backupTransferPreviewToken;
    if (!serverPreviewToken) return;
    if (typeof panel._confirm === "function") {
      const losesDraft = panel._configDirty && panel._draftAgentId === panel._agentId && sections.includes("configuration");
      const confirmed = await panel._confirm(
        "Restore selected sections?",
        `This will replace, not merge, the following sections on the current agent: ${labels}.${losesDraft ? " Your unsaved configuration changes will be replaced and lost." : ""}`,
        "Restore",
      );
      if (!confirmed) return;
    }
    if (panel._backupTransferSession !== sessionId || panel._transferPreviewToken !== previewToken || root.querySelector(`#${APPLY_ID}`)?.disabled) return;
    const button = root.querySelector(`#${APPLY_ID}`);
    panel._setSaving(button, true);
    try {
      await callBackupTransfer(panel, "import_restore", {session_id: sessionId, sections, preview_token: serverPreviewToken});
      panel._backupTransferSession = null;
      panel._backupTransferInspection = null;
      panel._backupTransferPreview = null;
      panel._backupTransferPreviewToken = null;
      root.querySelector("#restore-dialog").close();
      if (sections.includes("configuration")) panel._clearConfigDraft();
      await panel._loadAgents(panel._agentId);
      panel._toast("Selected sections restored");
    } catch (err) {
      panel._backupTransferSession = null;
      button.disabled = true;
      panel._toast(`Unable to restore: ${err.message || String(err)} Re-select the transfer file to retry.`, true);
    } finally {
      panel._setSaving(button, false);
    }
  });
}
