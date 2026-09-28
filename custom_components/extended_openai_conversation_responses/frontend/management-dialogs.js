// Core dialog handlers are delegated from the host so owned dialogs can be
// created on demand and removed when their route or agent changes.
export const PERSISTENT_DIALOGS = new Set([
  "knowledge-dialog", "memory-dialog", "session-dialog", "reassign-dialog", "confirm-dialog",
]);
const RETAINED_DIALOGS = new Set([...PERSISTENT_DIALOGS, "temporary-memory-dialog"]);

export function bindPanelDialogs(panel) {
  const root = panel.shadowRoot;
  if (root.__eocDialogsBound) return;
  root.__eocDialogsBound = true;
  root.addEventListener("click", (event) => {
    const button = event.target?.closest?.("button");
    const dialog = button?.closest?.("dialog");
    if (!PERSISTENT_DIALOGS.has(dialog?.id)) return;
    const actions = {
      "confirm-cancel": () => panel._resolveConfirm(false),
      "confirm-accept": () => panel._resolveConfirm(true),
      "knowledge-delete": () => panel._deleteSource(panel._editingSource?.source_id, true, button),
      "memory-delete": () => panel._deleteMemory(panel._editingMemory?.memory_id, true, button),
      "reassign-cancel": () => dialog.close(),
      "reassign-save": () => panel._saveReassign(button),
    };
    if (actions[button.id]) actions[button.id]();
    else if (button.classList.contains("close-editor")) {
      if (button.classList.contains("icon")) panel._requestEditorClose();
      else dialog.close();
    }
    else if (button.classList.contains("close-session")) dialog.close();
  });
  root.addEventListener("submit", (event) => {
    if (!["knowledge-form", "memory-form"].includes(event.target?.id)) return;
    event.preventDefault();
    if (event.target.id === "knowledge-form") void panel._saveKnowledge();
    else void panel._saveMemory();
  });
  root.addEventListener("input", (event) => {
    if (event.target?.id === "knowledge-content") panel._updateKnowledgeCounter();
  });
  root.addEventListener("cancel", (event) => {
    const dialog = event.target;
    if (!PERSISTENT_DIALOGS.has(dialog?.id)) return;
    event.preventDefault();
    if (dialog.id === "confirm-dialog") panel._resolveConfirm(false);
    else if (["knowledge-dialog", "memory-dialog"].includes(dialog.id)) panel._requestEditorClose();
    else dialog.close();
  }, true);
}

export function updateDialogs(panel, markup, {preserveEditors = false} = {}) {
  const root = panel.shadowRoot;
  const host = root.querySelector("#eoc-dialog-host");
  if (!host) return;
  const template = document.createElement("template");
  template.innerHTML = markup;
  const desired = new Set([...template.content.children].map(child => child.id));
  for (const child of [...host.children]) {
    if (!desired.has(child.id) || (!preserveEditors && !RETAINED_DIALOGS.has(child.id))) child.remove();
  }
  for (const child of [...template.content.children]) {
    if (!host.querySelector(`#${child.id}`)) host.append(child);
  }
}

export function knowledgeSourceAvailabilityControl() {
  return `<div class="config-toggle setting knowledge-source-availability-setting"><span class="setting-copy"><span class="setting-label-row"><label for="knowledge-source-enabled"><strong>Available to the assistant</strong></label></span><small>Turn this off to keep the source stored locally without including it in Knowledge retrieval.</small></span><label class="switch-control" for="knowledge-source-enabled"><input id="knowledge-source-enabled" type="checkbox" role="switch" checked><span class="switch-track" aria-hidden="true"></span></label></div>`;
}
