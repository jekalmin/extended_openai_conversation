import {filterHATools, isHALlmTool} from "./ha-llm-tools-list.js";
export {filterHATools, haToolName, isHALlmTool, renderHAToolCard, toolDescription} from "./ha-llm-tools-list.js";
export function bindHALlmTools(panel, synchronize) {
  if (!panel?._draft) return;
  const root = panel.shadowRoot;
  const host = root.querySelector(".tools-surface");
  if (!host || host.__eocHaBound) return;
  host.__eocHaBound = true;
  const agentId = panel._agentId;
  const load = () => {
    if (panel._haCatalogLoad?.agentId === agentId) return panel._haCatalogLoad.promise;
    const request = (async () => {
      const catalog = await panel._call("tools", "ha_catalog");
      if (panel._agentId !== agentId) throw new Error("The selected agent changed");
      panel._haCatalog = catalog;
      panel._haCatalogAgent = agentId;
      panel._haCatalogLoadedAt = Date.now();
      return catalog;
    })();
    const tracked = request.finally(() => {
      if (panel._haCatalogLoad?.promise === tracked) panel._haCatalogLoad = null;
    });
    panel._haCatalogLoad = {agentId, promise: tracked};
    return tracked;
  };
  if ((panel._draft.functions || []).some(isHALlmTool) && (panel._haCatalogAgent !== agentId || Date.now() - (panel._haCatalogLoadedAt || 0) > 30000) && !panel._haCatalogLoading) {
    panel._haCatalogLoading = true;
    load().then(() => { if (!root.querySelector("dialog[open]")) panel._render(); }).catch(err => panel._toast(err.message, true)).finally(() => { panel._haCatalogLoading = false; });
  }
  root.querySelector("#refresh-ha-tools")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    if (button.disabled) return;
    button.disabled = true;
    try { await load(); panel._render(); } catch (err) { panel._toast(err.message, true); }
    finally { if (button.isConnected) button.disabled = false; }
  });
  root.querySelector("#add-ha-tools")?.addEventListener("click", async () => {
    const existing = root.querySelector('dialog[data-ha-llm-tools-dialog][open]');
    if (existing) { existing.focus?.(); return; }
    const dialog = document.createElement("dialog");
    dialog.className = "editor-dialog";
    dialog.dataset.haLlmToolsDialog = "";
    dialog.setAttribute("aria-label", "Add Home Assistant LLM Tools");
    dialog.innerHTML = `<div class="dialog-header"><h2>Add LLM Tools</h2></div><div class="dialog-body">
      <p>These capabilities are supplied by Home Assistant or installed integrations/services. A source can be an integration's contribution or a complete LLM API, including an MCP server.</p>
      <p>Adding all saves the individual tools selected now. Future tools require explicit addition. Many schemas can increase input tokens; use Function Groups to load them when needed. HA tools are unavailable in Guest Mode.</p>
      <p role="status" data-status>Loading available tools…</p>
      <label>Search<input type="search" data-search></label><label>Sources<select multiple data-sources aria-label="Filter by sources"></select><small>Leave sources unselected to show all sources.</small></label>
      <label>Function Group<select data-group><option value="">Available on every request</option>${(panel._draft.function_groups || []).map(group => `<option value="${panel._e(group.id)}">${panel._e(group.name)}</option>`).join("")}</select></label>
      <button type="button" class="secondary" data-all>Select all shown</button><button type="button" class="secondary" data-clear>Clear selection</button><div data-tools></div></div>
      <div class="dialog-actions"><button type="button" class="secondary" data-cancel>Cancel</button><button type="button" data-add disabled>Add selected tools</button></div>`;
    root.append(dialog);
    dialog.showModal();
    dialog.addEventListener("close", () => dialog.remove());
    dialog.querySelector("[data-cancel]").onclick = () => dialog.close();
    try {
      const catalog = await load();
      if (!dialog.open) return;
      const selected = new Set();
      const available = catalog.tools || [];
      const indexByTool = new Map(available.map((tool, index) => [tool, index]));
      const status = dialog.querySelector("[data-status]");
      status.textContent = catalog.unavailable_sources?.length ? "Some sources are unavailable. Other tools can still be added." : "Preview uses your administrator context. Actual requests resolve tools with the caller's context and permissions.";
      dialog.querySelector("[data-sources]").innerHTML = [...new Set(available.map(tool => tool.source))].sort().map(source => `<option value="${panel._e(source)}">${panel._e(source)}</option>`).join("");
      const visible = () => filterHATools(available, dialog.querySelector("[data-search]").value, [...dialog.querySelector("[data-sources]").selectedOptions].map(option => option.value));
      const render = () => {
        dialog.querySelector("[data-tools]").innerHTML = visible().map(tool => {
          const index = indexByTool.get(tool);
          return `<label class="group-function-choice"><input type="checkbox" data-index="${index}" ${selected.has(index) ? "checked" : ""} ${tool.already_added ? "disabled" : ""}><span><strong>${panel._e(tool.name)}${tool.already_added ? " · Already added" : ""}</strong><small>${panel._e(tool.source)} · ${panel._e(tool.description)}</small></span></label>`;
        }).join("") || "No matching tools are available in this context.";
        dialog.querySelector("[data-add]").disabled = selected.size === 0;
        dialog.querySelector("[data-add]").textContent = `Add ${selected.size} selected tools`;
      };
      dialog.querySelector("[data-search]").oninput = render;
      dialog.querySelector("[data-sources]").onchange = render;
      dialog.querySelector("[data-tools]").onchange = event => {
        const index = Number(event.target.dataset.index);
        if (event.target.checked) selected.add(index); else selected.delete(index);
        dialog.querySelector("[data-add]").disabled = selected.size === 0;
        dialog.querySelector("[data-add]").textContent = `Add ${selected.size} selected tools`;
      };
      dialog.querySelector("[data-all]").onclick = () => { for (const tool of visible()) if (!tool.already_added) selected.add(indexByTool.get(tool)); render(); };
      dialog.querySelector("[data-clear]").onclick = () => { selected.clear(); render(); };
      dialog.querySelector("[data-add]").onclick = async () => {
        const button = dialog.querySelector("[data-add]");
        if (button.disabled) return;
        button.disabled = true;
        try {
          if (panel._agentId !== agentId) throw new Error("The selected agent changed");
          const result = await panel._call("tools", "ha_add", {tools: [...selected].map(index => available[index].reference), group_id: dialog.querySelector("[data-group]").value});
          if (panel._agentId !== agentId) return;
          synchronize(panel, result);
          for (const index of selected) available[index].already_added = true;
          if (result.ha_saved) {
            panel._haCatalog = {...catalog, saved: {...catalog.saved, ...result.ha_saved}};
            panel._haCatalogAgent = agentId;
            panel._haCatalogLoadedAt = Date.now();
          }
          dialog.close();
          panel._toast("HA LLM Tool references added");
          panel._render();
        } catch (err) { status.textContent = err.message || String(err); button.disabled = false; }
      };
      render();
    } catch (err) { dialog.querySelector("[data-status]").textContent = err.message || String(err); }
  });
}
