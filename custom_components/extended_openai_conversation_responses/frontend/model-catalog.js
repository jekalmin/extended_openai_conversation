// All model matching and capability choices come from the Python catalogue.
export async function lookupModelData(panel, model = "", action = "lookup", isCurrent = () => true) {
  const result = await panel._hass.callWS({
    type: "extended_openai_conversation_responses/model_catalog", action, model,
  });
  if (panel && result && isCurrent()) {
    panel._modelCatalogData = {...result, requested_model: String(model || "")};
    if (panel._result && result.model_capabilities) panel._result.model_capabilities = result.model_capabilities;
  }
  return result;
}

export function parameterControlState(capability = {}, effort = null, configuredValue = null) {
  const support = capability?.support || "undocumented";
  const configured = configuredValue !== undefined && configuredValue !== null && configuredValue !== "";
  if (support === "always") return {visible: true, enabled: true, inactive: false, reason: ""};
  if (support === "conditional") {
    const allowed = Array.isArray(capability.allowed_reasoning_efforts) ? capability.allowed_reasoning_efforts : [];
    if (allowed.includes(effort)) return {visible: true, enabled: true, inactive: false, reason: ""};
    return {
      visible: configured,
      enabled: false,
      inactive: configured,
      reason: configured
        ? `This saved value is inactive at reasoning effort “${effort || "unset"}” and will not be sent.`
        : "",
    };
  }
  return {
    visible: configured,
    enabled: false,
    inactive: configured,
    reason: configured
      ? `${support === "never" ? "This parameter is not supported by the selected model." : "Current model-specific documentation does not establish support for this parameter."} The saved value is inactive and will not be sent.`
      : "",
  };
}

export function apiPathSelectable(metadata = {}, api, toolsRequired = false, effort = null, webSearch = false) {
  if (api === "auto") {
    const key = `${effort == null ? "null" : effort}:${Number(Boolean(toolsRequired))}:${Number(Boolean(webSearch))}`;
    return metadata?.auto_paths && Object.hasOwn(metadata.auto_paths, key)
      ? Boolean(metadata.auto_paths[key])
      : true;
  }
  if (!metadata?.api?.[api]) return false;
  const evaluated = metadata?.evaluations?.[api]?.[String(effort)];
  if (evaluated) return evaluated.reasoning && (!toolsRequired || evaluated.function) && (!webSearch || evaluated.web_search);
  // Older fixtures and cached catalogue responses retain the v4 projection.
  if (!toolsRequired) return true;
  const support = metadata?.function_calling?.[api];
  return typeof support === "boolean" ? support : support?.allowed_reasoning_efforts?.includes(effort) || false;
}

export function pickerModels(result = {}, selectedModel = "") {
  const selected = String(selectedModel || "");
  const models = Array.isArray(result.catalog_models) ? result.catalog_models : [];
  return models.filter((item) => item?.status === "current" || item?.id === selected);
}

export function modelDataStatusText(result = {}) {
  if (result.incompatible_catalog) {
    return "A newer model catalogue is available, but it requires a newer version of Extended OpenAI Conversation.";
  }
  const active = Number(result.catalog_version);
  const available = Number(result.available_catalog_version);
  const activeLabel = Number.isInteger(active) ? `v${active}` : "the current version";
  if (result.last_error) return result.last_error;
  if (result.update_available && Number.isInteger(available)) {
    return `Update available: ${activeLabel} → v${available}. The newer catalogue will not be used until you apply it.`;
  }
  return `Up to date — model data ${activeLabel}`;
}

export function modelDataControls(panel) {
  const data = panel?._modelCatalogData || {};
  const available = Boolean(data.update_available);
  const applyLabel = available && Number.isInteger(Number(data.available_catalog_version)) ? `Apply v${Number(data.available_catalog_version)} update` : "Apply update";
  const card = (action, title, copy, label, hidden = false) => `<div class="eoc-model-data-action" ${hidden ? "hidden" : ""}><strong>${title}</strong><p>${copy}</p><button type="button" class="secondary" data-model-data="${action}" ${hidden ? "hidden disabled" : ""}>${label}</button></div>`;
  const status = data.catalog_version !== undefined ? modelDataStatusText(data) : "Model data is shared by all agents. Background checks run daily; updates are applied only when you choose to apply them.";
  const escape = panel?._e || ((value) => String(value).replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]));
  return `<div class="eoc-model-data-panel" data-eoc-model-data-panel><div class="eoc-model-data-heading"><h3>Model capability data</h3><p>Model capability data helps Extended OpenAI choose the right settings and features for each model. Updates are checked daily, but changes are only applied when you approve them.</p></div><div class="eoc-model-data-actions">${card("check", "Check for updates", "Check the remote model catalogue now instead of waiting for the next daily background check.", "Check for updates")}${card("apply", "Apply available update", "Activate the newer catalogue found by the most recent check. This changes the shared capability data used by all agents.", applyLabel, !available)}${card("reset", "Restore bundled data", "Return to the known-good model capability data shipped with this installed integration. Future checks will not replace it automatically.", "Restore bundled data")}</div><div class="eoc-model-data-status"><strong>Model data status</strong><p class="help" data-model-data-status role="status">${escape(status)}</p></div></div>`;
}

function syncModelDataControls(panel, result = panel?._modelCatalogData || {}) {
  const root = panel?.shadowRoot;
  if (!root) return;
  const apply = root.querySelector('[data-model-data="apply"]');
  if (apply) {
    apply.hidden = !result.update_available;
    const card = apply.closest?.(".eoc-model-data-action");
    if (card) card.hidden = !result.update_available;
    apply.disabled = !result.update_available;
    if (result.update_available && Number.isInteger(Number(result.available_catalog_version))) {
      apply.textContent = `Apply v${Number(result.available_catalog_version)} update`;
    } else {
      apply.textContent = "Apply update";
    }
  }
  const status = root.querySelector("[data-model-data-status]");
  if (status && result.catalog_version !== undefined) status.textContent = modelDataStatusText(result);
}

export function bindModelDataControls(panel, onUpdated = () => {}) {
  const root = panel.shadowRoot;
  const buttons = root.querySelectorAll("[data-model-data]");
  buttons.forEach((button) => button.addEventListener("click", async () => {
    buttons.forEach((item) => { item.disabled = true; });
    const status = root.querySelector("[data-model-data-status]");
    try {
      const model = root.querySelector('[data-config="chat_model"]')?.value || "";
      const result = await lookupModelData(panel, model, button.dataset.modelData);
      if (status) status.textContent = modelDataStatusText(result);
      syncModelDataControls(panel, result);
      if (!result.last_error) onUpdated(status?.textContent || "Model data updated.");
    } catch (err) {
      if (!status) return;
      if (button.dataset.modelData === "check") {
        status.textContent = "Unable to check for model data updates. The existing model data is still in use. Check Home Assistant's internet connection and try again.";
      } else if (button.dataset.modelData === "apply") {
        status.textContent = `Unable to apply the model data update: ${err.message || String(err)}`;
      } else {
        status.textContent = `Unable to restore bundled model data: ${err.message || String(err)}`;
      }
    } finally {
      buttons.forEach((item) => {
        if (item.dataset.modelData === "apply") item.disabled = !panel?._modelCatalogData?.update_available;
        else item.disabled = false;
      });
    }
  }));
}
