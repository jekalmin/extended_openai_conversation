import {bindConfigurationInputs} from "./configuration-inputs.js";
import {applyTargetedConfigDirty} from "./management-state-safety.js";
import {bindConfiguration as bindBaseConfiguration} from "./agent-config-editor-base.js";
import {lookupModelData} from "./model-catalog.js";

function currentConfig(panel) {
  return panel?._draft || panel?._result?.config || {};
}

function currentCatalogData(panel, model) {
  const data = panel?._modelCatalogData;
  return data?.requested_model === String(model || "") ? data : null;
}

async function ensureCatalogData(panel) {
  const config = currentConfig(panel);
  const model = String(config.chat_model || "");
  if (currentCatalogData(panel, model)) return;
  const searchTarget = panel._pendingSettingFocus;
  const agent = panel._agentId;
  const current = () => panel._agentId === agent && String(currentConfig(panel).chat_model || "") === model;
  try {
    await lookupModelData(panel, model, "lookup", current);
    if (!current()) return;
    // A late catalog response must not replace an editor while the user is
    // typing. The next ordinary render will use the now-cached model data.
    if (panel.shadowRoot?.activeElement?.matches?.("input,textarea,[contenteditable='true']")) return;
    if (searchTarget) panel._configRestoreFocus = `#${CSS.escape(searchTarget)}`;
    panel._render();
  } catch (err) {
    panel._toast?.(`Unable to inspect model capabilities: ${err.message || String(err)}`, true);
  }
}

function applyModelDefaults(panel, data) {
  const reasoning = data?.model_metadata?.reasoning || {};
  const efforts = Array.isArray(reasoning.efforts) ? reasoning.efforts : [];
  if (!reasoning.supported || !efforts.length) {
    delete panel._draft.reasoning_effort;
    return;
  }
  const recommended = data?.model_metadata?.recommended_profile?.reasoning_effort;
  panel._draft.reasoning_effort = efforts.includes(recommended) ? recommended : efforts[0];
}

export function bindConfiguration(panel) {
  const result = bindBaseConfiguration(panel);
  const root = panel?.shadowRoot;
  const modelInput = root?.querySelector('[data-config="chat_model"]');
  const reasoning = root?.querySelector('[data-config="reasoning_effort"]');
  const hasModelAwareControls = Boolean(
    modelInput
    || reasoning
    || root?.querySelector('[data-config="temperature"],[data-config="top_p"],[data-config="api_mode"],[data-config="max_tokens"]')
  );
  if (hasModelAwareControls) void ensureCatalogData(panel);

  bindConfigurationInputs(panel, {
    modelChanged: (control) => changeConfigurationModel(panel, control),
    reasoningChanged: () => {
      panel._configRestoreFocus = '[data-config="reasoning_effort"]';
      panel._render();
    },
    capabilityChanged: (control) => {
      panel._configRestoreFocus = `[data-config="${control.dataset.config}"]`;
      panel._render();
    },
  });
  return result;
}

export async function changeConfigurationModel(panel, control) {
  const model = control.value;
  const agent = panel._agentId;
  const draft = panel._draft;
  const token = (panel._eocModelChangeToken || 0) + 1;
  panel._eocModelChangeToken = token;
  const current = () => panel._agentId === agent && panel._draft === draft
    && panel._eocModelChangeToken === token && draft.chat_model === model;
  try {
    const data = await lookupModelData(panel, model, "lookup", current);
    if (!current()) return;
    applyModelDefaults(panel, data);
    applyTargetedConfigDirty(panel, ["chat_model", "reasoning_effort"], control, false);
    const validation = await panel._call("configuration", "validate", {config: panel._draft});
    if (!current() || !validation.valid) return;
    panel._result.model_capabilities = validation.model_capabilities;
    panel._configRestoreFocus = '[data-config="chat_model"]';
    panel._render();
  } catch (err) {
    if (current()) panel._toast?.(`Unable to inspect model options: ${err.message || String(err)}`, true);
  }
}
