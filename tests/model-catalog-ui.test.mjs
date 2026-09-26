import assert from "node:assert/strict";
import {bindModelDataControls, lookupModelData, modelDataControls, modelDataStatusText} from "../custom_components/extended_openai_conversation_responses/frontend/model-catalog.js";
import {syncRequestRuleRoutingControls} from "../custom_components/extended_openai_conversation_responses/frontend/request-rules-ui.js";

const calls = [];
let result = {
  source:"bundled",
  catalog_version:2,
  available_catalog_version:null,
  update_available:false,
  last_error:null,
  model_capabilities:{reasoning_effort_options:["minimal","high"]},
};
let failure = null;
const buttons = ["check","apply","reset"].map((action) => ({
  dataset:{modelData:action},
  disabled:action === "apply",
  hidden:action === "apply",
  textContent:action,
  addEventListener(_event, handler) { this.click = handler; },
}));
const status = {textContent:""};
const modelInput = {value:"gpt-future"};
const panel = {
  _result:{},
  _hass:{async callWS(message) { calls.push(message); if (failure) throw failure; return result; }},
  shadowRoot:{
    querySelectorAll:() => buttons,
    querySelector:(selector) => {
      if (selector === "[data-model-data-status]") return status;
      if (selector === '[data-config="chat_model"]') return modelInput;
      const match = selector.match(/^\[data-model-data="(.+)"\]$/);
      return match ? buttons.find((button) => button.dataset.modelData === match[1]) : null;
    },
  },
};

const controls = modelDataControls();
assert.match(controls, /Check for updates/);
assert.match(controls, /Apply update/);
assert.match(controls, /Restore bundled data/);
assert.equal(modelDataStatusText(result), "Up to date — model data v2");
assert.equal(
  modelDataStatusText({...result, incompatible_catalog:{schema_version:99,catalog_version:3}}),
  "A newer model catalogue is available, but it requires a newer version of Extended OpenAI Conversation.",
);

let refreshes = 0;
bindModelDataControls(panel, () => { refreshes++; });
const [checkButton, applyButton, resetButton] = buttons;
assert.equal(applyButton.hidden, true);
assert.equal(applyButton.disabled, true);

// A manual check stages newer data without presenting it as active.
result = {
  ...result,
  available_catalog_version:3,
  update_available:true,
};
await checkButton.click();
assert.deepEqual(calls[0], {
  type:"extended_openai_conversation_responses/model_catalog",
  action:"check",
  model:"gpt-future",
});
assert.equal(refreshes, 1);
assert.equal(applyButton.hidden, false);
assert.equal(applyButton.disabled, false);
assert.equal(applyButton.textContent, "Apply v3 update");
assert.match(status.textContent, /Update available: v2 → v3/);
assert.match(status.textContent, /will not be used until you apply it/);
assert.deepEqual(panel._result.model_capabilities.reasoning_effort_options, ["minimal","high"]);

// Applying switches the status to the downloaded catalogue and clears pending UI.
result = {
  ...result,
  source:"downloaded",
  catalog_version:3,
  available_catalog_version:null,
  update_available:false,
};
await applyButton.click();
assert.equal(calls[1].action, "apply");
assert.equal(refreshes, 2);
assert.equal(applyButton.hidden, true);
assert.equal(applyButton.disabled, true);
assert.equal(status.textContent, "Up to date — model data v3");

// Restoring bundled data remains a distinct explicit action.
result = {
  ...result,
  source:"bundled",
  catalog_version:2,
  available_catalog_version:3,
  update_available:true,
};
await resetButton.click();
assert.equal(calls[2].action, "reset");
assert.equal(refreshes, 3);
assert.equal(applyButton.hidden, false);
assert.equal(applyButton.disabled, false);
assert.match(status.textContent, /Update available: v2 → v3/);

result = {...result, last_error:"Previous catalogue retained"};
await checkButton.click();
assert.equal(refreshes, 3);
assert.equal(status.textContent, result.last_error);
assert.equal(checkButton.disabled, false);
assert.equal(resetButton.disabled, false);
assert.equal(applyButton.disabled, false);

failure = new Error("model_catalog_check_failed");
await checkButton.click();
assert.equal(refreshes, 3, "a failed manual check must not be reported as an update");
assert.equal(
  status.textContent,
  "Unable to check for model data updates. The existing model data is still in use. Check Home Assistant's internet connection and try again.",
);
assert.equal(checkButton.disabled, false);
assert.equal(resetButton.disabled, false);
assert.equal(applyButton.disabled, false);
failure = null;

await lookupModelData(panel, "another-model");
assert.equal(calls.at(-1).model, "another-model");
assert.equal(calls.at(-1).action, "lookup");

// Future model choices supplied by Python must work without another JS release.
const select = {
  value:"minimal",
  options:[{value:""},{value:"high"}],
  ownerDocument:{createElement:() => ({value:"",disabled:false,textContent:""})},
  replaceChildren(...options) { this.options = options; },
  querySelector(selector) { return this.options.find((item) => selector === `option[value="${item.value}"]`) || null; },
};
const scopeOption = {value:"request",disabled:false};
const scope = {value:"request", disabled:false, querySelector:() => scopeOption};
const root = {querySelector:(selector) => ({"#rule-action-type":{value:"model_routing"},"#rule-match":{value:"contains"},"#rule-scope":scope,"#rule-reasoning":select})[selector]};
syncRequestRuleRoutingControls(root, ["minimal"]);
assert.equal(select.value, "minimal");
assert.deepEqual(select.options.map((item) => item.value), ["", "minimal"]);
assert.equal(select.querySelector('option[value="high"]'), null);
