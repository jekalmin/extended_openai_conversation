import assert from "node:assert/strict";

import {
  apiPathSelectable,
  parameterControlState,
  pickerModels,
} from "../custom_components/extended_openai_conversation_responses/frontend/model-catalog.js";
import {modelFieldPresentation, webSearchControlState} from "../custom_components/extended_openai_conversation_responses/frontend/agent-config-model-presentation.js";

{
  const conditional = {
    support: "conditional",
    allowed_reasoning_efforts: ["none"],
  };
  assert.deepEqual(parameterControlState(conditional, "none", 0.2), {
    visible: true,
    enabled: true,
    inactive: false,
    reason: "",
  });
  const inactive = parameterControlState(conditional, "high", 0.2);
  assert.equal(inactive.visible, true);
  assert.equal(inactive.enabled, false);
  assert.equal(inactive.inactive, true);
  assert.match(inactive.reason, /will not be sent/i);
  assert.deepEqual(parameterControlState(conditional, "high", ""), {
    visible: false,
    enabled: false,
    inactive: false,
    reason: "",
  });
}

{
  assert.equal(parameterControlState({support: "always"}, null, null).enabled, true);
  assert.equal(parameterControlState({support: "never"}, "minimal", null).visible, false);
  const staleNever = parameterControlState({support: "never"}, "minimal", 0.7);
  assert.equal(staleNever.visible, true);
  assert.equal(staleNever.enabled, false);
  assert.match(staleNever.reason, /not supported/i);
  const undocumented = parameterControlState({support: "undocumented"}, "low", 0.7);
  assert.equal(undocumented.enabled, false);
  assert.match(undocumented.reason, /documentation/i);
}

{
  const astra = {
    api: {responses: true, chat_completions: true},
    function_calling: {responses: true, chat_completions: false},
  };
  assert.equal(apiPathSelectable(astra, "chat_completions", false), true);
  assert.equal(apiPathSelectable(astra, "chat_completions", true), false);
  assert.equal(apiPathSelectable(astra, "responses", true), true);
  assert.equal(apiPathSelectable(astra, "auto", true), true);
}

{
  const selectedModelMetadata = {
    api: {responses: true, chat_completions: true},
    function_calling: {
      responses: true,
      chat_completions: {support: "conditional", allowed_reasoning_efforts: ["none"]},
    },
  };
  assert.equal(apiPathSelectable(selectedModelMetadata, "chat_completions", true, "none"), true);
  assert.equal(apiPathSelectable(selectedModelMetadata, "chat_completions", true, "high"), false);
  assert.equal(apiPathSelectable(selectedModelMetadata, "chat_completions", false, "high"), true);
  // Reasoning changes project the metadata already held by the Configuration view.
  assert.equal(apiPathSelectable(selectedModelMetadata, "responses", true, "high"), true);
}

{
  const capabilities = {
    api:{responses:true,chat_completions:true},
    reasoning:{supported:true,efforts:["none","high","max"],by_api:{responses:{efforts:["none","high","max"]},chat_completions:{efforts:["none","high"]}}},
    recommended_profile:{reasoning_effort:"high"},
    evaluations:{responses:{none:{reasoning:true,function:true,web_search:true},high:{reasoning:true,function:true,web_search:true},max:{reasoning:true,function:true,web_search:true}},chat_completions:{none:{reasoning:true,function:true,web_search:false},high:{reasoning:true,function:false,web_search:false},max:{reasoning:false,function:false,web_search:false}}},
    auto_paths:{"high:0:0":"chat_completions","high:1:0":"responses","max:0:0":"responses","max:1:1":null},
  };
  const panel = {_draft:{chat_model:"gpt-5.6",api_mode:"chat_completions",reasoning_effort:"high"},_result:{model_capabilities:capabilities,options:{api_mode:[{value:"auto"},{value:"responses"},{value:"chat_completions"}]}}};
  assert.equal(apiPathSelectable(capabilities,"chat_completions",false,"max"),false);
  assert.equal(apiPathSelectable(capabilities,"chat_completions",true,"high"),false);
  assert.equal(apiPathSelectable(capabilities,"chat_completions",true,"none"),true);
  assert.equal(apiPathSelectable(capabilities,"auto",true,"max",true),false);
  assert.deepEqual(modelFieldPresentation(panel,"reasoning_effort","high").options.map(({value})=>value),["none","high"]);
  panel._draft.api_mode="responses";
  assert.deepEqual(modelFieldPresentation(panel,"reasoning_effort","high").options.map(({value})=>value),["none","high","max"]);
  panel._draft.api_mode="auto";
  panel._draft.reasoning_effort="max";
  assert.deepEqual(modelFieldPresentation(panel,"reasoning_effort","max").options.map(({value})=>value),["none","high","max"]);
  panel._draft.chat_model="gpt-5-mini";
  panel._draft.api_mode="responses";
  panel._draft.reasoning_effort="minimal";
  panel._draft.web_search=true;
  capabilities.evaluations.responses.minimal={reasoning:true,function:true,web_search:false};
  assert.equal(webSearchControlState(panel).disabled,true);
  assert.match(webSearchControlState(panel).note,/saved setting is retained/);
  panel._draft.reasoning_effort="high";
  assert.equal(webSearchControlState(panel).disabled,false);
  panel._draft.reasoning_effort=null;
  panel._draft.api_mode="auto";
  capabilities.recommended_profile.reasoning_effort=null;
  capabilities.auto_paths["null:0:1"]="responses";
  capabilities.evaluations.responses.null={reasoning:true,function:true,web_search:true};
  assert.equal(webSearchControlState(panel).disabled,false);
}

{
  const result = {
    catalog_models: [
      {id: "gpt-6-astra", status: "current"},
      {id: "gpt-4-legacy", status: "deprecated"},
      {id: "gpt-5.3", status: "unknown"},
    ],
  };
  assert.deepEqual(pickerModels(result, "").map((item) => item.id), ["gpt-6-astra"]);
  assert.deepEqual(
    pickerModels(result, "gpt-4-legacy").map((item) => item.id),
    ["gpt-6-astra", "gpt-4-legacy"],
  );
  assert.equal(pickerModels(result, "").some((item) => item.id === "gpt-5.3"), false);
}
