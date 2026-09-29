import assert from "node:assert/strict";

import {
  commonCapturedSlotNames,
  renameResultReferences,
  suggestResultAlias,
} from "../custom_components/extended_openai_conversation_responses/frontend/request-rules-ui-impl.js";

assert.deepEqual(
  commonCapturedSlotNames("ask {question}\nanswer {question}"),
  ["question"],
);
assert.deepEqual(
  commonCapturedSlotNames("ask {question} about {topic}\nanswer {question}"),
  ["question"],
);
assert.deepEqual(
  commonCapturedSlotNames("ask {question}\nanswer {topic}"),
  [],
);
assert.deepEqual(commonCapturedSlotNames(""), []);

assert.equal(suggestResultAlias("get_battery"), "get_battery");
assert.equal(suggestResultAlias("get-battery"), "get_battery");
assert.equal(suggestResultAlias("123 battery"), "battery");
assert.equal(suggestResultAlias("get_battery", ["get_battery"]), "get_battery_2");
assert.equal(suggestResultAlias("request"), "request_2");

const nested = [
  {
    action:"extended_openai_conversation_responses.call_function",
    data:{function:"battery", result_alias:"battery", step_id:"stable"},
  },
  {
    choose:[{
      conditions:[],
      sequence:[{
        action:"notify.test",
        data:{message:"Battery {battery.level} / {other.level}"},
      }],
    }],
  },
];
assert.deepEqual(renameResultReferences(nested, "battery", "power"), [
  {
    action:"extended_openai_conversation_responses.call_function",
    data:{function:"battery", result_alias:"battery", step_id:"stable"},
  },
  {
    choose:[{
      conditions:[],
      sequence:[{
        action:"notify.test",
        data:{message:"Battery {power.level} / {other.level}"},
      }],
    }],
  },
]);
assert.equal(
  renameResultReferences("Battery {battery.level}", "battery", "power"),
  "Battery {power.level}",
);
assert.equal(
  renameResultReferences("Template {{ battery.level }}", "battery", "power"),
  "Template {{ battery.level }}",
);

console.log("Request Rules capture and result-alias frontend contracts passed");
