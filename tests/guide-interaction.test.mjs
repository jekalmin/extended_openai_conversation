import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";

const guideSource = await readFile(
  new URL("../custom_components/extended_openai_conversation_responses/frontend/guide-page-impl.js", import.meta.url),
  "utf8",
);

assert.match(guideSource, /const GUIDE_TOPIC_SEARCH = new Map/);
assert.match(guideSource, /GUIDE_TOPIC_SEARCH\.get\(topic\.id\)\?\.includes\(query\)/);
assert.match(guideSource, /applyGuideSearch\(panel, panel\._guideQuery\)/);
assert.doesNotMatch(
  guideSource.match(/export function bindGuide\(panel\) \{[\s\S]*$/)?.[0] || "",
  /panel\._render\(\)/,
  "Guide typing must not rerender the route",
);
assert.doesNotMatch(guideSource, /querySelectorAll\("\.guide-topic\[open\]"\)/);
assert.match(guideSource, /panel\._openGuideTopicElement/);
assert.doesNotMatch(guideSource, /return `<style>/);

const guideContentSource = await readFile(
  new URL("../custom_components/extended_openai_conversation_responses/frontend/guide-content.js", import.meta.url),
  "utf8",
);
const requestRulesStart = guideContentSource.indexOf('id: "request-rules"');
const requestRulesEnd = guideContentSource.indexOf("\n  {\n    id:", requestRulesStart + 1);
const requestRulesGuide = guideContentSource.slice(requestRulesStart, requestRulesEnd);
assert.match(requestRulesGuide, /ExtendedOpenAI sentence patterns/);
assert.match(requestRulesGuide, /Continue matching after this rule/);
assert.match(requestRulesGuide, /Captured value/);
assert.match(requestRulesGuide, /Rule Sharing/);
assert.match(requestRulesGuide, /Set conversation response overrides the generic success response/);
assert.match(requestRulesGuide, /Continue to AI runs only after normal completion/);
assert.match(requestRulesGuide, /response_variable stay within the action sequence/);
assert.match(requestRulesGuide, /Newly created rules start enabled/);
assert.match(requestRulesGuide, /groups only help organise and filter rules/i);
assert.doesNotMatch(requestRulesGuide, /Home Assistant sentence patterns|Hassil sentence format/);
