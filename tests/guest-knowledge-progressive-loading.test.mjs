import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {renderKnowledge} from "../custom_components/extended_openai_conversation_responses/frontend/management-knowledge-feature.js";

const frontend = (name) => new URL(
  `../custom_components/extended_openai_conversation_responses/frontend/${name}`,
  import.meta.url,
);

const routeSource = await readFile(frontend("management-route.js"), "utf8");
const knowledgeSource = await readFile(frontend("management-knowledge-feature.js"), "utf8");
const panelSource = await readFile(frontend("management-panel.js"), "utf8");
const backendSource = await readFile(
  new URL("../custom_components/extended_openai_conversation_responses/management_ui.py", import.meta.url),
  "utf8",
);

assert.doesNotMatch(
  routeSource,
  /\["capabilities\/home-assistant", "capabilities\/web-skills", "data-memory\/knowledge"\]/,
  "Knowledge must not wait for the broad Capabilities feature",
);
assert.doesNotMatch(
  routeSource,
  /\["data-memory\/memories", "data-memory\/knowledge", "usage-maintenance\/diagnostics"\]/,
  "Knowledge must not wait for the diagnostics/status feature",
);
assert.match(knowledgeSource, /management-feature-status-core\.js/);
assert.match(knowledgeSource, /knowledge-presentation\.js/);
assert.doesNotMatch(knowledgeSource, /management-capabilities-ia\.js|management-feature-status\.js/);
assert.match(panelSource, /\$\{this\._knowledge\(\)\}/);
const knowledgeHtml = renderKnowledge({_e: String, _query: "", _empty: String,
  _data: {is_admin: true}, _result: {sources: []},
  _selectedAgent: () => ({knowledge_enabled: true})});
assert.match(knowledgeHtml, /data-knowledge-collection/);
assert.equal((knowledgeHtml.match(/id="knowledge-status"/g) || []).length, 1);
assert.equal((knowledgeHtml.match(/Add a source to make Knowledge available/g) || []).length, 1);
assert.match(knowledgeSource, /\$\{knowledgeAvailabilityMarkup\(panel\)\}/);

assert.match(backendSource, /if action == "details":/);
assert.match(panelSource, /_call\("guest_mode", "details"\)/);
assert.match(panelSource, /loading: \{\.\.\.\(result\.loading \|\| \{\}\), details: true\}/);
assert.match(panelSource, /Guest Mode capabilities/);
