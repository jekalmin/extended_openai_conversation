import assert from "node:assert/strict";

import {knowledgeAvailabilityMarkup} from "../custom_components/extended_openai_conversation_responses/frontend/management-knowledge-feature.js";
import {
  diagnosticResultMarkup,
  diagnosticStatusMeta,
  featureStatusMarkup,
  overviewAgentFeatureProjection,
} from "../custom_components/extended_openai_conversation_responses/frontend/management-feature-status.js";

const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (character) => ({
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
})[character]);

const panel = {
  _data: {is_admin: true},
  _e: escape,
};

const markup = featureStatusMarkup(panel, "Persistent memory", {
  state: "enabled",
  label: "Manual",
  detail: "Automatic inclusion is off; memory search tools remain available.",
}, {
  page: "data-memory",
  subsection: "memory-settings",
  label: "Configure memory",
});
assert.match(markup, /Persistent memory/);
assert.match(markup, /Manual/);
assert.match(markup, /memory search tools remain available/);
assert.match(markup, /data-page="data-memory"/);
assert.match(markup, /data-subsection="memory-settings"/);
assert.match(markup, /status-value on/);

const disabledMarkup = featureStatusMarkup(panel, "Knowledge Library", {
  state: "empty",
  label: "Needs sources",
  detail: "Knowledge tools remain unavailable until at least one source exists.",
});
assert.match(disabledMarkup, /Needs sources/);
assert.doesNotMatch(disabledMarkup, /status-value on/);
assert.doesNotMatch(disabledMarkup, /inline-route/);

const knowledgePanel = {
  _e: escape,
  _data: {is_admin: true},
  _result: {feature_status: {state: "empty", enabled: true, source_count: 0}},
  _selectedAgent: () => ({feature_status: {knowledge: {state: "disabled", enabled: false}}}),
};
assert.match(knowledgeAvailabilityMarkup(knowledgePanel), /knowledge-enabled-toggle[^>]*checked/);
knowledgePanel._result.feature_status = {state: "disabled", enabled: false, source_count: 2};
assert.doesNotMatch(knowledgeAvailabilityMarkup(knowledgePanel), /knowledge-enabled-toggle[^>]*checked/);

const projected = overviewAgentFeatureProjection({
  memory_mode: "automatic",
  feature_status: {
    memory: {label: "Automatic"},
    knowledge: {label: "Available"},
  },
});
assert.equal(projected.memory_mode, "Automatic · Knowledge Available");

const original = {memory_mode: "manual"};
assert.equal(overviewAgentFeatureProjection(original), original);

assert.deepEqual(diagnosticStatusMeta("Passed"), {label: "Passed", icon: "✓", className: "passed"});
assert.deepEqual(diagnosticStatusMeta("Warning"), {label: "Warning", icon: "!", className: "warning"});
assert.deepEqual(diagnosticStatusMeta("Failed"), {label: "Failed", icon: "×", className: "failed"});

const diagnostics = diagnosticResultMarkup(panel, {
  status: "Warning",
  authentication_rejected: false,
  checks: [
    {name: "Authentication", status: "Passed", message: "API client is available"},
    {name: "Exposed entities", status: "Warning", message: "0"},
    {name: "Configuration", status: "Failed", message: "Unsafe <script>alert('x')</script>"},
  ],
});
assert.match(diagnostics, /diagnostic-summary warning/);
assert.match(diagnostics, /Authentication/);
assert.match(diagnostics, /API client is available/);
assert.match(diagnostics, /diagnostic-check warning/);
assert.match(diagnostics, /diagnostic-check failed/);
assert.match(diagnostics, /Show raw diagnostic data/);
assert.match(diagnostics, /&lt;script&gt;/);
assert.doesNotMatch(diagnostics, /<script>/);
assert.match(diagnostics, /authentication_rejected/);
