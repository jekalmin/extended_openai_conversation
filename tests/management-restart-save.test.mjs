import assert from "node:assert/strict";
import {saveConfigurationAcrossRestart} from "../custom_components/extended_openai_conversation_responses/frontend/management-actions.js";

const conflict = new Error("Configuration changed in another tab. Reload the latest saved settings before saving.");
const initial = {revision: "chain", server_epoch: "old", title: "Assistant", config: {temperature: 0.5}};
const latest = {revision: "content", server_epoch: "new", title: "Assistant", config: {temperature: 0.5}};
const draft = {temperature: 0.7};
const payload = {config: draft, revision: initial.revision};

function scenario(next) {
  const calls = [];
  const panel = {
    _agentId: "agent-1", _configData: initial, _draft: draft, _draftTitle: "Assistant",
    async _call(section, action, data) {
      calls.push([section, action, data]);
      if (action === "get") return next;
      if (calls.filter((call) => call[1] === "save").length === 1) throw conflict;
      return {valid: true};
    },
  };
  return {panel, calls};
}

{
  const {panel, calls} = scenario(latest);
  assert.deepEqual(await saveConfigurationAcrossRestart(panel, payload, draft, "Assistant"), {valid: true});
  assert.deepEqual(calls.map((call) => call[1]), ["save", "get", "save"]);
  assert.equal(calls[2][2].revision, latest.revision);
}

for (const changed of [
  {...latest, server_epoch: "old"},
  {...latest, config: {temperature: 0.6}},
  {...latest, title: "Other assistant"},
]) {
  const {panel, calls} = scenario(changed);
  await assert.rejects(saveConfigurationAcrossRestart(panel, payload, draft, "Assistant"), conflict);
  assert.deepEqual(calls.map((call) => call[1]), ["save", "get"]);
}

{
  const {panel, calls} = scenario(latest);
  panel._draft = {temperature: 0.8};
  await assert.rejects(saveConfigurationAcrossRestart(panel, payload, draft, "Assistant"), conflict);
  assert.deepEqual(calls.map((call) => call[1]), ["save", "get"]);
}
