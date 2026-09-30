import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";

import {renderRequestRules} from "../custom_components/extended_openai_conversation_responses/frontend/request-rules-ui.js";
import {createRequestRuleActionSelector, createRequestRuleConditionSelector, loadRequestRuleActions, readRequestRuleActions, renameResultReferences, suggestResultAlias, requestRulesDialog} from "../custom_components/extended_openai_conversation_responses/frontend/request-rules-ui-impl.js";

const escape = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
const panel = {
  _e: escape,
  _query: "",
  _result: {
    defaults: {word_forms:true,wording_alternatives:true,fuzzy:false,fuzzy_threshold:90},
    wording_groups: [{canonical:"turn on",alternatives:["switch on"]}],
    rules: [{
      id:"one",name:"Good night",enabled:true,phrases:["good night","bed time"],match_type:"equals",
      action_type:"local_action",action:{actions:[{domain:"script",service:"turn_on",target:{entity_id:["script.goodnight"]},data:{}}],success_response:"Done"},
      matching_behavior:"defaults",matching:{word_forms:true,wording_alternatives:true,fuzzy:false,fuzzy_threshold:90},order:0,
    }, {
      id:"two",name:"Think carefully",enabled:true,phrases:["think carefully"],match_type:"starts_with",
      action_type:"model_routing",action:{model:"gpt-5",reasoning_effort:"high",scope:"conversation",reset:false,success_response:"Updated"},
      matching_behavior:"defaults",matching:{word_forms:true,wording_alternatives:true,fuzzy:false,fuzzy_threshold:90},order:1,
    }],
  },
};

const html = renderRequestRules(panel);
assert.match(html, /<details id="rule-sharing"/);
assert.ok(html.indexOf('id="rule-sharing"') > html.indexOf('class="content-card rule-test-tools"'));
assert.doesNotMatch(html.slice(0,html.indexOf('id="rule-sharing"')), /rule-pack-export/);
assert.match(html, /local commands that skip the AI call/);
assert.doesNotMatch(html, /class="notice on"/);
assert.match(html, /class="guide-topic-link guide-link" data-guide-topic="request-rules">Learn how routing works/);
assert.doesNotMatch(html, /class="rule-routing-help"/);
assert.ok(html.indexOf('class="rule-list"') < html.indexOf('class="content-card rule-settings"'));
assert.ok(html.indexOf('class="content-card rule-settings"') < html.indexOf('class="content-card rule-wording"'));
assert.ok(html.indexOf('class="content-card rule-wording"') < html.indexOf('class="content-card rule-test-tools"'));
assert.doesNotMatch(html, /request-only reset bypasses a conversation override/);
assert.match(html, /<h2 id="rule-test-title">Test rules<\/h2>/);
assert.match(html, /id="eoc-rule-live-test"/);
assert.match(html, /class="eoc-live-label">Live/);
assert.match(html, /class="rule-safe-label">Safe preview/);
assert.match(html, /Good night/);
assert.match(html, /#1 <span class="rule-group-chip">Ungrouped/);
assert.match(html, /id="rule-group-filter"/);
assert.match(html, /Evaluated in the order shown/);
assert.doesNotMatch(html, /rule-group-section/);
const grouped = {...panel, _ruleGroupFilter:"lighting", _result:{...panel._result, groups:[{id:"lighting",name:"Lighting"}], rules:[
  {...panel._result.rules[0], group_id:"lighting"},
  {...panel._result.rules[1], group_id:null},
]}};
const groupedHtml=renderRequestRules(grouped);
assert.match(groupedHtml, /#1 <span class="rule-group-chip">Lighting/);
assert.doesNotMatch(groupedHtml, /data-rule-key="two"/);
assert.match(groupedHtml, /Switch to All rules to change priority/);
const ungroupedHtml=renderRequestRules({...grouped,_ruleGroupFilter:"ungrouped"});
assert.match(ungroupedHtml, /#2 <span class="rule-group-chip">Ungrouped/);
assert.doesNotMatch(ungroupedHtml, /data-rule-key="one"/);
assert.match(renderRequestRules({...grouped,_ruleGroupFilter:"lighting",_query:"missing"}), /No rules match your search/);
assert.match(html, /Move to top/);
assert.match(html, /Move to bottom/);
assert.match(html, /<h2 id="rules-title">Rules \(2\)<\/h2>/);
assert.match(html, /Group: <select id="rule-group-filter" aria-label="Group filter"><option value="all" >All<\/option><option value="ungrouped" >Ungrouped<\/option>/);
assert.doesNotMatch(html, /<option[^>]*>\+ Create group<\/option>/);
assert.match(html, /id="rule-groups-manage">Manage groups/);
assert.match(html, /<dialog id="rule-groups-dialog"/);
assert.match(html, /Deleting a group moves its rules to Ungrouped\. Their priority and order stay the same/);
assert.match(html, /Default matching/);
assert.match(html, /Create rule/);
assert.match(html, /id="rule-search"/);
assert.match(html, /class="count" aria-live="polite" hidden>Showing 2 of 2 rules/);
assert.match(html, /Fuzzy matching/);
assert.match(html, /Wording alternatives/);
assert.match(html, /Used by rules that have not chosen custom matching\./);
assert.match(html, /Normal matches are always preferred before fuzzy matching is tried\./);
assert.match(html, /class="matching-setting"/);
assert.match(html, /class="matching-copy"/);
assert.match(html, /Treats simple variations such as “light” and “lights” as the same\./);
assert.match(html, /Uses your saved alternative phrases, such as “switch on” matching “turn on”\./);
assert.match(html, /Allows small speech-recognition or typing mistakes when no normal match succeeds\./);
assert.match(html, /Controls how close a phrase must be before fuzzy matching is accepted\. Conservative is least likely to match the wrong rule\./);
assert.match(html, /Main phrase/);
assert.match(html, /Other ways to say it/);
assert.doesNotMatch(html, /Save wording alternatives|Save defaults/);
const zeroHtml = renderRequestRules({...panel, _result: {...panel._result, rules: []}});
assert.match(zeroHtml, /Create your first Request Rule/);
assert.match(zeroHtml, /class="rule-toolbar" hidden/);
assert.match(zeroHtml, /id="rule-empty-add"/);
assert.match(zeroHtml, /id="rule-add">Create rule/);
assert.match(html, /Model: gpt-5 · High reasoning · rest of conversation/);
const resetHtml = renderRequestRules({...panel,_result:{...panel._result,rules:[{...panel._result.rules[1],action:{...panel._result.rules[1].action,reset:true}}]}});
assert.match(resetHtml, /Returns model and reasoning to the assistant's configured defaults for the active conversation/);
assert.doesNotMatch(resetHtml, /high reasoning/);
assert.match(requestRulesDialog(panel), /Alternatives must use the same variable names/);
assert.match(requestRulesDialog(panel), /Variable values let part of the request change each time/);
assert.match(requestRulesDialog(panel), /id="rule-action-sequence-host"/);
assert.match(requestRulesDialog(panel), /id="rule-condition-host"/);
assert.match(requestRulesDialog(panel), /id="rule-local-continue-to-ai"/);
assert.match(requestRulesDialog(panel), /id="rule-group"/);
assert.doesNotMatch(requestRulesDialog(panel), /rule-enabled-edit|<span>Enabled<\/span>/);
assert.match(requestRulesDialog(panel), /Set conversation response overrides it/);
assert.match(requestRulesDialog(panel), /unless an action sets a conversation response or stops the script/);
assert.match(requestRulesDialog(panel), /id="rule-continue-matching"/);
assert.doesNotMatch(requestRulesDialog(panel), /<ha-selector/);
assert.match(requestRulesDialog(panel), /Conditions, delays, choose, repeat, parallel/);
assert.match(requestRulesDialog(panel), /extended_openai_conversation_responses\.call_function/);
assert.match(requestRulesDialog(panel), /\{\{ item \}\}/);
const bindingSource = await readFile(new URL("../custom_components/extended_openai_conversation_responses/frontend/request-rules-ui-impl.js", import.meta.url), "utf8");
const literalHelpWords = "Home Assistant sentence pattern";
const literalHtml = renderRequestRules({...panel, _result: {...panel._result, rules: [
  {...panel._result.rules[0], name: literalHelpWords, phrases: [literalHelpWords]},
]}});
assert.match(literalHtml, /<h2>Home Assistant sentence pattern<\/h2>/);
assert.match(literalHtml, /<span>Home Assistant sentence pattern<\/span>/);
assert.match(literalHtml, /1 trigger phrase · Equals · Default matching/);
assert.equal((literalHtml.match(/Equals/g) || []).length, 1, "matching method is shown once, not repeated on each phrase");
assert.match(html, /class="rule-move-menu"><summary aria-label="Move Good night">Move<\/summary>/);
assert.match(bindingSource, /Captured values:/);
assert.match(bindingSource, /enabled:previous\?\.enabled\?\?true/);
assert.match(bindingSource, /selector = \{action:\{\}\}/);
assert.match(bindingSource, /selector = \{condition:\{\}\}/);
assert.doesNotMatch(requestRulesDialog(panel), /<textarea[^>]*id="rule-condition/);
assert.equal(renameResultReferences("{battery.level} {battery} {battery_other.level} {{ battery.level }}", "battery", "power"), "{power.level} {power} {battery_other.level} {{ battery.level }}");
assert.equal(renameResultReferences("{battery.items.0.name}", "battery", "power"), "{power.items.0.name}");
assert.equal(suggestResultAlias("get_battery", ["get_battery"]), "get_battery_2");
assert.equal(suggestResultAlias("request"), "request_2");

{
  const selector={addEventListener(){},};
  const host={ownerDocument:{createElement:()=>selector},replaceChildren(child){assert.equal(child,selector);}};
  const hass={};
  assert.equal(createRequestRuleConditionSelector({_hass:hass},host),selector);
  assert.deepEqual(selector.selector,{condition:{}});
  assert.equal(selector.hass,hass);
  assert.deepEqual(selector.value,[]);
}

{
  const hass = {localize: () => "localized"};
  const assignments = [];
  let connected = false;
  let valueChanged;
  const actionSelector = {
    set hass(value) { assignments.push(["hass", connected]); this._hass = value; },
    get hass() { return this._hass; },
    set selector(value) { assignments.push(["selector", connected]); this._selector = value; },
    get selector() { return this._selector; },
    set value(value) { assignments.push(["value", connected]); this._value = value; },
    get value() { return this._value; },
    addEventListener(type, listener) {
      assert.equal(type, "value-changed");
      assignments.push(["listener", connected]);
      valueChanged = listener;
    },
  };
  const host = {
    ownerDocument: {createElement: (tagName) => {
      assert.equal(tagName, "ha-selector");
      return actionSelector;
    }},
    replaceChildren(child) {
      assert.equal(child, actionSelector);
      connected = true;
    },
  };
  assert.equal(createRequestRuleActionSelector({_hass:hass}, host), actionSelector);
  assert.equal(actionSelector.hass, hass);
  assert.deepEqual(actionSelector.selector, {action:{}});
  assert.deepEqual(actionSelector.value, []);
  assert.deepEqual(assignments, [["hass",false],["selector",false],["value",false],["listener",false]]);

  loadRequestRuleActions(actionSelector, null);
  assert.deepEqual(actionSelector.value, []);

  const existingActions = [{action:"light.turn_on",data:{brightness_pct:50}}];
  loadRequestRuleActions(actionSelector, {action:{actions:existingActions}});
  assert.deepEqual(actionSelector.value, existingActions);
  assert.notEqual(actionSelector.value, existingActions);
  actionSelector.value[0].data.brightness_pct = 75;
  assert.equal(existingActions[0].data.brightness_pct, 50);
  const editedActions = [{delay:1}];
  valueChanged({detail:{value:editedActions}});
  assert.equal(readRequestRuleActions(actionSelector), editedActions);
}

assert.match(requestRulesDialog(panel), /Rest of this conversation/);
assert.match(requestRulesDialog(panel), /before optional AI continuation/);
assert.match(requestRulesDialog(panel), /ExtendedOpenAI sentence pattern/);
assert.doesNotMatch(requestRulesDialog(panel), /Home Assistant sentence pattern/);
const sentenceDialog = requestRulesDialog(panel);
assert.ok(sentenceDialog.indexOf('id="rule-match"') < sentenceDialog.indexOf('id="sentence-pattern-builder"'));
assert.doesNotMatch(sentenceDialog, /script\.turn_on/);
assert.match(requestRulesDialog(panel), /\{room=kitchen\|bedroom\}/);
assert.match(requestRulesDialog(panel), /\{level=0\.\.100\}/);
assert.match(requestRulesDialog(panel), /named expansions/i);
assert.match(requestRulesDialog(panel), /1\. What will you say\?/);
assert.match(requestRulesDialog(panel), /4\. What should the assistant say\?/);
assert.match(requestRulesDialog(panel), /class="matching-setting"/);
assert.match(requestRulesDialog(panel), /Treats simple variations such as “light” and “lights” as the same\./);

const diagnosticHtml = renderRequestRules({...panel,_result:{...panel._result,diagnostics:{one:"Sentence pattern is inactive: permutations are not supported"}}});
assert.match(diagnosticHtml, /Rule inactive/);
assert.match(diagnosticHtml, /permutations are not supported/);
