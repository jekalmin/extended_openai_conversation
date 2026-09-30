import {adoptKeyedElements, elementFromMarkup, keyedElement, placeChildren, pruneKeys} from "./keyed-collection.js";
import {requestRuleEmptyContent, updateRequestRuleEmptyState} from "./request-rule-empty-state.js";

export const fuzzyThresholdValue = (value) => {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed >= 70 && parsed <= 100 ? parsed : 90;
};
const matchLabel = (value) => ({equals:"Equals",starts_with:"Starts with",ends_with:"Ends with",contains:"Contains",sentence_pattern:"Sentence pattern"}[value] || value);
const titleCase = (value) => String(value || "").replaceAll("_"," ").replace(/\b\w/g, (letter) => letter.toUpperCase());

export function requestRuleSummary(rule = {}, defaults = {}) {
  const phrases = Array.isArray(rule.phrases) ? rule.phrases : [];
  const match = matchLabel(rule.match_type || "match");
  const phraseLabel = `${phrases.length} trigger phrase${phrases.length === 1 ? "" : "s"}`;
  let matching;
  if (rule.match_type === "sentence_pattern") {
    matching = `${phraseLabel} · ExtendedOpenAI sentence pattern`;
  } else {
    const source = rule.matching_behavior === "defaults" ? "Default matching" : "Custom matching";
    const settings = rule.matching_behavior === "defaults" ? defaults : (rule.matching || {});
    const threshold = Number(settings.fuzzy_threshold ?? 90);
    const fuzzyLabel = threshold >= 93 ? "conservative" : threshold >= 88 ? "normal" : "tolerant";
    const fuzzy = settings.fuzzy ? ` · Fuzzy fallback: ${titleCase(fuzzyLabel)}` : "";
    matching = `${phraseLabel} · ${match} · ${source}${fuzzy}`;
  }
  if (rule.action_type === "local_action") {
    const actions = Array.isArray(rule.action?.actions) ? rule.action.actions : [];
    const response = String(rule.action?.success_response || "").trim();
    return {
      action: `Runs ${actions.length} local step${actions.length === 1 ? "" : "s"}${rule.action?.continue_to_ai ? " then continues to AI" : ` without an AI request${response ? ` · replies “${response}”` : ""}`}`,
      matching,
      hiddenPhrases: Math.max(0, phrases.length - 4),
    };
  }
  if (rule.action?.reset) return {
    action: "Returns model and reasoning to the assistant's configured defaults for the active conversation.",
    matching,
    hiddenPhrases: Math.max(0, phrases.length - 4),
  };
  const model = rule.action?.model ? `Model: ${rule.action.model}` : "Keep current model";
  const reasoning = rule.action?.reasoning_effort ? `${titleCase(rule.action.reasoning_effort)} reasoning` : "Keep current reasoning";
  const scope = rule.action?.scope === "conversation" ? "rest of conversation" : "this request only";
  return {action: `${model} · ${reasoning} · ${scope}`, matching, hiddenPhrases: Math.max(0, phrases.length - 4)};
}

export async function recoverRequestRuleMutation(panel, error, label) {
  const message = error?.message || String(error || "Unknown error");
  panel._toast?.(`${label}: ${message}`, true);
  const cacheKey = panel._sectionCacheKey?.();
  if (cacheKey) panel._sectionCache?.delete(cacheKey);
  try { await panel._loadSection(true); }
  catch (refreshError) {
    panel._toast?.(`Unable to refresh Request Rules: ${refreshError?.message || String(refreshError || "Unknown error")}`, true);
  }
}

export const matchingControls = (prefix, values, hidden = false) => `<div id="${prefix}-matching-controls" class="matching-settings" ${hidden ? "hidden" : ""}><label class="matching-setting"><span class="matching-copy"><span class="matching-title">Normalize word forms</span><small>Treats simple variations such as “light” and “lights” as the same.</small></span><input id="${prefix}-word-forms" type="checkbox" ${values.word_forms ? "checked" : ""}></label><label class="matching-setting"><span class="matching-copy"><span class="matching-title">Wording alternatives</span><small>Uses your saved alternative phrases, such as “switch on” matching “turn on”.</small></span><input id="${prefix}-wording" type="checkbox" ${values.wording_alternatives ? "checked" : ""}></label><label class="matching-setting"><span class="matching-copy"><span class="matching-title">Fuzzy matching</span><small>Allows small speech-recognition or typing mistakes when no normal match succeeds.</small></span><input id="${prefix}-fuzzy" type="checkbox" ${values.fuzzy ? "checked" : ""}></label><label class="matching-setting fuzzy-sensitivity ${values.fuzzy ? "" : "is-disabled"}"><span class="matching-copy"><span class="matching-title">Fuzzy sensitivity</span><small>Controls how close a phrase must be before fuzzy matching is accepted. Conservative is least likely to match the wrong rule.</small></span><span class="matching-control"><input id="${prefix}-threshold" type="number" min="70" max="100" step="1" value="${fuzzyThresholdValue(values.fuzzy_threshold)}" ${values.fuzzy ? "" : "disabled"}></span></label></div>`;

const wordingEditor = (panel, groups) => `<details class="wording-editor eoc-details-base"><summary>Edit wording alternatives</summary><p class="help">Add different ways of saying the same thing. Separate multiple alternatives with commas.</p><div id="wording-groups">${groups.map((group) => `<div class="wording-group"><label>Main phrase<input class="wording-canonical" maxlength="100" value="${panel._e(group.canonical)}"></label><label>Other ways to say it<input class="wording-alternatives" value="${panel._e(group.alternatives.join(", "))}" placeholder="Comma-separated alternatives"></label><button type="button" class="icon wording-remove" aria-label="Remove wording alternative">×</button></div>`).join("")}</div><div class="section-actions"><button type="button" class="secondary" id="wording-add">Add wording alternative</button></div></details>`;
const groupManagerRows = (panel, groups, rules = []) => groups.map((group) => {
  const count = rules.filter((rule) => rule.group_id === group.id).length;
  return `<div class="rule-group-row" data-group-id="${panel._e(group.id)}"><label><span class="sr-only">Group name</span><input class="rule-group-name" value="${panel._e(group.name)}" aria-label="Group name"></label><span class="rule-group-count">${count} rule${count === 1 ? "" : "s"}</span><button type="button" class="secondary rule-group-rename" data-id="${panel._e(group.id)}">Rename</button><button type="button" class="danger secondary-danger rule-group-delete" data-id="${panel._e(group.id)}">Delete</button></div>`;
}).join("");
const groupManager = (panel, result) => `<dialog id="rule-groups-dialog" class="rule-groups" aria-labelledby="rule-groups-title"><div class="dialog-header"><h2 id="rule-groups-title">Manage groups</h2><button type="button" class="icon" id="rule-groups-close" aria-label="Close group manager">×</button></div><div class="dialog-body"><p class="help">Groups organise and filter rules without changing their global priority.</p><div class="rule-group-manager-rows">${groupManagerRows(panel, result.groups || [], result.rules || [])}</div><div class="rule-group-create"><label>New group name<input id="rule-new-group-name" maxlength="100" autocomplete="off" placeholder="e.g. Lighting"></label><button type="button" id="rule-group-add" class="secondary">Create group</button></div><p class="help">Deleting a group moves its rules to Ungrouped. Their priority and order stay the same.</p></div><div class="dialog-actions"><button type="button" class="secondary" id="rule-groups-done">Done</button></div></dialog>`;

function requestRuleCard(panel, rule, index) {
  const result = panel._result || {}, rules = result.rules || [], summary = requestRuleSummary(rule, result.defaults || {}), canReorder = !panel._query && (!panel._ruleGroupFilter || panel._ruleGroupFilter === "all");
  return `<article data-rule-key="${panel._e(rule.id)}" draggable="${canReorder}" class="request-rule-card ${rule.enabled ? "" : "disabled"}"><div class="rule-card-heading"><div><span class="type-badge ${rule.action_type === "local_action" ? "local" : "routing"}">${rule.action_type === "local_action" ? "Local command" : "AI routing"}</span><h2>${panel._e(rule.name)}</h2><span class="meta">#${index + 1} <span class="rule-group-chip">${panel._e((result.groups || []).find((group) => group.id === rule.group_id)?.name || "Ungrouped")}</span></span></div><label class="switch-label"><span class="sr-only">Enable ${panel._e(rule.name)}</span><input class="rule-enabled" data-id="${panel._e(rule.id)}" type="checkbox" ${rule.enabled ? "checked" : ""}></label></div><div class="phrase-chips">${rule.phrases.slice(0,4).map((phrase) => `<span>${panel._e(phrase)}</span>`).join("")}${summary.hiddenPhrases ? `<span class="eoc-more-phrases">+${summary.hiddenPhrases} more</span>` : ""}</div><p>${panel._e(summary.action)}</p><p class="meta">${panel._e(summary.matching)}</p>${rule.sensitive_matching_warning && rule.match_type !== "sentence_pattern" ? '<p class="sensitive-warning">Review tolerant matching carefully: this rule controls a potentially sensitive Home Assistant domain.</p>' : ""}<div class="actions">${result.diagnostics?.[rule.id] ? `<p class="sensitive-warning"><strong>Rule inactive:</strong> ${panel._e(result.diagnostics[rule.id])} Edit and save this rule to use the current sentence-pattern syntax.</p>` : ""}<details class="rule-move-menu"><summary aria-label="Move ${panel._e(rule.name)}">Move</summary><div class="rule-move-options"><button type="button" class="secondary rule-move" data-id="${panel._e(rule.id)}" data-direction="up" ${!canReorder || index === 0 ? "disabled" : ""}>Move up</button><button type="button" class="secondary rule-move" data-id="${panel._e(rule.id)}" data-direction="down" ${!canReorder || index === rules.length - 1 ? "disabled" : ""}>Move down</button><button type="button" class="secondary rule-move" data-id="${panel._e(rule.id)}" data-direction="top" ${!canReorder || index === 0 ? "disabled" : ""}>Move to top</button><button type="button" class="secondary rule-move" data-id="${panel._e(rule.id)}" data-direction="bottom" ${!canReorder || index === rules.length - 1 ? "disabled" : ""}>Move to bottom</button></div></details><button type="button" class="secondary rule-edit" data-id="${panel._e(rule.id)}">Edit</button><button type="button" class="secondary rule-duplicate" data-id="${panel._e(rule.id)}">Duplicate</button><button type="button" class="danger secondary-danger rule-delete" data-id="${panel._e(rule.id)}">Delete</button></div></article>`;
}
const EMPTY_RULES_MARKUP = '<section class="content-card empty-state"><h2>Create your first Request Rule</h2><p>Add a fast local command such as “good night”.</p><button type="button" id="rule-empty-add">Create rule</button></section>';
const SAFE_TESTER = '<div id="rule-match-tester" class="rule-preview-panel"><h3>Preview match <span class="rule-safe-label">Safe preview</span></h3><p class="help">See which enabled rules would match and where checking stops. No Home Assistant action executes, conversation routing changes, or AI provider call occurs.</p><div class="search-row"><input id="rule-match-test-text" type="text" maxlength="2048" placeholder="Turn off the kitchen light" aria-label="Request text to test against Request Rules"><button type="button" id="rule-match-test">Preview match</button></div><p class="help">Preview and live Request Rule matching inspect at most 2048 characters (and 256 words).</p><div id="rule-match-test-result" aria-live="polite"></div></div>';
const LIVE_TESTER = '<details id="eoc-rule-live-test" class="eoc-live-request-test"><summary><span>Run live request</span><span class="eoc-live-label">Live · real effects possible</span></summary><div class="eoc-live-request-body"><p>Runs text through the same full processing path as a real request to this assistant.</p><div class="notice"><strong>This can have real effects</strong><p>Unlike the safe preview above, this may execute Home Assistant actions, change conversation routing, or call the AI provider. A confirmation is shown before it runs.</p></div><div class="search-row"><input id="eoc-rule-live-text" type="text" placeholder="Turn off the kitchen light" aria-label="Live request text"><button type="button" id="eoc-rule-live-run">Run live request</button></div><pre id="eoc-rule-live-result" class="eoc-live-request-result" aria-live="polite"></pre></div></details>';

function globalRuleMarkup(panel, filtered) {
  return filtered.map(({rule,index}) => requestRuleCard(panel,rule,index)).join("");
}

function groupFilterOptions(panel, groups) {
  return [{id:"all",name:"All"},{id:"ungrouped",name:"Ungrouped"},...groups].map((group) => `<option value="${panel._e(group.id)}" ${panel._ruleGroupFilter === group.id ? "selected" : ""}>${panel._e(group.name)}</option>`).join("");
}

function ruleVisible(rule, groupFilter, search) {
  const groupMatches = groupFilter === "all" || !groupFilter || (groupFilter === "ungrouped" ? !rule.group_id : rule.group_id === groupFilter);
  return groupMatches && (!search || `${rule.name} ${rule.phrases.join(" ")} ${rule.action_type}`.toLowerCase().includes(search));
}

function renderRulesPage(panel, {query = panel._query || "", inPlaceSearch = false} = {}) {
  const result = panel._result || {}, rules = result.rules || [];
  const defaults = {...{word_forms:true,wording_alternatives:true,fuzzy:false,fuzzy_threshold:90}, ...((panel._rulesSettingsDraft || result).defaults || {})};
  const search = inPlaceSearch ? "" : String(query).trim().toLowerCase();
  const groupFilter = panel._ruleGroupFilter || "all";
  const scoped = groupFilter !== "all" || Boolean(String(query).trim());
  const filtered = rules.map((rule,index)=>({rule,index})).filter(({rule}) => ruleVisible(rule,groupFilter,search));
  return `<section class="page-intro rule-page-intro"><h1>Request Rules</h1><p>Create fast local commands that skip the AI call, or route AI requests by phrase before they reach the AI provider. <button type="button" class="guide-topic-link guide-link" data-guide-topic="request-rules">Learn how routing works</button></p></section>
    <section class="rule-collection" aria-labelledby="rules-title">
      <div class="rule-list-heading"><div><h2 id="rules-title">Rules (${rules.length})</h2><p class="help">Evaluated in the order shown. Drag to change priority.</p></div><div class="rule-heading-actions"><label class="rule-show-filter">Group: <select id="rule-group-filter" aria-label="Group filter">${groupFilterOptions(panel,result.groups || [])}</select></label><button type="button" class="secondary" id="rule-groups-manage">Manage groups</button><button type="button" id="rule-add">Create rule</button></div></div>
      <div class="rule-toolbar" ${rules.length ? "" : "hidden"}><input id="rule-search" type="search" value="${panel._e(query)}" placeholder="Search rules" aria-label="Search Request Rules"><span class="count" aria-live="polite" ${groupFilter === "all" && !query ? "hidden" : ""}>Showing ${filtered.length} of ${rules.length} rules</span></div>
      <p class="help rule-filter-help" ${groupFilter === "all" && !search ? "hidden" : ""}>Switch to All rules to change priority.</p>
      <section class="rule-list">${globalRuleMarkup(panel,filtered)} ${rules.length || scoped ? "" : EMPTY_RULES_MARKUP}<section class="content-card empty-state" data-eoc-rule-search-empty ${inPlaceSearch || filtered.length || !rules.length && !scoped ? "hidden" : ""}>${requestRuleEmptyContent(query, groupFilter)}</section></section>
    </section>
    <section class="content-card rule-settings" aria-labelledby="rule-matching-title"><div class="section-heading"><div><h2 id="rule-matching-title">Matching defaults</h2><p>Used by rules that have not chosen custom matching.</p></div></div><details class="eoc-details-base"><summary>Show default settings</summary><p class="help">Normal matches are always preferred before fuzzy matching is tried.</p>${matchingControls("rules-default",defaults)}</details></section>
    <section class="content-card rule-wording" aria-labelledby="rule-wording-title"><div class="section-heading"><div><h2 id="rule-wording-title">Wording alternatives</h2><p>Manage different ways of saying the same thing.</p></div></div>${wordingEditor(panel,(panel._rulesSettingsDraft || result).wording_groups || [])}</section>
    <section class="content-card rule-test-tools" aria-labelledby="rule-test-title"><div class="section-heading"><div><h2 id="rule-test-title">Test rules</h2><p>Preview a match safely before choosing to run a live request.</p></div></div>${SAFE_TESTER}${LIVE_TESTER}</section>
    <details id="rule-sharing" class="content-card eoc-details-base rule-sharing"><summary>Sharing</summary><p class="help">Import or export Request Rules for reuse or sharing.</p><div id="rule-sharing-content"></div></details>
    ${groupManager(panel,result)}`;
}
export function renderRequestRules(panel, presentation = {query:panel._query || "",inPlaceSearch:Boolean(panel._eocInPlaceRequestRuleSearch)}) { return renderRulesPage(panel,presentation); }

const ruleCollections = new WeakMap();
const pendingRuleButtons = new WeakSet();
const moveDisabled = (index,length,direction) => index < 0 || (["up", "top"].includes(direction) ? index === 0 : index === length - 1);
const ruleSettingsSignature = panel => JSON.stringify([(panel._rulesSettingsDraft || panel._result)?.defaults,(panel._rulesSettingsDraft || panel._result)?.wording_groups]);
function prepareRequestRulesCollection(panel) {
  const list = panel.shadowRoot.querySelector(".rule-list");
  if (!list || ruleCollections.has(list)) return;
  ruleCollections.set(list,{cards:adoptKeyedElements(list,"[data-rule-key]","ruleKey"),settings:ruleSettingsSignature(panel),empty:list.querySelector(":scope > .empty-state:not([data-eoc-rule-search-empty])")});
  reconcileRequestRules(panel);
}
export function reconcileRequestRules(panel) {
  const list=panel.shadowRoot.querySelector(".rule-list"), state=ruleCollections.get(list);
  if (!state || state.settings !== ruleSettingsSignature(panel)) return false;
  const result=panel._result || {}, rules=result.rules || [], filter=panel._ruleGroupFilter || "all", query=String(panel._query || "").trim().toLowerCase();
  const groupNames=new Map((result.groups || []).map((group)=>[group.id,group.name]));
  const canReorder=filter === "all" && !query;
  const nodes=rules.map((rule,index)=>{
    const {order:_order,...cardRule}=rule;
    const record=keyedElement(state.cards,rule.id,JSON.stringify([cardRule,result.defaults,result.diagnostics?.[rule.id]]),()=>requestRuleCard(panel,rule,index));
    const priority=record.node.querySelector(".rule-card-heading .meta");
    const groupName=groupNames.get(rule.group_id) || "Ungrouped";
    if(priority && priority.textContent !== `#${index + 1} ${groupName}`)priority.innerHTML=`#${index + 1} <span class="rule-group-chip">${panel._e(groupName)}</span>`;
    record.moves ||= [...record.node.querySelectorAll(".rule-move")];
    for(const button of record.moves) button.disabled = pendingRuleButtons.has(button) || !canReorder || moveDisabled(index,rules.length,button.dataset.direction);
    record.node.draggable=canReorder;
    record.node.hidden=!ruleVisible(rule,filter,query);
    return record.node;
  });
  if(!rules.length && filter === "all" && !query){state.empty ||= elementFromMarkup(EMPTY_RULES_MARKUP);nodes.push(state.empty);}
  const searchEmpty=list.querySelector("[data-eoc-rule-search-empty]");if(searchEmpty){updateRequestRuleEmptyState(searchEmpty,query,filter);searchEmpty.hidden=!rules.length && filter === "all" && !query || nodes.some(node=>node.matches?.("[data-rule-key]") && !node.hidden);nodes.push(searchEmpty);}
  placeChildren(list,nodes);pruneKeys(state.cards,new Set(rules.map(rule=>rule.id)));
  const toolbar=panel.shadowRoot.querySelector(".rule-toolbar");if(toolbar)toolbar.hidden=!rules.length;
  const title=panel.shadowRoot.querySelector("#rules-title");if(title)title.textContent=`Rules (${rules.length})`;
  const count=panel.shadowRoot.querySelector(".rule-toolbar .count"), visible=rules.filter(rule=>ruleVisible(rule,filter,query)).length;
  if(count){count.hidden=filter === "all" && !query;count.textContent=`Showing ${visible} of ${rules.length} rules`;}
  const help=panel.shadowRoot.querySelector(".rule-filter-help");if(help)help.hidden=canReorder;
  panel._eocRequestRuleCollectionRevision=(panel._eocRequestRuleCollectionRevision||0)+1;panel._eocRequestRuleSearchCache=null;return true;
}
export function requestRulesDialog(){ return ""; }

function syncScopeRevision(panel, result) {
  const scope=panel._unsavedState?.scopes?.get?.("capabilities/request-rules");
  if (scope && typeof result?.revision === "string") {
    scope.revision=result.revision;
    scope.result=panel._result;
  }
}
export function applyRulePackImportMutation(panel, result) {
  panel._result={...panel._result,
    rules:[...(panel._result?.rules||[]),...result.rules],
    groups:result.groups,
    revision:result.revision};
  syncScopeRevision(panel,panel._result);
  const root=panel.shadowRoot;
  const filter=root?.querySelector("#rule-group-filter");
  if(filter){filter.innerHTML=groupFilterOptions(panel,result.groups);filter.value=panel._ruleGroupFilter||"all";}
  const rows=root?.querySelector(".rule-group-manager-rows");
  if(rows)rows.innerHTML=groupManagerRows(panel,result.groups,panel._result.rules);
  const collection=ruleCollections.get(root?.querySelector(".rule-list"));
  if(collection)collection.settings=ruleSettingsSignature(panel);
  reconcileRequestRules(panel);
}
function finishMutation(panel,result,rules){
  panel._result={...(panel._result||{}),rules,revision:result.revision};
  const cacheKey=panel._sectionCacheKey?.(); if(cacheKey) panel._sectionCache?.delete(cacheKey);
  syncScopeRevision(panel,result);
  const reconciled = panel.shadowRoot ? reconcileRequestRules(panel) : false;
  if (!reconciled) panel._render();
}
export function applyRequestRuleMutation(panel, action, result, context={}) {
  if (!result || result.revision == null) return false;
  let rules=[...(panel._result?.rules || [])];
  const presentedRule = result.rule
    ? {
        ...result.rule,
        ...(Object.prototype.hasOwnProperty.call(result, "sensitive_matching_warning")
          ? {sensitive_matching_warning: Boolean(result.sensitive_matching_warning)}
          : {}),
      }
    : null;
  const ruleId=context.ruleId || presentedRule?.id;
  if (action === "delete") {
    rules=rules.filter(rule=>rule.id!==ruleId);
  } else if (action === "move") {
    const index=rules.findIndex(rule=>rule.id===ruleId);
    let target=({up:index-1,down:index+1,top:0,bottom:rules.length-1})[context.direction];
    if(context.direction==="before"||context.direction==="after"){
      target=rules.findIndex(rule=>rule.id===context.targetRuleId);
      if(index<target)target--;
      if(context.direction==="after")target++;
    }
    if(index>=0 && target>=0 && target<rules.length) rules.splice(target,0,rules.splice(index,1)[0]);
    if(presentedRule){const moved=rules.findIndex(rule=>rule.id===presentedRule.id);if(moved>=0)rules[moved]={...rules[moved],...presentedRule};}
  } else if (action === "create" || action === "duplicate") {
    if (!presentedRule) return false;
    const existing=rules.findIndex(rule=>rule.id===presentedRule.id);
    if(existing>=0) rules[existing]=presentedRule;
    else {
      const order=Number.isInteger(presentedRule.order) ? Math.max(0,Math.min(presentedRule.order,rules.length)) : rules.length;
      rules.splice(order,0,presentedRule);
    }
  } else if (action === "update") {
    if (!presentedRule) return false;
    const index=rules.findIndex(rule=>rule.id===presentedRule.id);
    if(index<0) return false;
    rules[index]=presentedRule;
  } else return false;
  rules=rules.map((rule,index)=>({...rule,order:index}));
  finishMutation(panel,result,rules);
  return true;
}

const setFuzzyState=(root,prefix)=>{const toggle=root.querySelector(`#${prefix}-fuzzy`),select=root.querySelector(`#${prefix}-threshold`);if(!toggle||!select)return;select.disabled=!toggle.checked;select.closest(".fuzzy-sensitivity")?.classList.toggle("is-disabled",!toggle.checked);};
function bindWordingEditor(panel){
  const root=panel.shadowRoot;
  const bindRemove=()=>root.querySelectorAll(".wording-remove").forEach(button=>{if(button.dataset.eocBound)return;button.dataset.eocBound="";button.addEventListener("click",()=>button.closest(".wording-group")?.remove());});
  bindRemove();
  const add=root.querySelector("#wording-add");
  if(add && !add.dataset.eocBound){add.dataset.eocBound="";add.addEventListener("click",()=>{const wrapper=document.createElement("div");wrapper.className="wording-group";wrapper.innerHTML='<label>Main phrase<input class="wording-canonical" maxlength="100"></label><label>Other ways to say it<input class="wording-alternatives" placeholder="Comma-separated alternatives"></label><button type="button" class="icon wording-remove" aria-label="Remove wording alternative">×</button>';root.querySelector("#wording-groups")?.append(wrapper);bindRemove();});}
}
export function bindRequestRulesCore(panel,{openEditor,activateSafeTester,activateLiveTester}={}) {
  const root=panel.shadowRoot,list=root.querySelector(".rule-list"); prepareRequestRulesCollection(panel);
  const sharing=root.querySelector("#rule-sharing");
  if(sharing&&!sharing.dataset.eocBound){sharing.dataset.eocBound="";sharing.addEventListener("toggle",()=>{if(sharing.open&&!sharing.dataset.eocLoaded){sharing.dataset.eocLoaded="";void import("./request-rule-sharing-ui.js").then(({bindRuleSharing})=>bindRuleSharing(panel,sharing));}});}
  setFuzzyState(root,"rules-default"); root.querySelector("#rules-default-fuzzy")?.addEventListener("change",()=>setFuzzyState(root,"rules-default"),{once:false});
  bindWordingEditor(panel);
  const filter=root.querySelector("#rule-group-filter");
  if(filter&&!filter.dataset.eocBound){filter.dataset.eocBound="";filter.addEventListener("change",()=>{panel._ruleGroupFilter=filter.value;reconcileRequestRules(panel);root.querySelector("#rule-search")?.dispatchEvent(new Event("input",{bubbles:true}));});}
  const groupManager=root.querySelector(".rule-groups");
  if(groupManager && !groupManager.dataset.eocBound){
    groupManager.dataset.eocBound="";
    const opener=root.querySelector("#rule-groups-manage");
    opener?.addEventListener("click",()=>{
      groupManager.querySelector(".rule-group-manager-rows").innerHTML=groupManagerRows(panel,panel._result?.groups||[],panel._result?.rules||[]);
      groupManager.showModal();
    });
    for(const id of ["rule-groups-close","rule-groups-done"]){
      groupManager.querySelector(`#${id}`)?.addEventListener("click",()=>groupManager.close());
    }
    groupManager.addEventListener("close",()=>opener?.focus());
    groupManager.querySelector("#rule-new-group-name")?.addEventListener("keydown",(event)=>{
      if(event.key==="Enter"){event.preventDefault();groupManager.querySelector("#rule-group-add")?.click();}
    });
    groupManager.addEventListener("click",async(event)=>{
      const button=event.target.closest?.("button");if(!button||button.disabled||groupManager.dataset.eocMutationPending==="true")return;
      const groups=[...(panel._result?.groups||[])],id=button.dataset.id;
      if(button.id==="rule-group-add"){
        const name=groupManager.querySelector("#rule-new-group-name")?.value.trim();if(!name)return;
        groups.push({name});
      }else if(button.matches(".rule-group-rename")){
        const name=button.closest(".rule-group-row")?.querySelector(".rule-group-name")?.value.trim();if(!name)return;
        const group=groups.find((item)=>item.id===id);if(group)group.name=name;
      }else if(button.matches(".rule-group-delete")){
        const group=groups.find((item)=>item.id===id);
        const count=(panel._result?.rules||[]).filter((rule)=>rule.group_id===id).length;
        if(!await panel._confirm(`Delete “${group?.name || "this"}” group?`,`${count} rule${count===1?"":"s"} will become Ungrouped and keep their existing priority and order.`,"Delete"))return;
        const index=groups.findIndex((item)=>item.id===id);if(index<0)return;groups.splice(index,1);
      }else return;
      button.disabled=true;groupManager.dataset.eocMutationPending="true";
      try{
        const result=await panel._call("request_rules","groups",{groups});
        panel._result={...(panel._result||{}),groups:result.groups,rules:result.rules,revision:result.revision};
        const cacheKey=panel._sectionCacheKey?.();if(cacheKey)panel._sectionCache?.delete(cacheKey);
        syncScopeRevision(panel,result);
        groupManager.querySelector(".rule-group-manager-rows").innerHTML=groupManagerRows(panel,result.groups,result.rules);
        groupManager.querySelector("#rule-new-group-name").value="";
        const filter=root.querySelector("#rule-group-filter");if(filter){filter.innerHTML=groupFilterOptions(panel,result.groups);if(!["all","ungrouped"].includes(panel._ruleGroupFilter)&&!result.groups.some(group=>group.id===panel._ruleGroupFilter))panel._ruleGroupFilter="all";filter.value=panel._ruleGroupFilter;}
        if (!reconcileRequestRules(panel)) panel._render();
        panel._toast("Groups saved");
      }catch(err){await recoverRequestRuleMutation(panel,err,"Unable to save groups");}
      finally{delete groupManager.dataset.eocMutationPending;button.disabled=false;}
    });
  }
  if(list && !list.dataset.eocRuleCoreBound){
    list.dataset.eocRuleCoreBound="";
    let dragging=null;
    // A disclosure of ordinary buttons uses native Enter/Space and Tab behavior.
    // Escape, outside clicks and completed actions dismiss it without losing focus.
    const closeMoveMenus = (except = null) => {
      list.querySelectorAll(".rule-move-menu[open]").forEach((menu) => {
        if (menu !== except) menu.open = false;
      });
    };
    list.addEventListener("keydown", (event) => {
      const menu = event.target.closest?.(".rule-move-menu[open]");
      if (event.key !== "Escape" || !menu) return;
      event.preventDefault();
      event.stopPropagation();
      menu.open = false;
      menu.querySelector("summary").focus();
    });
    // The listener belongs to this collection; replacing the route discards it.
    list.addEventListener("focusout", (event) => {
      // activeElement can be null while focus transfers. Use the destination
      // instead so Tab and pointer activation do not close the menu mid-transfer.
      closeMoveMenus(event.relatedTarget?.closest?.(".rule-move-menu"));
    });
    if (!root.__eocRuleMoveDismissBound) {
      root.__eocRuleMoveDismissBound = true;
      root.addEventListener("pointerdown", (event) => {
        const current = event.target.closest?.(".rule-move-menu");
        root.querySelectorAll(".rule-move-menu[open]").forEach((menu) => {
          if (menu !== current) menu.open = false;
        });
      });
    }
    list.addEventListener("dragstart",event=>{const card=event.target.closest?.("[data-rule-key]");if(!card||!card.draggable){event.preventDefault();return;}dragging=card.dataset.ruleKey;event.dataTransfer.effectAllowed="move";event.dataTransfer.setData("text/plain",dragging);});
    list.addEventListener("dragover",event=>{if(dragging&&event.target.closest?.("[data-rule-key]"))event.preventDefault();});
    list.addEventListener("dragend",()=>{dragging=null;});
    list.addEventListener("drop",async event=>{const card=event.target.closest?.("[data-rule-key]");const id=dragging;dragging=null;if(!id||!card||!card.draggable||card.dataset.ruleKey===id)return;event.preventDefault();const targetId=card.dataset.ruleKey,direction=event.clientY>card.getBoundingClientRect().top+card.getBoundingClientRect().height/2?"after":"before";try{const result=await panel._call("request_rules","move",{rule_id:id,direction,target_rule_id:targetId});applyRequestRuleMutation(panel,"move",result,{ruleId:id,direction,targetRuleId:targetId});}catch(err){await recoverRequestRuleMutation(panel,err,"Unable to move Request Rule");}});
    list.addEventListener("click",async(event)=>{
      const button=event.target.closest?.("button");if(!button||button.disabled)return;
      if(button.id==="rule-empty-add"||button.matches(".rule-edit")){void openEditor?.(button.dataset.id||null);return;}
      const id=button.dataset.id;
      if(button.matches(".rule-move")){
        const menu = button.closest(".rule-move-menu");
        const trigger = menu?.querySelector("summary");
        if (menu) { menu.open = false; trigger.focus(); }
        let restoreFocus = false;
        pendingRuleButtons.add(button);button.disabled=true;
        try{
          const result=await panel._call("request_rules","move",{rule_id:id,direction:button.dataset.direction});
          // Moving a keyed DOM node can drop focus. Do not steal it back if the
          // user intentionally focused another control while the save ran.
          restoreFocus = root.activeElement === trigger;
          applyRequestRuleMutation(panel,"move",result,{ruleId:id,direction:button.dataset.direction});
        }
        catch(err){await recoverRequestRuleMutation(panel,err,"Unable to move Request Rule");}
        finally{
          pendingRuleButtons.delete(button);
          reconcileRequestRules(panel);
          if (restoreFocus && trigger?.isConnected) trigger.focus();
        }
        return;
      }
      if(button.matches(".rule-duplicate")){
        button.disabled=true;
        try{const result=await panel._call("request_rules","duplicate",{rule_id:id});applyRequestRuleMutation(panel,"duplicate",result,{ruleId:id});panel._toast("Request Rule duplicated");}
        catch(err){await recoverRequestRuleMutation(panel,err,"Unable to duplicate Request Rule");} finally{button.disabled=false;} return;
      }
      if(button.matches(".rule-delete")){
        const deleting=(panel._result?.rules||[]).find((rule)=>rule.id===id);
        panel._eocDecisionConfirmSubject=deleting?.name?`Request Rule “${deleting.name}”`:"Request Rule";
        if(!await panel._confirm("Delete Request Rule?","This cannot be undone.","Delete"))return;
        button.disabled=true;
        try{const result=await panel._call("request_rules","delete",{rule_id:id,confirm:true});applyRequestRuleMutation(panel,"delete",result,{ruleId:id});panel._toast("Request Rule deleted");}
        catch(err){await recoverRequestRuleMutation(panel,err,"Unable to delete Request Rule");} finally{button.disabled=false;}
      }
    });
    list.addEventListener("change",async(event)=>{
      const input=event.target;if(!input.matches?.(".rule-enabled")||input.disabled)return;
      const rule=(panel._result?.rules||[]).find(item=>item.id===input.dataset.id);if(!rule)return;
      const previous=!input.checked;input.disabled=true;
      try{const result=await panel._call("request_rules","update",{rule_id:rule.id,rule:{...rule,enabled:input.checked,sensitive_matching_warning:undefined}});applyRequestRuleMutation(panel,"update",result,{ruleId:rule.id});panel._toast("Changes saved");}
      catch(err){input.checked=previous;await recoverRequestRuleMutation(panel,err,"Unable to update Request Rule");}
      finally{input.disabled=false;}
    });
  }
  const add=root.querySelector("#rule-add");if(add&&!add.dataset.eocRuleCoreBound){add.dataset.eocRuleCoreBound="";add.addEventListener("click",()=>{void openEditor?.(null);});}
  const safe=root.querySelector("#rule-match-tester");if(safe&&!safe.dataset.eocLazyBound){safe.dataset.eocLazyBound="";for(const type of ["focusin","pointerdown"])safe.addEventListener(type,()=>{void activateSafeTester?.();},{once:true,capture:true});}
  const live=root.querySelector("#eoc-rule-live-test");if(live&&!live.dataset.eocLazyBound){live.dataset.eocLazyBound="";live.addEventListener("toggle",()=>{if(live.open)void activateLiveTester?.();},{once:true});}
}
