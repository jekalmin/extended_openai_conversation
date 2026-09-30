import {applyRequestRuleMutation, fuzzyThresholdValue, matchingControls} from "./request-rules-ui-core.js";
export {fuzzyThresholdValue, recoverRequestRuleMutation, reconcileRequestRules, renderRequestRules} from "./request-rules-ui-core.js";

export function applySentencePatternHelper(value, selectionStart, selectionEnd, kind) {
  const start = Math.max(0, Math.min(Number(selectionStart) || 0, value.length));
  const end = Math.max(start, Math.min(Number(selectionEnd) || start, value.length));
  const selected = value.slice(start, end);
  let snippet;
  let editStart;
  let editEnd;
  if (kind === "optional") {
    const inner = selected || "optional words";
    snippet = `[${inner}]`;
    editStart = start + 1;
    editEnd = editStart + inner.length;
  } else if (kind === "choice") {
    const inner = selected ? `${selected}|alternative` : "one|two";
    snippet = `(${inner})`;
    editStart = start + 1;
    editEnd = editStart + inner.length;
  } else if (kind === "variable") {
    const name = /^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(selected) ? selected : "name";
    snippet = `{${name}}`;
    editStart = start + 1;
    editEnd = editStart + name.length;
  } else if (kind === "range") {
    const name = /^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(selected) ? selected : "level";
    snippet = `{${name}=0..100}`;
    editStart = start + 1;
    editEnd = editStart + name.length;
  } else {
    throw new Error(`Unknown sentence-pattern helper: ${kind}`);
  }
  return {
    value: `${value.slice(0, start)}${snippet}${value.slice(end)}`,
    selectionStart: editStart,
    selectionEnd: editEnd,
  };
}

export function requestRulesDialog() {
  return `<dialog id="rule-dialog" class="editor-dialog wide request-rule-dialog" aria-labelledby="rule-dialog-title"><form id="rule-form"><div class="dialog-header"><h2 id="rule-dialog-title">Create Request Rule</h2><button type="button" class="icon rule-close" aria-label="Close">×</button></div><div class="dialog-body"><label>Rule name<input id="rule-name" required maxlength="120" placeholder="Shopping list"></label><label>Group<select id="rule-group"><option value="">Ungrouped</option></select><small>Groups organize rules; the global top-to-bottom order still determines priority.</small></label><section><h3>1. What will you say?</h3><label>Trigger phrases or patterns<textarea id="rule-phrases" required placeholder="Add {item} to my shopping list"></textarea><small>Put each alternative on a new line. Alternatives must use the same variable names.</small></label><div id="rule-slot-help" class="notice" hidden><strong>Variable values</strong><p>Variable values let part of the request change each time. You can use the captured value in actions or responses.</p><p id="rule-slot-list"></p></div><label>How should it match?<select id="rule-match"><option value="equals">Equals</option><option value="starts_with">Starts with</option><option value="ends_with">Ends with</option><option value="contains">Contains</option><option value="sentence_pattern">ExtendedOpenAI sentence pattern</option></select></label><div id="sentence-pattern-builder" class="section-actions" hidden><span class="help">Insert pattern:</span><button type="button" class="secondary pattern-helper" data-pattern-helper="optional">Optional</button><button type="button" class="secondary pattern-helper" data-pattern-helper="choice">Choice</button><button type="button" class="secondary pattern-helper" data-pattern-helper="variable">Variable</button><button type="button" class="secondary pattern-helper" data-pattern-helper="range">Number range</button></div><div id="sentence-pattern-help" hidden><p class="help">Example: <code>[please] set {room} to {level=0..100}</code>.</p><details class="eoc-details-base"><summary>Sentence-pattern syntax reference</summary><p>Use <code>[optional words]</code>, <code>(one|two)</code>, free-text values such as <code>{room}</code>, constrained values such as <code>{room=kitchen|bedroom}</code>, and integer ranges such as <code>{level=0..100}</code>. Escape syntax characters with <code>\\</code>, including <code>\\|</code> inside choices. Sentence-ending punctuation is tolerated. This is ExtendedOpenAI syntax; named expansions and permutations are not supported.</p></details></div></section><section id="rule-conditions-section"><h3>2. Only when</h3><button type="button" class="secondary" id="rule-add-conditions" aria-controls="rule-conditions-body">Add conditions — optional</button><div id="rule-conditions-body" hidden><p class="help">Optional Home Assistant conditions. A matching rule is skipped when these are false; the next rule is checked.</p><div id="rule-condition-host"></div></div></section><section><h3>3. What should happen?</h3><label>Behaviour<select id="rule-action-type"><option value="local_action">Run actions locally</option><option value="model_routing">Route through AI with different settings</option></select></label><div id="rule-local-config"><p class="help">Build a native Home Assistant action sequence that runs locally before optional AI continuation. Conditions, delays, choose, repeat, parallel, and templates use the same editor and syntax as scripts and automations.</p><div id="rule-action-sequence-host"></div><div id="rule-result-aliases"></div><label class="matching-setting"><span class="matching-copy"><span class="matching-title">Continue to AI</span><small>After the local sequence completes, send the selected AI input to the provider unless an action sets a conversation response or stops the script.</small></span><input id="rule-local-continue-to-ai" type="checkbox"></label><div id="rule-action-slot-help" class="notice" hidden><strong>Captured values in actions</strong><p id="rule-action-slot-list"></p><p>Use a captured value as a script variable, for example <code>{{ item }}</code>. The same values are also available under <code>request.slots</code>.</p><p>To call an enabled configured function, add <code>extended_openai_conversation_responses.call_function</code> and provide its name and arguments.</p></div></div><div id="rule-routing-config" hidden><p class="help"><strong>Equals</strong> and <strong>ExtendedOpenAI sentence pattern</strong> are complete commands by default. Enable <strong>Continue to AI</strong> to send the selected AI input after applying the route. Broader Starts/Ends/Contains matches continue to the provider by default.</p><p class="help" id="rule-routing-scope-help"></p><label class="matching-setting"><span class="matching-copy"><span class="matching-title">Continue to AI</span><small>After applying these routing settings, send the selected AI input to the provider.</small></span><input id="rule-continue-to-ai" type="checkbox" checked></label><div class="form-grid"><label>Model<input id="rule-model" placeholder="gpt-5-mini"></label><label>Reasoning effort<select id="rule-reasoning"><option value="">Keep current</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></label><label>Scope<select id="rule-scope"><option value="request">This request only</option><option value="conversation">Rest of this conversation</option></select></label><label class="toggle"><span>Reset to configured defaults</span><input id="rule-reset" type="checkbox"></label></div></div><div id="rule-ai-input-settings"><label>AI input<select id="rule-ai-input-mode"><option value="original">Original request</option><option value="capture">Captured value</option></select></label><label id="rule-ai-capture-label" hidden>Captured value<select id="rule-ai-input-capture"></select></label><p id="rule-ai-input-help" class="help" hidden></p></div></section><section><h3>4. What should the assistant say?</h3><div id="rule-local-responses" class="form-grid"><label>Success response<input id="rule-success" value="Done"><small>Used when local actions finish or Stop succeeds without a conversation response. Set conversation response overrides it. Example: <code>Added {item}</code>.</small></label><label>Failure response<input id="rule-failure" value="Sorry, that did not work"></label></div><p id="rule-routing-ai-response" class="help" hidden>The AI provider will generate the response.</p><label id="rule-routing-response" hidden>Acknowledgement<input id="rule-routing-success" value="Updated"></label><details id="rule-response-reference" class="eoc-details-base rule-response-reference"><summary>Response substitution reference</summary><p>Use request captures such as <code>{item}</code> or named Function results such as <code>{battery.level}</code>; native script variables are only available inside actions.</p></details></section><details id="rule-advanced" class="advanced-context-formatting eoc-details-base"><summary>Advanced matching and action configuration</summary><label class="matching-setting"><span class="matching-copy"><span class="matching-title">Continue matching after this rule</span><small>Let later matching rules run too. Rules are checked in order. AI handoff or an error stops further matching.</small></span><input id="rule-continue-matching" type="checkbox"></label><label>Matching behaviour<select id="rule-matching-behavior"><option value="defaults">Use default settings</option><option value="custom">Customize for this rule</option></select></label>${matchingControls("rule", {word_forms:true,wording_alternatives:true,fuzzy:false,fuzzy_threshold:90}, true)}<p class="help">Sentence patterns use ExtendedOpenAI's bounded matcher, so fuzzy matching, wording alternatives, and word-form normalization do not apply. Advanced Home Assistant JSON can use <code>{slot}</code> in text values.</p></details><div id="rule-error" class="inline-error" role="alert"></div></div><div class="dialog-actions"><button type="button" class="secondary rule-close">Cancel</button><button type="submit" id="rule-save">Save</button></div></form></dialog>`;
}


export function capturedSlotNames(text) {
  // Discover editor bindings without interpreting constrained values as syntax.
  // Backend validation remains authoritative for incomplete/invalid patterns.
  const source = String(text || ""), names = new Set();
  for (let index = 0; index < source.length; index += 1) {
    if (source[index] === "\\") { index += 1; continue; }
    if (source[index] !== "{") continue;
    let body = "", closed = false;
    while (++index < source.length) {
      if (source[index] === "\\") {
        body += source[index];
        if (++index < source.length) body += source[index];
      } else if (source[index] === "}") {
        closed = true;
        break;
      } else body += source[index];
    }
    const name = body.split("=", 1)[0].trim();
    if (closed && /^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(name)) names.add(name);
  }
  return [...names];
}

export function commonCapturedSlotNames(text) {
  const variants = String(text || "").split("\n").map((line) => line.trim()).filter(Boolean);
  if (!variants.length) return [];
  const required=(variant)=>{
    let index=0;
    const sequence=(closing=null)=>{
      const branches=[];
      let closed=false;
      let names=new Set();
      while(index<variant.length){
        const char=variant[index++];
        if(char==="\\"){index++;continue;}
        if(char===closing){branches.push(names);closed=true;break;}
        if(char===")"||char==="]")return null;
        if(char==="|"&&closing===")"){branches.push(names);names=new Set();continue;}
        if(char==="("||char==="["){
          const nested=sequence(char==="("?")":"]");
          if(nested===null)return null;
          if(char==="(")for(const name of nested)names.add(name);
          continue;
        }
        if(char==="{"){
          let body="",closed=false;
          while(index<variant.length){
            const next=variant[index++];
            if(next==="\\"&&index<variant.length){body+=next+variant[index++];continue;}
            if(next==="}"){closed=true;break;}
            body+=next;
          }
          if(!closed)return null;
          const name=body.split("=",1)[0].trim();
          if(/^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(name))names.add(name);
        }
      }
      if(closing&&!closed)return null;
      if(!closing)branches.push(names);
      const common=new Set(branches[0]||[]);
      for(const branch of branches.slice(1))for(const name of common)if(!branch.has(name))common.delete(name);
      return common;
    };
    const names=sequence();
    return names===null?[]:[...names];
  };
  return required(variants[0]).filter((name) => variants.every((variant) => required(variant).includes(name)));
}

const setFuzzyState = (root, prefix) => { const toggle = root.querySelector(`#${prefix}-fuzzy`), select = root.querySelector(`#${prefix}-threshold`); if (!toggle || !select) return; select.disabled = !toggle.checked; select.closest(".fuzzy-sensitivity")?.classList.toggle("is-disabled", !toggle.checked); };


const editorTranslations = new WeakMap();

export function ensureRequestRuleTranslations(panel) {
  const hass = panel.hass || panel._hass;
  if (!hass?.loadFragmentTranslation) return Promise.resolve();
  const language = hass.language || hass.locale?.language;
  const cached = editorTranslations.get(panel);
  if (cached?.language === language && cached.connection === hass.connection) return cached.promise;
  const promise = Promise.resolve(hass.loadFragmentTranslation("config")).then(localize => {
    if ((panel.hass || panel._hass)?.language !== hass.language) return;
    panel.shadowRoot.querySelectorAll("#rule-condition-host ha-selector, #rule-action-sequence-host ha-selector").forEach(selector => {
      selector.hass = localize ? {...(panel.hass || panel._hass), localize} : (panel.hass || panel._hass);
    });
  }).catch(error => {
    editorTranslations.delete(panel);
    panel._toast?.(`Unable to load Home Assistant editor labels: ${error.message || String(error)}`, true);
  });
  editorTranslations.set(panel, {language, connection:hass.connection, promise});
  return promise;
}

export function createRequestRuleActionSelector(panel, host) {
  const actionSelector = host.ownerDocument.createElement("ha-selector");
  actionSelector.hass = panel._hass;
  actionSelector.selector = {action:{}};
  actionSelector.value = [];
  actionSelector.addEventListener("value-changed", (event) => {
    actionSelector.value = event.detail.value || [];
  });
  host.replaceChildren(actionSelector);
  return actionSelector;
}

export function createRequestRuleConditionSelector(panel, host) {
  const selector = host.ownerDocument.createElement("ha-selector");
  selector.hass = panel._hass;
  selector.selector = {condition:{}};
  selector.value = [];
  selector.addEventListener("value-changed", (event) => { selector.value = event.detail.value || []; void labelConditionAddControl(panel, selector); });
  host.replaceChildren(selector);
  void labelConditionAddControl(panel, selector);
  return selector;
}

async function labelConditionAddControl(panel, selector) {
  // The native selector owns the plus button inside its nested shadow roots.
  // Label that control after Lit finishes rendering, without replacing its UI.
  const registry = selector.ownerDocument?.defaultView?.customElements || globalThis.customElements;
  if (!registry) return;
  await registry.whenDefined("ha-selector");
  let element = selector;
  for (const tag of ["ha-selector-condition", "ha-automation-condition"]) {
    await element.updateComplete;
    await registry.whenDefined(tag);
    element = element.shadowRoot?.querySelector(tag);
    if (!element) return;
  }
  await element.updateComplete;
  const control = element.shadowRoot?.querySelector(".buttons ha-button, .buttons ha-icon-button");
  if (!control) return;
  const label = panel._hass?.localize?.("ui.panel.config.automation.editor.conditions.add") || "Add condition";
  control.setAttribute("aria-label", label);
  control.label = label;
  await control.updateComplete;
  control.shadowRoot?.querySelector("button")?.setAttribute("aria-label", label);
}

export function renameResultReferences(value, oldAlias, newAlias) {
  if (typeof value === "string") return value.replace(/(?<!\{)\{[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_0-9][A-Za-z0-9_]*)*\}(?!\})/g,
    (whole) => { const body=whole.slice(1,-1), alias=body.split(".",1)[0]; return alias === oldAlias ? `{${newAlias}${body.slice(alias.length)}}` : whole; });
  if (Array.isArray(value)) return value.map((item) => renameResultReferences(item, oldAlias, newAlias));
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, key === "result_alias" ? item : renameResultReferences(item, oldAlias, newAlias)]));
  return value;
}

export function suggestResultAlias(functionName, used = []) {
  const base = (String(functionName || "result").replace(/[^A-Za-z0-9_]/g, "_").replace(/^[^A-Za-z_]+/, "") || "result").slice(0,60);
  const taken = new Set(used);
  for (const reserved of ["request","conversation","system","trigger","this","repeat","wait"]) taken.add(reserved);
  if (!taken.has(base)) return base;
  for (let suffix=2; suffix<1000; suffix++) if (!taken.has(`${base}_${suffix}`)) return `${base}_${suffix}`;
  return "result";
}

function renderResultAliases(panel, selector) {
  const host = panel.shadowRoot.querySelector("#rule-result-aliases");
  if (!host) return;
  const actions = selector.value || [];
  const calls = actions.map((action, index) => ({action, index})).filter(({action}) => action?.action === "extended_openai_conversation_responses.call_function" && action?.data?.function);
  const suggested = new Set(calls.map(({action}) => action.data.result_alias).filter(Boolean));
  host.innerHTML = calls.map(({action, index}) => {
    const alias = action.data.result_alias || "";
    const suggestion = suggestResultAlias(action.data.function, suggested);
    suggested.add(suggestion);
    return `<label>Result alias for ${panel._e(action.data.function)} (step ${index + 1})<input class="rule-result-alias" data-index="${index}" value="${panel._e(alias)}" placeholder="${panel._e(suggestion)}" pattern="[A-Za-z_][A-Za-z0-9_]*"><small>Optional. Name the result to use it in later steps or the local response. Each captured result needs a unique alias.</small></label>`;
  }).join("");
  host.querySelectorAll(".rule-result-alias").forEach((input) => input.addEventListener("change", () => {
    const index = Number(input.dataset.index), actions = structuredClone(selector.value || []);
    if (!actions[index]) return;
    actions[index].data ||= {};
    if (input.value.trim()) actions[index].data.result_alias = input.value.trim();
    else delete actions[index].data.result_alias;
    selector.value = actions;
  }));
}

export function loadRequestRuleActions(actionSelector, rule) {
  actionSelector.value = structuredClone(rule?.action?.actions || []);
}

export function readRequestRuleActions(actionSelector) {
  return actionSelector.value || [];
}



let modelCatalogModule = null;
let modelCatalogPromise = null;
function ensureModelCatalog() {
  if (modelCatalogModule) return Promise.resolve(modelCatalogModule);
  if (!modelCatalogPromise) {
    modelCatalogPromise = import("./model-catalog.js").then((module) => {
      modelCatalogModule = module;
      return module;
    }).finally(() => { modelCatalogPromise = null; });
  }
  return modelCatalogPromise;
}

function setReasoningOptions(root, efforts, selected = "") {
  const select = root?.querySelector("#rule-reasoning");
  if (!select) return;
  const values = Array.isArray(efforts) ? efforts : [];
  const keepCurrent = select.ownerDocument.createElement("option");
  keepCurrent.value = "";
  keepCurrent.textContent = "Keep current";
  const options = [keepCurrent, ...values.map((value) => {
    const option = select.ownerDocument.createElement("option");
    option.value = value;
    option.textContent = value.charAt(0).toUpperCase() + value.slice(1);
    return option;
  })];
  select.replaceChildren(...options);
  select.value = selected && values.includes(selected) ? selected : "";
}

export function syncRequestRuleRoutingControls(root, efforts = null, selectedEffort = null) {
  if (!root) return;
  const actionType = root.querySelector("#rule-action-type");
  const matchType = root.querySelector("#rule-match");
  const scope = root.querySelector("#rule-scope");
  const continueToAi = root.querySelector("#rule-continue-to-ai");
  const reasoning = root.querySelector("#rule-reasoning");
  const help = root.querySelector("#rule-routing-scope-help");
  if (!actionType || !matchType || !scope) return;
  const modelRouting = actionType.value === "model_routing";
  const consumed = modelRouting && ["equals", "sentence_pattern"].includes(matchType.value) && !(continueToAi?.checked ?? false);
  const requestOption = scope.querySelector('option[value="request"]');
  if (requestOption) requestOption.disabled = consumed;
  if (consumed) scope.value = "conversation";
  scope.disabled = consumed;
  if (reasoning && efforts) {
    const desired = selectedEffort ?? reasoning.value;
    setReasoningOptions(root, efforts, desired);
  }
  if (help) help.textContent = consumed
    ? "This is a complete routing command. It is acknowledged locally and is not sent to the AI provider, so it must change or reset the rest of this conversation."
    : `This sends the ${root.querySelector("#rule-ai-input-mode")?.value === "capture" ? "selected captured value" : "original request"} to the AI after applying the route. This request only affects that provider call; Rest of this conversation also changes later requests.`;
}

function ensureDialog(panel) {
  const root = panel.shadowRoot;
  let dialog = root.querySelector("#rule-dialog");
  if (dialog) return dialog;
  const template = document.createElement("template");
  template.innerHTML = requestRulesDialog();
  dialog = template.content.firstElementChild;
  (root.querySelector("#eoc-dialog-host") || root).append(dialog);
  return dialog;
}

function editorState(panel) {
  panel._eocRuleEditorState ||= {dialog:null, actionSelector:null, revision:null, modelRevision:0};
  return panel._eocRuleEditorState;
}

function refreshEditor(panel) {
  const root=panel.shadowRoot, q=(selector)=>root.querySelector(selector);
  const local=q("#rule-action-type").value==="local_action";
  const grammar=q("#rule-match").value==="sentence_pattern";
  const slots=capturedSlotNames(q("#rule-phrases").value);
  q("#rule-local-config").hidden=!local;
  q("#rule-routing-config").hidden=local;
  q("#rule-local-responses").hidden=!local || q("#rule-local-continue-to-ai").checked;
  const continueToAi=q("#rule-continue-to-ai")?.checked ?? true;
  q("#rule-routing-response").hidden=local||continueToAi;
  q("#rule-routing-ai-response").hidden=local ? !q("#rule-local-continue-to-ai").checked : !continueToAi;
  q("#sentence-pattern-help").hidden=!grammar;
  q("#sentence-pattern-builder").hidden=!grammar;
  q("#rule-slot-help").hidden=!slots.length;
  q("#rule-slot-list").textContent=slots.length?`Captured values: ${slots.join(", ")}`:"";
  q("#rule-action-slot-help").hidden=!slots.length;
  q("#rule-action-slot-list").textContent=slots.length?slots.map((name)=>`{{ ${name} }}`).join(", "):"";
  q("#rule-matching-behavior").disabled=grammar;
  q("#rule-matching-controls").hidden=grammar||q("#rule-matching-behavior").value!=="custom";
  const handoff=local ? q("#rule-local-continue-to-ai").checked : continueToAi;
  q("#rule-response-reference").hidden=handoff;
  const aiSettings=q("#rule-ai-input-settings"), mode=q("#rule-ai-input-mode"), capture=q("#rule-ai-input-capture"), label=q("#rule-ai-capture-label"), help=q("#rule-ai-input-help");
  aiSettings.hidden=!handoff && mode.value!=="capture";
  const common=q("#rule-match").value==="sentence_pattern" ? commonCapturedSlotNames(q("#rule-phrases").value) : [];
  const previous=capture.value;
  capture.replaceChildren(...common.map((name)=>{const option=document.createElement("option");option.value=name;option.textContent=name;return option;}));
  if(previous && !common.includes(previous)){const option=document.createElement("option");option.value=previous;option.textContent=`${previous} (unavailable)`;capture.prepend(option);}
  capture.value=previous || (common.length===1 ? common[0] : "");
  label.hidden=mode.value!=="capture";
  const invalid=mode.value==="capture" && (!handoff || !common.length || !common.includes(capture.value));
  capture.setAttribute("aria-invalid",String(invalid));
  help.hidden=!invalid;
  help.textContent=!handoff ? "Enable Continue to AI or choose Original request." : q("#rule-match").value!=="sentence_pattern" ? "Captured value requires Sentence Pattern matching." : common.length ? "The selected captured value is not available in every trigger. Choose a common value or Original request." : "No captured values are available for every trigger. Use Sentence Pattern matching and make each trigger capture the same value, such as {question}.";
  syncRequestRuleRoutingControls(root);
  if(local)return;
  const model=q("#rule-model")?.value.trim();
  if(!model)return;
  const state=editorState(panel), current=++state.modelRevision;
  const selected=(panel._result?.rules||[]).find((item)=>item.id===panel._editingRuleId)?.action?.reasoning_effort || q("#rule-reasoning")?.value || "";
  void ensureModelCatalog().then((module)=>module.lookupModelData(panel,model)).then((data)=>{
    if(current===state.modelRevision && root.querySelector("#rule-dialog")?.open) syncRequestRuleRoutingControls(root,data.reasoning_effort_options,selected);
  }).catch((err)=>panel._toast(`Unable to load model choices: ${err.message || String(err)}`,true));
}

function hasRuleConditions(value) {
  if (Array.isArray(value)) return value.length > 0;
  if (value && typeof value === "object") return Object.keys(value).length > 0;
  return Boolean(value);
}

function showRuleConditions(panel, visible = true, focus = false) {
  const root = panel.shadowRoot;
  root.querySelector("#rule-add-conditions").hidden = visible;
  root.querySelector("#rule-conditions-body").hidden = !visible;
  if (visible && focus) {
    const selector = editorState(panel).conditionSelector;
    selector.tabIndex = -1;
    selector.focus();
  }
}

export function bindRequestRuleEditor(panel) {
  const root=panel.shadowRoot, dialog=ensureDialog(panel), state=editorState(panel);
  if(state.dialog===dialog)return dialog;
  state.dialog=dialog;
  const q=(selector)=>root.querySelector(selector);
  state.actionSelector=createRequestRuleActionSelector(panel,q("#rule-action-sequence-host"));
  state.conditionSelector=createRequestRuleConditionSelector(panel,q("#rule-condition-host"));
  q("#rule-add-conditions").addEventListener("click", () => showRuleConditions(panel, true, true));
  q("#rule-form").addEventListener("invalid", (event) => {
    if (q("#rule-conditions-body").contains(event.target)) showRuleConditions(panel);
  }, true);
  state.actionSelector.addEventListener("value-changed",()=>queueMicrotask(()=>renderResultAliases(panel,state.actionSelector)));
  q("#rule-fuzzy")?.addEventListener("change",()=>setFuzzyState(root,"rule"));
  q("#rule-action-type")?.addEventListener("change",()=>refreshEditor(panel));
  q("#rule-match")?.addEventListener("change",()=>refreshEditor(panel));
  q("#rule-continue-to-ai")?.addEventListener("change",()=>refreshEditor(panel));
  q("#rule-local-continue-to-ai")?.addEventListener("change",()=>refreshEditor(panel));
  q("#rule-model")?.addEventListener("input",()=>refreshEditor(panel));
  q("#rule-ai-input-mode")?.addEventListener("change",()=>refreshEditor(panel));
  q("#rule-ai-input-capture")?.addEventListener("change",()=>refreshEditor(panel));
  q("#rule-scope")?.addEventListener("change",()=>syncRequestRuleRoutingControls(root));
  q("#rule-continue-matching")?.addEventListener("change",()=>syncRequestRuleRoutingControls(root));
  q("#rule-reset")?.addEventListener("change",()=>syncRequestRuleRoutingControls(root));
  root.querySelectorAll(".pattern-helper").forEach((button)=>button.addEventListener("click",()=>{
    const textarea=q("#rule-phrases");
    const result=applySentencePatternHelper(textarea.value,textarea.selectionStart,textarea.selectionEnd,button.dataset.patternHelper);
    textarea.value=result.value;textarea.focus();textarea.setSelectionRange(result.selectionStart,result.selectionEnd);refreshEditor(panel);
  }));
  q("#rule-phrases")?.addEventListener("input",()=>refreshEditor(panel));
  q("#rule-matching-behavior")?.addEventListener("change",()=>refreshEditor(panel));
  root.querySelectorAll(".rule-close").forEach((button)=>button.addEventListener("click",()=>dialog.close()));
  q("#rule-form")?.addEventListener("submit",async(event)=>{
    event.preventDefault();
    const save=q("#rule-save");if(save.disabled)return;
    panel._setSaving(save,true);
    try{
      const actionType=q("#rule-action-type").value;
      let actions=readRequestRuleActions(state.actionSelector);
      if(actionType==="local_action"&&!actions.length)throw new Error("Add at least one action before saving this rule.");
      const previous=(panel._result?.rules||[]).find((item)=>item.id===panel._editingRuleId);
      for (const oldStep of previous?.action?.actions || []) {
        const oldAlias=oldStep?.data?.result_alias, stepId=oldStep?.data?.step_id;
        if (!oldAlias || !stepId) continue;
        const edited=actions.find((step)=>step?.data?.step_id===stepId);
        if (edited?.data?.result_alias && edited.data.result_alias!==oldAlias) {
          actions=renameResultReferences(actions,oldAlias,edited.data.result_alias);
          q("#rule-success").value=renameResultReferences(q("#rule-success").value,oldAlias,edited.data.result_alias);
        }
      }
      const rules=panel._result?.rules||[];
      if(q("#rule-ai-input-mode").value==="capture" && (q("#rule-ai-input-settings").hidden || q("#rule-ai-input-capture").getAttribute("aria-invalid")==="true")) throw new Error(q("#rule-ai-input-help").textContent || "Choose a capture available in every trigger.");
      const rule={ai_input_mode:q("#rule-ai-input-mode").value,ai_input_capture:q("#rule-ai-input-mode").value==="capture"?q("#rule-ai-input-capture").value:null,name:q("#rule-name").value,enabled:previous?.enabled??true,phrases:q("#rule-phrases").value.split("\n").map((item)=>item.trim()).filter(Boolean),match_type:q("#rule-match").value,action_type:actionType,action:actionType==="local_action"?{actions,success_response:q("#rule-success").value,failure_response:q("#rule-failure").value,continue_to_ai:q("#rule-local-continue-to-ai").checked}:{model:q("#rule-model").value,reasoning_effort:q("#rule-reasoning").value,scope:q("#rule-scope").value,reset:q("#rule-reset").checked,continue_to_ai:q("#rule-continue-to-ai").checked,success_response:q("#rule-routing-success").value},matching_behavior:q("#rule-matching-behavior").value,matching:{word_forms:q("#rule-word-forms").checked,wording_alternatives:q("#rule-wording").checked,fuzzy:q("#rule-fuzzy").checked,fuzzy_threshold:fuzzyThresholdValue(q("#rule-threshold").value)},order:rules.find((item)=>item.id===panel._editingRuleId)?.order??rules.length,conditions:state.conditionSelector.value||[],group_id:q("#rule-group").value||null,continue_matching:q("#rule-continue-matching").checked};
      const action=panel._editingRuleId?"update":"create";
      const result=await panel._call("request_rules",action,{...(panel._editingRuleId?{rule_id:panel._editingRuleId}:{}),rule,revision:state.revision});
      dialog.close();
      applyRequestRuleMutation(panel,action,result,{ruleId:panel._editingRuleId});
    }catch(err){
      const message = err.message || String(err);
      q("#rule-error").textContent = message;
      if (hasRuleConditions(state.conditionSelector.value)
          || /condition/i.test(`${err.code || ""} ${err.field || ""} ${message}`)) {
        showRuleConditions(panel);
      }
    }
    finally{panel._setSaving(save,false);}
  });
  return dialog;
}

export function openRequestRuleEditor(panel,id=null) {
  const dialog=bindRequestRuleEditor(panel), root=panel.shadowRoot, q=(selector)=>root.querySelector(selector), state=editorState(panel);
  // Native YAML children retain their own draft and mode. Each opening owns
  // fresh selectors, so Cancel cannot leak an unsaved child draft into saved data.
  state.actionSelector=createRequestRuleActionSelector(panel,q("#rule-action-sequence-host"));
  state.conditionSelector=createRequestRuleConditionSelector(panel,q("#rule-condition-host"));
  state.actionSelector.addEventListener("value-changed",()=>queueMicrotask(()=>renderResultAliases(panel,state.actionSelector)));
  void ensureRequestRuleTranslations(panel);
  state.revision=panel._result?.revision;
  const rule=(panel._result?.rules||[]).find((item)=>item.id===id);
  panel._editingRuleId=id;
  q("#rule-dialog-title").textContent=rule?"Edit Request Rule":"Create Request Rule";
  q("#rule-name").value=rule?.name||"";
  q("#rule-continue-matching").checked=rule?.continue_matching??false;
  q("#rule-ai-input-mode").value=rule?.ai_input_mode||"original";
  q("#rule-ai-input-capture").innerHTML=rule?.ai_input_capture?`<option value="${panel._e(rule.ai_input_capture)}">${panel._e(rule.ai_input_capture)}</option>`:"";
  q("#rule-phrases").value=(rule?.phrases||[]).join("\n");
  q("#rule-match").value=rule?.match_type||"equals";
  q("#rule-action-type").value=rule?.action_type||"local_action";
  q("#rule-success").value=rule?.action?.success_response||"Done";
  q("#rule-failure").value=rule?.action?.failure_response||"Sorry, that did not work";
  q("#rule-local-continue-to-ai").checked=rule?.action_type==="local_action" ? Boolean(rule?.action?.continue_to_ai) : false;
  state.conditionSelector.value=structuredClone(rule?.conditions||[]);
  showRuleConditions(panel, hasRuleConditions(state.conditionSelector.value));
  const groupSelect=q("#rule-group");
  groupSelect.replaceChildren(...[{id:"",name:"Ungrouped"},...(panel._result?.groups||[])].map((group)=>{const option=groupSelect.ownerDocument.createElement("option");option.value=group.id;option.textContent=group.name;return option;}));
  groupSelect.value=rule?.group_id||"";
  q("#rule-model").value=rule?.action?.model||"";
  q("#rule-reasoning").value=rule?.action?.reasoning_effort||"";
  q("#rule-scope").value=rule?.action?.scope||"request";
  q("#rule-reset").checked=rule?.action?.reset||false;
  q("#rule-continue-to-ai").checked=rule?.action_type==="model_routing"?(rule?.action?.continue_to_ai??!["equals","sentence_pattern"].includes(rule?.match_type)):true;
  q("#rule-routing-success").value=rule?.action?.success_response||"Updated";
  q("#rule-matching-behavior").value=rule?.matching_behavior||"defaults";
  q("#rule-word-forms").checked=rule?.matching?.word_forms??true;
  q("#rule-wording").checked=rule?.matching?.wording_alternatives??true;
  q("#rule-fuzzy").checked=rule?.matching?.fuzzy??false;
  q("#rule-threshold").value=String(fuzzyThresholdValue(rule?.matching?.fuzzy_threshold??90));
  loadRequestRuleActions(state.actionSelector,rule);
  renderResultAliases(panel,state.actionSelector);
  setFuzzyState(root,"rule");q("#rule-error").textContent="";
  dialog.showModal();
  refreshEditor(panel);
  panel._captureDialogBaseline?.(dialog);
}
