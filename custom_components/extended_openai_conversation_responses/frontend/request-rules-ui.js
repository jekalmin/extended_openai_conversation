const ROUTE_STYLE = `.rule-move-menu{position:relative;align-self:start}.rule-move-menu>summary{display:flex;align-items:center;justify-content:center;gap:8px;list-style:none;min-height:42px;padding:9px 16px;border-radius:9px;background:var(--secondary-background-color);color:var(--primary-text-color);cursor:pointer}.rule-move-menu>summary::-webkit-details-marker{display:none}.rule-move-menu>summary::after{content:"▾"}.rule-move-menu>summary:focus-visible{outline:2px solid var(--primary-color);outline-offset:2px}.rule-move-options{position:absolute;inset-inline-start:0;top:calc(100% + 4px);z-index:5;display:grid;gap:4px;min-width:170px;padding:6px;border:1px solid var(--divider-color);border-radius:9px;background:var(--card-background-color);box-shadow:0 3px 10px #0002}.rule-move-options button{text-align:start;white-space:nowrap}.rule-response-reference{grid-column:1/-1}.rule-list{display:grid;gap:14px}.request-rule-card{border:1px solid var(--divider-color);border-radius:14px;padding:18px;background:var(--card-background-color)}.request-rule-card.disabled{opacity:.62}.rule-card-heading{display:flex;justify-content:space-between;gap:16px;align-items:start}.rule-card-heading h2{font-size:18px;margin:6px 0 0}.type-badge{display:inline-flex;border-radius:999px;padding:4px 9px;font-size:11px;font-weight:700}.type-badge.local{color:var(--success-color,#0f9d58);background:color-mix(in srgb,var(--success-color,#0f9d58) 14%,transparent)}.type-badge.routing{color:var(--primary-color);background:color-mix(in srgb,var(--primary-color) 14%,transparent)}.phrase-chips{display:flex;flex-wrap:wrap;gap:7px;margin-top:14px}.phrase-chips span{padding:6px 9px;border-radius:8px;background:var(--secondary-background-color);font-size:13px}.phrase-chips b{font-weight:700;margin-right:4px}.sensitive-warning{color:var(--warning-color,#b26a00);font-weight:600}.request-rule-card>.actions{display:flex;gap:8px;justify-content:flex-end}.rule-settings details{padding-block:8px}.rule-settings details+details{border-top:1px solid var(--divider-color)}.rule-settings details>summary{font-weight:700;cursor:pointer}.matching-settings{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:14px}.fuzzy-sensitivity.is-disabled{opacity:.5}.wording-group{display:grid;grid-template-columns:1fr 2fr auto;gap:10px;align-items:end;margin-top:10px}.ha-action-row{display:grid;grid-template-columns:2fr 1fr 2fr auto;gap:10px;align-items:end;border:1px solid var(--divider-color);border-radius:10px;padding:12px;margin-bottom:10px}.ha-action-row label{margin:0}.ha-action-row ha-selector{display:block;min-width:0}.ha-service-fields{grid-column:1/-1;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.ha-action-advanced{grid-column:1/-1}.ha-action-advanced summary{cursor:pointer;font-weight:600}.ha-action-advanced textarea{min-height:140px;font-family:var(--code-font-family,ui-monospace,monospace)}.request-rule-dialog section{border-top:1px solid var(--divider-color);margin-top:20px;padding-top:16px}.request-rule-dialog h3{margin:0 0 8px}.request-rule-dialog .toggle{justify-content:flex-start;gap:10px}.request-rule-dialog .toggle input,.request-rule-dialog .matching-setting>input[type="checkbox"]{flex:0 0 20px;width:20px;height:20px;min-height:20px;padding:0;margin:0;align-self:flex-start}.request-rule-dialog #rule-local-responses{align-items:start}.request-rule-dialog #sentence-pattern-builder{margin-top:10px;gap:8px}.request-rule-dialog #sentence-pattern-builder .pattern-helper{min-height:34px;padding:5px 11px}.request-rule-dialog #sentence-pattern-help{margin-top:12px}.rule-list-heading{margin:20px 0 10px}.rule-list-heading h2{margin:0 0 3px;font-size:20px}.rule-list-heading .help{margin:0}.rule-filter-help{margin-top:6px!important}.rule-toolbar{flex-wrap:wrap}.rule-show-filter{display:inline-flex;align-items:center;gap:8px;white-space:nowrap}.rule-show-filter select{min-width:150px}.rule-list{gap:12px}.rule-group-chip{display:inline-flex;vertical-align:middle;border-radius:999px;padding:2px 8px;background:var(--secondary-background-color);font-size:11px;color:var(--secondary-text-color)}.request-rule-card[draggable="true"]{cursor:grab}.request-rule-card[draggable="true"]:active{cursor:grabbing}.rule-match-chain{padding-left:22px;margin:10px 0}.rule-match-chain li{margin:4px 0}.rule-group-section>summary{cursor:pointer;display:flex;align-items:baseline;justify-content:space-between;gap:12px;font-size:17px;font-weight:650}.rule-group-section>summary small{font-size:12px;font-weight:400;color:var(--secondary-text-color)}.rule-group-items{display:grid;gap:12px;margin-top:14px}.rule-group-items .request-rule-card{background:var(--secondary-background-color)}.rule-group-empty{color:var(--secondary-text-color);margin:0}.rule-global-order-note{margin:10px 0 4px}.rule-group-row{display:flex;gap:8px;align-items:center;margin:8px 0}.rule-group-row input{flex:1;min-width:0}.rule-group-section[hidden],.rule-group-empty[hidden]{display:none}.rule-sharing{margin-top:18px}.rule-sharing>summary{font-weight:650;cursor:pointer}.rule-sharing-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.rule-sharing-grid h3{margin:10px 0 4px}.rule-sharing-grid .help{margin:0 0 10px}.rule-sharing-grid button{margin-top:10px}.rule-pack-review{margin-top:16px}.rule-pack-review ul{padding-left:20px}.rule-pack-review li{margin:10px 0}#rule-pack-rule-list{display:grid;gap:8px}#rule-pack-rule-list label{display:block}@media(max-width:679px){.rule-sharing-grid{grid-template-columns:1fr}}@media(max-width:679px){.request-rule-card>.actions{display:grid;grid-template-columns:1fr 1fr}.matching-settings,.ha-service-fields{grid-template-columns:1fr}.wording-group{grid-template-columns:1fr auto}.ha-action-row{grid-template-columns:1fr}}`;
function ensureRouteStyle(panel) {
  if (typeof document === "undefined") return;
  if (panel.shadowRoot.querySelector("style[data-eoc-feature-style=\"request-rules\"]")) return;
  const style = document.createElement("style");
  style.dataset.eocFeatureStyle = "request-rules";
  style.textContent = ROUTE_STYLE;
  panel.shadowRoot.append(style);
}

import {
  applyRequestRuleMutation,
  bindRequestRulesCore,
  reconcileRequestRules,
  renderRequestRules,
  requestRulesDialog,
} from "./request-rules-ui-core.js";

let editorModule = null;
let editorPromise = null;
function ensureEditorModule() {
  if (editorModule) return Promise.resolve(editorModule);
  if (!editorPromise) {
    editorPromise = import("./request-rules-ui-impl.js")
      .then((module) => {
        editorModule = module;
        return module;
      })
      .finally(() => { editorPromise = null; });
  }
  return editorPromise;
}

let safeTesterModule = null;
let safeTesterPromise = null;
function ensureSafeTester(panel) {
  if (safeTesterModule) {
    safeTesterModule.bindRequestRuleMatchTester(panel);
    return Promise.resolve(safeTesterModule);
  }
  if (!safeTesterPromise) {
    safeTesterPromise = import("./request-rules-match-test-ui.js")
      .then((module) => {
        safeTesterModule = module;
        return module;
      })
      .finally(() => { safeTesterPromise = null; });
  }
  return safeTesterPromise.then((module) => {
    if (panel._viewKey?.() === "capabilities/request-rules") module.bindRequestRuleMatchTester(panel);
    return module;
  });
}

let liveTesterModule = null;
let liveTesterPromise = null;
function ensureLiveTester(panel) {
  if (liveTesterModule) {
    liveTesterModule.bindDecisionRequestRules(panel);
    return Promise.resolve(liveTesterModule);
  }
  if (!liveTesterPromise) {
    liveTesterPromise = import("./management-decision-guidance.js")
      .then((module) => {
        liveTesterModule = module;
        return module;
      })
      .finally(() => { liveTesterPromise = null; });
  }
  return liveTesterPromise.then((module) => {
    if (panel._viewKey?.() === "capabilities/request-rules") module.bindDecisionRequestRules(panel);
    return module;
  });
}

function warmEditorAfterPaint() {
  const warm = () => { void ensureEditorModule().catch(() => {}); };
  if (typeof globalThis.requestIdleCallback === "function") {
    globalThis.requestIdleCallback(warm, {timeout: 1500});
  } else {
    setTimeout(warm, 0);
  }
}

export function syncRequestRuleRoutingControls(root, efforts = null, selectedEffort = null) {
  const actionType = root?.querySelector?.("#rule-action-type");
  const matchType = root?.querySelector?.("#rule-match");
  const scope = root?.querySelector?.("#rule-scope");
  const continueToAi = root?.querySelector?.("#rule-continue-to-ai");
  const continueMatching = root?.querySelector?.("#rule-continue-matching");
  const reasoning = root?.querySelector?.("#rule-reasoning");
  const help = root?.querySelector?.("#rule-routing-scope-help");
  if (!actionType || !matchType || !scope) return;
  const consumed = actionType.value === "model_routing"
    && !(continueToAi?.checked ?? false)
    && !(continueMatching?.checked ?? false);
  const requestOption = scope.querySelector('option[value="request"]');
  if (requestOption) requestOption.disabled = consumed;
  if (consumed) scope.value = "conversation";
  scope.disabled = consumed;
  if (reasoning && Array.isArray(efforts)) {
    const desired = selectedEffort ?? reasoning.value;
    reasoning.replaceChildren(
      ...["", ...efforts].map((value) => {
        const option = reasoning.ownerDocument.createElement("option");
        option.value = value;
        option.textContent = value ? value.charAt(0).toUpperCase() + value.slice(1) : "Keep current";
        return option;
      }),
    );
    reasoning.value = desired && efforts.includes(desired) ? desired : "";
  }
  if (help) help.textContent = consumed
    ? "This is a complete routing command. It is acknowledged locally and is not sent to the AI provider, so it must change or reset the rest of this conversation."
    : continueToAi?.checked
      ? "This sends the original request to the AI unchanged after applying the route. This request only affects that provider call; Rest of this conversation also changes later requests."
      : "Later matching rules can run after this route. This request only affects a later AI handoff in this request; Rest of this conversation also changes later requests.";
}

export function bindRequestRules(panel) {
  ensureRouteStyle(panel);
  bindRequestRulesCore(panel, {
    openEditor: async (id) => {
      const module = await ensureEditorModule();
      if (panel._viewKey?.() === "capabilities/request-rules") module.openRequestRuleEditor(panel, id);
    },
    activateSafeTester: () => ensureSafeTester(panel),
    activateLiveTester: () => ensureLiveTester(panel),
  });
  warmEditorAfterPaint();
}

export async function formatRequestRuleMatchResult(panel, response) {
  const module = await import("./request-rules-match-test-ui.js");
  return module.formatRequestRuleMatchResult(panel, response);
}

export {
  applyRequestRuleMutation,
  reconcileRequestRules,
  renderRequestRules,
  requestRulesDialog,
};
