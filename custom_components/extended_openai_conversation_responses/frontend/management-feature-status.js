export {embeddedFeatureStatusMarkup} from "./management-feature-status-core.js";

export function selectedFeatureStatus(panel, featureName) {
  const sectionStatus = panel._result?.feature_status;
  if (sectionStatus && !sectionStatus[featureName]) return sectionStatus;
  return sectionStatus?.[featureName] || panel._selectedAgent?.()?.feature_status?.[featureName] || null;
}

export function featureStatusMarkup(panel, title, status, configureTarget = null) {
  if (!status) return "";
  const positive = ["enabled", "available"].includes(status.state);
  const configure = panel._data?.is_admin && configureTarget
    ? `<button type="button" class="secondary inline-route" data-page="${panel._e(configureTarget.page)}" data-subsection="${panel._e(configureTarget.subsection)}">${panel._e(configureTarget.label)}</button>`
    : "";
  return `<section class="content-card feature-status-card"><div class="compact-status"><span><strong>${panel._e(title)}</strong><small>${panel._e(status.detail || status.summary || "")}</small></span><strong class="status-value ${positive ? "on" : ""}">${panel._e(status.label || "Unknown")}</strong></div>${configure}</section>`;
}

export function overviewAgentFeatureProjection(agent) {
  const memory = agent?.feature_status?.memory;
  const knowledge = agent?.feature_status?.knowledge;
  if (!memory && !knowledge) return agent;
  const memoryLabel = memory?.label || agent.memory_mode || "Unknown";
  const labels = [memoryLabel];
  if (knowledge?.label) labels.push(`Knowledge ${knowledge.label}`);
  return {...agent, memory_mode: labels.join(" · ")};
}

export function diagnosticStatusMeta(status) {
  const normalized = String(status || "").toLowerCase();
  if (normalized === "passed") return {label: "Passed", icon: "✓", className: "passed"};
  if (normalized === "warning") return {label: "Warning", icon: "!", className: "warning"};
  if (normalized === "failed") return {label: "Failed", icon: "×", className: "failed"};
  return {label: status || "Unknown", icon: "?", className: "unknown"};
}

export function diagnosticResultMarkup(panel, result) {
  const overall = diagnosticStatusMeta(result?.status);
  const checks = Array.isArray(result?.checks) ? result.checks : [];
  const rows = checks.map((check) => {
    const status = diagnosticStatusMeta(check?.status);
    return `<div class="diagnostic-check ${status.className}"><span class="diagnostic-icon" aria-hidden="true">${panel._e(status.icon)}</span><span class="diagnostic-copy"><strong>${panel._e(check?.name || "Check")}</strong><small>${panel._e(check?.message || "")}</small></span><span class="diagnostic-status">${panel._e(status.label)}</span></div>`;
  }).join("");
  const raw = panel._e(JSON.stringify(result || {}, null, 2));
  return `<div class="diagnostic-summary ${overall.className}" role="status"><span class="diagnostic-icon" aria-hidden="true">${panel._e(overall.icon)}</span><span><strong>${panel._e(overall.label)}</strong><small>${overall.className === "passed" ? "All diagnostic checks passed." : overall.className === "warning" ? "The assistant is working, but one or more checks need attention." : overall.className === "failed" ? "One or more diagnostic checks failed." : "Diagnostic test completed."}</small></span></div><div class="diagnostic-checks">${rows || '<p class="meta">No individual checks were returned.</p>'}</div><details class="diagnostic-raw eoc-details-base"><summary>Show raw diagnostic data</summary><pre>${raw}</pre></details>`;
}

export function diagnosticsMarkup(panel, agent) {
  return `<style>${FEATURE_STATUS_STYLES}</style><section class="page-intro"><h1>Diagnostics</h1><p>Check this assistant's current setup and test its provider connection.</p></section><section class="metric-grid">${panel._metric("Agent", agent.title)}${panel._metric("Provider", agent.provider)}${panel._metric("Model", agent.model)}${panel._metric("Conversation archive", agent.archive_enabled ? "Enabled" : "Disabled")}${panel._metric("Guest Mode", panel._titleCase(String(agent.guest_mode?.state || "inactive").replaceAll("_", " ")))}</section><section class="content-card"><div class="card-heading"><div><h2>Test assistant</h2><p>Checks the assistant configuration and sends one minimal request to verify the selected provider and model. It does not run Home Assistant actions.</p></div></div><button type="button" id="test-agent">Run diagnostics</button><div id="test-result" class="diagnostic-result" aria-live="polite"></div><small>If a check fails, its result below will show what needs attention.</small></section>`;
}

export async function testAgent(panel) {
  const output = panel.shadowRoot?.querySelector("#test-result");
  const button = panel.shadowRoot?.querySelector("#test-agent");
  if (!output) return;
  if (button) button.disabled = true;
  panel.shadowRoot?.dispatchEvent(new CustomEvent("eoc-diagnostics-result", {detail: {result: null}}));
  output.innerHTML = '<div class="diagnostic-loading"><span class="spinner" aria-hidden="true"></span><span>Running diagnostic checks…</span></div>';
  try {
    const result = await panel._call("diagnostics", "test_agent");
    panel.shadowRoot?.dispatchEvent(new CustomEvent("eoc-diagnostics-result", {detail: {result}}));
    output.innerHTML = diagnosticResultMarkup(panel, result);
  } catch (err) {
    panel.shadowRoot?.dispatchEvent(new CustomEvent("eoc-diagnostics-result", {detail: {result: null}}));
    const message = err?.message || String(err);
    output.innerHTML = `<div class="diagnostic-summary failed" role="alert"><span class="diagnostic-icon" aria-hidden="true">×</span><span><strong>Unable to run diagnostics</strong><small>${panel._e(message)}</small></span></div>`;
  } finally {
    if (button) button.disabled = false;
  }
}

export const FEATURE_STATUS_STYLES = `
    .diagnostic-result{display:grid;gap:16px;margin:20px 0 14px}
    .diagnostic-loading{display:flex;align-items:center;gap:10px;min-height:52px;color:var(--secondary-text-color)}
    .diagnostic-summary,.diagnostic-check{display:grid;grid-template-columns:auto minmax(0,1fr) auto;align-items:center;gap:12px;border:1px solid var(--divider-color);border-radius:11px;padding:14px 16px}
    .diagnostic-summary{grid-template-columns:auto minmax(0,1fr);border-left:4px solid var(--divider-color);background:var(--secondary-background-color)}
    .diagnostic-summary.passed{border-left-color:var(--success-color,#0f9d58)}
    .diagnostic-summary.warning{border-left-color:var(--warning-color,#f9ab00)}
    .diagnostic-summary.failed{border-left-color:var(--error-color,#db4437)}
    .diagnostic-summary>span:last-child,.diagnostic-copy{display:grid;gap:3px;min-width:0}
    .diagnostic-summary small,.diagnostic-copy small{line-height:1.4;overflow-wrap:anywhere}
    .diagnostic-checks{display:grid;gap:8px}
    .diagnostic-check{padding:12px 14px}
    .diagnostic-icon{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:50%;font-weight:700;background:var(--secondary-background-color);color:var(--secondary-text-color)}
    .diagnostic-check.passed .diagnostic-icon,.diagnostic-summary.passed .diagnostic-icon{color:var(--success-color,#0f9d58)}
    .diagnostic-check.warning .diagnostic-icon,.diagnostic-summary.warning .diagnostic-icon{color:var(--warning-color,#f9ab00)}
    .diagnostic-check.failed .diagnostic-icon,.diagnostic-summary.failed .diagnostic-icon{color:var(--error-color,#db4437)}
    .diagnostic-status{font-size:12px;font-weight:600;color:var(--secondary-text-color)}
    .diagnostic-check.passed .diagnostic-status{color:var(--success-color,#0f9d58)}
    .diagnostic-check.warning .diagnostic-status{color:var(--warning-color,#f9ab00)}
    .diagnostic-check.failed .diagnostic-status{color:var(--error-color,#db4437)}
    .diagnostic-raw{margin-top:0;padding-top:12px}
    .diagnostic-raw pre{max-height:360px;overflow:auto;margin-bottom:0}
    @media(max-width:600px){.diagnostic-check{grid-template-columns:auto minmax(0,1fr)}.diagnostic-status{grid-column:2}}
  `;
