import {tokenCount, tokenBreakdown, formatUsageNumber, formatUsageTimestamp} from "./usage-format.js";
import {loadUsageWindow} from "./usage-data.js";
export {loadAllUsageDays} from "./usage-data.js";
export {tokenBreakdown, formatUsageNumber, formatUsageTimestamp} from "./usage-format.js";

const USAGE_WINDOW_OPTIONS = [
  {id: "7", label: "7 days"},
  {id: "30", label: "30 days"},
  {id: "90", label: "90 days"},
  {id: "year", label: "Year to date"},
  {id: "all", label: "All available"},
];
const DEFAULT_USAGE_WINDOW = "30";
const DATE_KEY = /^(\d{4})-(\d{2})-(\d{2})$/;
const usageDateKeyFormatters = new Map();
const usageDisplayFormatters = new Map();

function cachedDateTimeFormat(cache, locales, options) {
  const key = JSON.stringify([locales || null, options]);
  let formatter = cache.get(key);
  if (!formatter) {
    formatter = new Intl.DateTimeFormat(locales, options);
    cache.set(key, formatter);
  }
  return formatter;
}

function normalizeUsageWindow(value) {
  const key = String(value || "");
  return USAGE_WINDOW_OPTIONS.some((item) => item.id === key) ? key : DEFAULT_USAGE_WINDOW;
}

function dateParts(value) {
  const match = DATE_KEY.exec(String(value || ""));
  if (!match) return null;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const date = new Date(Date.UTC(year, month - 1, day));
  if (
    date.getUTCFullYear() !== year ||
    date.getUTCMonth() !== month - 1 ||
    date.getUTCDate() !== day
  ) return null;
  return {year, month, day};
}

function dateKeyFromUtc(date) {
  return `${String(date.getUTCFullYear()).padStart(4, "0")}-${String(date.getUTCMonth() + 1).padStart(2, "0")}-${String(date.getUTCDate()).padStart(2, "0")}`;
}

export function addUsageCalendarDays(value, amount) {
  const parts = dateParts(value);
  if (!parts || !Number.isInteger(amount)) return null;
  const date = new Date(Date.UTC(parts.year, parts.month - 1, parts.day));
  date.setUTCDate(date.getUTCDate() + amount);
  return dateKeyFromUtc(date);
}

export function localUsageDateKey(value = new Date(), timeZone = undefined) {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  try {
    const parts = Object.fromEntries(
      cachedDateTimeFormat(usageDateKeyFormatters, "en-US", {
        year: "numeric", month: "2-digit", day: "2-digit", timeZone,
      }).formatToParts(date).filter((item) => item.type !== "literal").map((item) => [item.type, item.value]),
    );
    return `${parts.year}-${parts.month}-${parts.day}`;
  } catch (_) {
    return date.toISOString().slice(0, 10);
  }
}

export function usageWindowBounds(window, today) {
  const id = normalizeUsageWindow(window);
  const validToday = dateParts(today) ? today : localUsageDateKey();
  const label = USAGE_WINDOW_OPTIONS.find((item) => item.id === id)?.label || "30 days";
  if (id === "all") return {id, label, startDate: null, endDate: validToday};
  if (id === "year") return {id, label, startDate: `${validToday.slice(0, 4)}-01-01`, endDate: validToday};
  const days = Number(id);
  return {id, label, startDate: addUsageCalendarDays(validToday, -(days - 1)), endDate: validToday};
}

function formatUsageDate(value, locales = undefined, monthOnly = false) {
  const parts = dateParts(monthOnly ? `${value}-01` : value);
  if (!parts) return String(value || "");
  const date = new Date(Date.UTC(parts.year, parts.month - 1, parts.day, 12));
  try {
    const options = monthOnly
      ? {year: "numeric", month: "short", timeZone: "UTC"}
      : {year: "numeric", month: "short", day: "numeric", timeZone: "UTC"};
    return cachedDateTimeFormat(usageDisplayFormatters, locales, options).format(date);
  } catch (_) {
    return String(value || "");
  }
}

const diagnosticCounters = [
  "run_count", "successful_run_count", "failed_run_count", "api_request_count",
  "successful_request_count", "failed_request_count", "input_tokens", "output_tokens",
  "total_tokens", "cached_input_tokens", "reasoning_tokens", "tool_call_count",
  "web_search_run_count", "total_run_duration_ms",
];

function mergeBreakdown(target, source) {
  if (!source || typeof source !== "object") return;
  for (const [name, value] of Object.entries(source)) {
    target[name] = (target[name] || 0) + tokenCount(value);
  }
}

export function summarizeUsageDiagnostics(days = []) {
  const summary = Object.fromEntries(diagnosticCounters.map((key) => [key, 0]));
  summary.provider_breakdown = {};
  summary.model_breakdown = {};
  summary.api_mode_breakdown = {};
  for (const day of Array.isArray(days) ? days : []) {
    for (const key of diagnosticCounters) summary[key] += tokenCount(day?.[key]);
    mergeBreakdown(summary.provider_breakdown, day?.provider_breakdown);
    mergeBreakdown(summary.model_breakdown, day?.model_breakdown);
    mergeBreakdown(summary.api_mode_breakdown, day?.api_mode_breakdown);
  }
  summary.cache_percent = summary.input_tokens ? Math.min(100, summary.cached_input_tokens / summary.input_tokens * 100) : null;
  summary.run_success_percent = summary.run_count ? summary.successful_run_count / summary.run_count * 100 : null;
  summary.request_success_percent = summary.api_request_count ? summary.successful_request_count / summary.api_request_count * 100 : null;
  summary.average_tokens_per_run = summary.run_count ? summary.total_tokens / summary.run_count : 0;
  summary.average_requests_per_run = summary.run_count ? summary.api_request_count / summary.run_count : 0;
  summary.average_duration_ms = summary.run_count ? summary.total_run_duration_ms / summary.run_count : 0;
  return summary;
}

export function selectUsageHistory(days = [], window = DEFAULT_USAGE_WINDOW, today = localUsageDateKey()) {
  const bounds = usageWindowBounds(window, today);
  const byDate = new Map();
  for (const day of Array.isArray(days) ? days : []) {
    const date = String(day?.date || "");
    if (!dateParts(date) || date > bounds.endDate) continue;
    byDate.set(date, day);
  }
  const availableDays = [...byDate.values()].sort((left, right) => String(left.date).localeCompare(String(right.date)));
  const selectedDays = bounds.startDate
    ? availableDays.filter((day) => day.date >= bounds.startDate && day.date <= bounds.endDate)
    : availableDays;
  return {
    ...bounds,
    days: selectedDays,
    summary: summarizeUsageDiagnostics(selectedDays),
    allSummary: summarizeUsageDiagnostics(availableDays),
    availableStart: availableDays[0]?.date || null,
    availableEnd: availableDays.at(-1)?.date || null,
    partialStart: Boolean(bounds.startDate && availableDays[0]?.date && availableDays[0].date > bounds.startDate),
  };
}

export function usageLifetimeDiffersFromDaily(lifetime = {}, dailySummary = {}) {
  return [
    ["conversation_count", "run_count"],
    ["api_request_count", "api_request_count"],
    ["successful_request_count", "successful_request_count"],
    ["failed_request_count", "failed_request_count"],
    ["input_tokens", "input_tokens"],
    ["output_tokens", "output_tokens"],
    ["total_tokens", "total_tokens"],
    ["cached_input_tokens", "cached_input_tokens"],
    ["reasoning_tokens", "reasoning_tokens"],
  ].some(([lifetimeKey, dailyKey]) => tokenCount(lifetime?.[lifetimeKey]) !== tokenCount(dailySummary?.[dailyKey]));
}

export function usageChartBuckets(days = [], window = DEFAULT_USAGE_WINDOW) {
  const id = normalizeUsageWindow(window);
  if (!["year", "all"].includes(id)) {
    return (Array.isArray(days) ? days : []).map((day) => ({
      label: String(day?.date || ""),
      firstDate: String(day?.date || ""),
      lastDate: String(day?.date || ""),
      total_tokens: tokenCount(day?.total_tokens),
      cached_input_tokens: tokenCount(day?.cached_input_tokens),
    }));
  }
  const buckets = new Map();
  for (const day of Array.isArray(days) ? days : []) {
    const date = String(day?.date || "");
    if (!dateParts(date)) continue;
    const key = date.slice(0, 7);
    const bucket = buckets.get(key) || {
      label: key,
      firstDate: date,
      lastDate: date,
      total_tokens: 0,
      cached_input_tokens: 0,
    };
    bucket.lastDate = date;
    bucket.total_tokens += tokenCount(day?.total_tokens);
    bucket.cached_input_tokens += tokenCount(day?.cached_input_tokens);
    buckets.set(key, bucket);
  }
  return [...buckets.values()];
}

export function sortedUsageBreakdown(values = {}) {
  return Object.entries(values || {})
    .map(([name, tokens]) => ({name, tokens: tokenCount(tokens)}))
    .sort((a, b) => b.tokens - a.tokens || a.name.localeCompare(b.name));
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${value.toFixed(value >= 99.95 ? 0 : 1)}%` : "—";
}

function formatDuration(value) {
  const milliseconds = Number(value) || 0;
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  const seconds = milliseconds / 1000;
  return seconds < 60 ? `${seconds.toFixed(seconds >= 10 ? 1 : 2)} s` : `${(seconds / 60).toFixed(1)} min`;
}

function diagnosticMetric(panel, title, value, detail = "") {
  return `<article class="usage-diagnostic-metric"><span>${panel._e(title)}</span><strong>${panel._e(value)}</strong>${detail ? `<small>${panel._e(detail)}</small>` : ""}</article>`;
}

function breakdownList(panel, title, values) {
  const rows = sortedUsageBreakdown(values);
  const total = rows.reduce((sum, row) => sum + row.tokens, 0);
  return `<section class="usage-breakdown"><h3>${panel._e(title)}</h3>${rows.length ? `<div class="usage-breakdown-list">${rows.map((row) => {
    const share = total ? row.tokens / total * 100 : 0;
    return `<div class="usage-breakdown-row"><span title="${panel._e(row.name)}">${panel._e(row.name)}</span><strong>${formatUsageNumber(row.tokens)}</strong><small>${panel._e(formatPercent(share))}</small></div>`;
  }).join("")}</div>` : `<p class="usage-diagnostic-empty">No recorded data in this period.</p>`}</section>`;
}

function resultToday(result, panel) {
  const configured = String(result?.summary?.today?.date || "");
  if (dateParts(configured)) return configured;
  const rows = result?.days?.days || [];
  const lastStored = String(rows.at(-1)?.date || "");
  if (dateParts(lastStored)) return lastStored;
  return localUsageDateKey(new Date(), panel?._hass?.config?.time_zone);
}

function historyRangeLabel(history) {
  if (history.id === "all") {
    return history.availableStart
      ? `${history.label} · ${history.availableStart} to ${history.availableEnd}`
      : `${history.label} · no recorded daily aggregates`;
  }
  return `${history.label} · ${history.startDate} to ${history.endDate}`;
}

export function renderUsageDiagnostics(panel, result = {}) {
  const history = selectUsageHistory(
    result.days?.days || [],
    panel._usageHistoryWindow || DEFAULT_USAGE_WINDOW,
    resultToday(result, panel),
  );
  const summary = history.summary;
  const recentRuns = result.runs?.runs || [];
  const failedRuns = recentRuns.filter((run) => run.successful === false).slice(0, 5);
  const recentLocalRuns = recentRuns.filter((run) => tokenCount(run.request_count) === 0).length;
  const cacheDetail = summary.input_tokens
    ? `${formatUsageNumber(summary.cached_input_tokens)} of ${formatUsageNumber(summary.input_tokens)} input tokens`
    : "No provider-reported input tokens";

  return `<style>
      .usage-diagnostics{display:grid;gap:22px}
      .usage-diagnostics-heading{display:flex;justify-content:space-between;align-items:start;gap:20px}
      .usage-diagnostics-heading h2,.usage-diagnostics-heading p{margin:0}.usage-diagnostics-heading p{margin-top:6px;color:var(--secondary-text-color);line-height:1.5}
      .usage-window{white-space:nowrap;color:var(--secondary-text-color);font-size:13px}
      .usage-diagnostic-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
      .usage-diagnostic-metric{display:grid;gap:5px;padding:15px;border:1px solid var(--divider-color);border-radius:10px;background:var(--secondary-background-color)}
      .usage-diagnostic-metric span,.usage-diagnostic-metric small{color:var(--secondary-text-color);font-size:13px}.usage-diagnostic-metric strong{font-size:20px;font-weight:600}
      .usage-diagnostic-columns{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
      .usage-diagnostic-panel{padding:18px;border:1px solid var(--divider-color);border-radius:11px}.usage-diagnostic-panel h3{margin:0 0 13px;font-size:16px}
      .usage-facts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.usage-fact{display:grid;gap:3px}.usage-fact span{color:var(--secondary-text-color);font-size:13px}.usage-fact strong{font-size:16px}
      .usage-breakdowns{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.usage-breakdown{min-width:0;padding:17px;border:1px solid var(--divider-color);border-radius:11px}.usage-breakdown h3{margin:0 0 12px;font-size:15px}
      .usage-breakdown-list{display:grid;gap:9px}.usage-breakdown-row{display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:9px;align-items:baseline}.usage-breakdown-row span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.usage-breakdown-row small{min-width:48px;text-align:right}.usage-diagnostic-empty{margin:0;color:var(--secondary-text-color);font-size:13px}
      .usage-failures{display:grid;gap:9px}.usage-failure{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;padding:11px 0;border-top:1px solid var(--divider-color)}.usage-failure:first-child{border-top:0;padding-top:0}.usage-failure p{margin:0}.usage-failure small{display:block;margin-top:4px}.usage-run-details{cursor:pointer;color:var(--primary-color);background:transparent;border:1px solid var(--primary-color);min-height:36px;padding:6px 10px}
      .usage-request-list{display:grid;gap:12px}.usage-request-card{padding:14px;border:1px solid var(--divider-color);border-radius:10px}.usage-request-card h3,.usage-request-card p{margin:0}.usage-request-card p{margin-top:6px;color:var(--secondary-text-color)}.usage-request-meta{display:flex;gap:8px 14px;flex-wrap:wrap;margin-top:9px;color:var(--secondary-text-color);font-size:13px}
      @media(max-width:950px){.usage-diagnostic-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.usage-breakdowns{grid-template-columns:1fr}}
      @media(max-width:680px){.usage-diagnostics-heading{display:grid}.usage-window{white-space:normal}.usage-diagnostic-grid,.usage-diagnostic-columns,.usage-facts{grid-template-columns:1fr}.usage-failure{grid-template-columns:1fr}.usage-run-details{width:100%}}
    </style>
    <section class="content-card usage-diagnostics" aria-label="Usage diagnostics">
      <div class="usage-diagnostics-heading"><div><h2>Usage diagnostics</h2><p>Use these figures to spot repeated provider calls, slow turns, failures, and how much input is being served from cache.</p></div><span class="usage-window">${panel._e(historyRangeLabel(history))}</span></div>
      <div class="usage-diagnostic-grid">
        ${diagnosticMetric(panel, "Cached input", formatPercent(summary.cache_percent), cacheDetail)}
        ${diagnosticMetric(panel, "Run success", formatPercent(summary.run_success_percent), `${formatUsageNumber(summary.failed_run_count)} failed run${summary.failed_run_count === 1 ? "" : "s"}`)}
        ${diagnosticMetric(panel, "Requests per run", summary.average_requests_per_run.toFixed(summary.average_requests_per_run >= 10 ? 1 : 2), `${formatUsageNumber(summary.api_request_count)} provider requests`)}
        ${diagnosticMetric(panel, "Average run time", formatDuration(summary.average_duration_ms), `${formatUsageNumber(summary.run_count)} completed runs`)}
      </div>
      <div class="usage-diagnostic-columns">
        <section class="usage-diagnostic-panel"><h3>Token mix</h3><div class="usage-facts">
          <div class="usage-fact"><span>Input</span><strong>${formatUsageNumber(summary.input_tokens)}</strong></div>
          <div class="usage-fact"><span>Output</span><strong>${formatUsageNumber(summary.output_tokens)}</strong></div>
          <div class="usage-fact"><span>Cached input</span><strong>${formatUsageNumber(summary.cached_input_tokens)}</strong></div>
          <div class="usage-fact"><span>Reasoning</span><strong>${formatUsageNumber(summary.reasoning_tokens)}</strong></div>
          <div class="usage-fact"><span>Average total / run</span><strong>${formatUsageNumber(Math.round(summary.average_tokens_per_run))}</strong></div>
          <div class="usage-fact"><span>Request success</span><strong>${panel._e(formatPercent(summary.request_success_percent))}</strong></div>
        </div></section>
        <section class="usage-diagnostic-panel"><h3>Activity</h3><div class="usage-facts">
          <div class="usage-fact"><span>Conversation runs</span><strong>${formatUsageNumber(summary.run_count)}</strong></div>
          <div class="usage-fact"><span>Provider requests</span><strong>${formatUsageNumber(summary.api_request_count)}</strong></div>
          <div class="usage-fact"><span>Tool calls</span><strong>${formatUsageNumber(summary.tool_call_count)}</strong></div>
          <div class="usage-fact"><span>Web-search runs</span><strong>${formatUsageNumber(summary.web_search_run_count)}</strong></div>
          <div class="usage-fact"><span>Retained zero-request runs</span><strong>${formatUsageNumber(recentLocalRuns)}</strong></div>
          <div class="usage-fact"><span>Failed provider requests</span><strong>${formatUsageNumber(summary.failed_request_count)}</strong></div>
        </div></section>
      </div>
      <div class="usage-breakdowns">
        ${breakdownList(panel, "Models", summary.model_breakdown)}
        ${breakdownList(panel, "Providers", summary.provider_breakdown)}
        ${breakdownList(panel, "API modes", summary.api_mode_breakdown)}
      </div>
      <section class="usage-diagnostic-panel"><h3>Recent failed runs (retained detail)</h3>${failedRuns.length ? `<div class="usage-failures">${failedRuns.map((run) => {
        const completed = formatUsageTimestamp(run.completed_at, undefined, panel._hass?.config?.time_zone);
        return `<div class="usage-failure"><div><p><strong>${panel._e(run.error_type || "Failed")}</strong></p><small>${panel._e(completed.display)} · ${formatUsageNumber(run.request_count)} request${run.request_count === 1 ? "" : "s"} · ${panel._e(formatDuration(run.duration_ms))}</small></div><button type="button" class="usage-run-details" data-usage-run-id="${panel._e(run.run_id)}">View requests</button></div>`;
      }).join("")}</div>` : `<p class="usage-diagnostic-empty">No failed runs are present in the retained recent-run details.</p>`}</section>
      <p class="help">Token, request, run, model, provider, and API-mode diagnostics all use the selected daily-aggregate period shown above. Retained recent-run details are separate and can be shorter than that period. Figures show provider-reported usage only; they do not estimate API cost or expose prompt, response, tool-argument, or reasoning content.</p>
    </section>`;
}

export function requestDetailsDialog() {
  return `<dialog id="usage-request-dialog" class="editor-dialog wide" aria-labelledby="usage-request-title"><div class="dialog-header"><h2 id="usage-request-title">Provider requests</h2><button type="button" class="icon close-usage-requests" aria-label="Close">×</button></div><div id="usage-request-body" class="dialog-body"></div><div class="dialog-actions"><button type="button" class="secondary close-usage-requests">Close</button></div></dialog>`;
}

function renderRequestDetails(panel, requests) {
  if (!requests.length) return `<div class="empty">No retained provider-request details are available for this run.</div>`;
  return `<div class="usage-request-list">${requests.map((request, index) => {
    const tokens = tokenBreakdown(request.total_tokens, request.cached_input_tokens);
    const timestamp = formatUsageTimestamp(request.timestamp, undefined, panel._hass?.config?.time_zone);
    return `<article class="usage-request-card"><h3>Request ${index + 1} · ${panel._e(request.successful ? "Success" : request.error_type || "Failed")}</h3><p>${panel._e(request.provider || "Unknown provider")} · ${panel._e(request.model || "Unknown model")} · ${panel._e(request.api_mode || "Unknown API mode")}</p><div class="usage-request-meta"><span>${panel._e(timestamp.display)}</span><span>${panel._e(request.request_stage || "other")}</span><span>${formatUsageNumber(tokens.total)} tokens</span><span>${formatUsageNumber(tokens.cached)} cached</span><span>${formatUsageNumber(request.reasoning_tokens || 0)} reasoning</span><span>${panel._e(formatDuration(request.duration_ms))}</span>${request.tool_calls_requested ? `<span>${formatUsageNumber(request.tool_calls_requested)} tool call${request.tool_calls_requested === 1 ? "" : "s"}</span>` : ""}${request.web_search_used ? `<span>Web search</span>` : ""}</div></article>`;
  }).join("")}</div>`;
}

function renderUsageBar(panel, bucket, max) {
  const {total, cached, uncached} = tokenBreakdown(bucket.total_tokens, bucket.cached_input_tokens);
  const height = Math.max(2, total / max * 100);
  const cachedShare = total ? cached / total * 100 : 0;
  const uncachedShare = total ? uncached / total * 100 : 0;
  const period = bucket.firstDate === bucket.lastDate ? bucket.firstDate : `${bucket.firstDate} to ${bucket.lastDate}`;
  const details = `${period} · ${formatUsageNumber(total)} total · ${formatUsageNumber(cached)} cached input · ${formatUsageNumber(uncached)} uncached`;
  return `<span class="chart-column" tabindex="0" aria-label="${panel._e(details)}" data-tooltip="${panel._e(details)}" style="height:${height}%"><span class="chart-segment cached" style="height:${cachedShare}%"></span><span class="chart-segment uncached" style="height:${uncachedShare}%"></span></span>`;
}

function usageWarnings(panel, result) {
  return (result.load_errors || []).map((issue) => `<div class="notice"><strong>${panel._e(issue.label)} unavailable</strong><p>${panel._e(issue.message)} Other usage information is still shown where available.</p></div>`).join("");
}

function usageRecentRows(panel, result) {
  const rows = (result.runs?.runs || []).map((run) => {
    const tokens = tokenBreakdown(run.total_tokens, run.cached_input_tokens);
    const completed = formatUsageTimestamp(run.completed_at, undefined, panel._hass?.config?.time_zone);
    return `<tr><td><time datetime="${panel._e(completed.datetime)}" title="${panel._e(completed.datetime)}">${panel._e(completed.display)}</time></td><td>${formatUsageNumber(tokens.total)}</td><td>${formatUsageNumber(tokens.cached)}</td><td>${formatUsageNumber(tokens.uncached)}</td><td>${formatUsageNumber(run.request_count)}</td><td>${panel._e(`${formatUsageNumber(run.duration_ms)} ms`)}</td><td>${panel._e(run.successful ? "Success" : run.error_type || "Failed")}</td></tr>`;
  }).join("");
  return result.loading?.runs ? `<tr><td colspan="7">Loading recent runs…</td></tr>`
    : rows || `<tr><td colspan="7">No retained recent runs.</td></tr>`;
}

export function reconcileUsageSecondary(panel, key) {
  if (panel._viewKey?.() !== "usage-maintenance/usage" || panel._busy) return false;
  const root = panel.shadowRoot;
  const warnings = root?.querySelector?.("[data-eoc-usage-warnings]");
  if (!warnings) return false;
  warnings.innerHTML = usageWarnings(panel, panel._result || {});
  if (key === "runs") {
    const diagnostics = root.querySelector("[data-eoc-usage-diagnostics]");
    const rows = root.querySelector("[data-eoc-usage-runs]");
    if (!diagnostics || !rows) return false;
    diagnostics.innerHTML = renderUsageDiagnostics(panel, panel._result || {});
    rows.innerHTML = usageRecentRows(panel, panel._result || {});
    bindUsageDiagnostics(panel);
  }
  return true;
}

export function renderUsagePage(panel, result = {}) {
  const today = resultToday(result, panel);
  const history = selectUsageHistory(result.days?.days || [], panel._usageHistoryWindow || DEFAULT_USAGE_WINDOW, today);
  const summary = history.summary;
  const lifetime = result.summary?.lifetime || {};
  const latest = result.summary?.latest || null;
  const buckets = usageChartBuckets(history.days, history.id);
  const chartMax = Math.max(1, ...buckets.map((bucket) => bucket.total_tokens));
  const chartByMonth = ["year", "all"].includes(history.id);
  const chartAxis = buckets.length ? `<div class="chart-axis" aria-hidden="true"><span>${panel._e(formatUsageDate(buckets[0].label, undefined, chartByMonth))}</span><span>${panel._e(formatUsageDate(buckets[Math.floor((buckets.length - 1) / 2)].label, undefined, chartByMonth))}</span><span>${panel._e(formatUsageDate(buckets.at(-1).label, undefined, chartByMonth))}</span></div>` : "";
  const cachedMeta = (value) => `${formatUsageNumber(value || 0)} cached input`;
  const completeHistory = result.days?.complete_history === true;
  const dailyMismatch = completeHistory && usageLifetimeDiffersFromDaily(lifetime, history.allSummary);
  const availableText = history.availableStart
    ? `${formatUsageDate(history.availableStart)} to ${formatUsageDate(history.availableEnd)}${completeHistory ? "" : " (loaded window)"}`
    : "No recorded daily aggregates yet";
  const selectedText = history.id === "all"
    ? availableText
    : `${formatUsageDate(history.startDate)} to ${formatUsageDate(history.endDate)}`;
  const gapText = !completeHistory
    ? " Older daily aggregates are loaded only when you select a wider history window."
    : dailyMismatch
      ? " Lifetime counters contain accounting that is not represented exactly by the stored daily aggregates; this can include usage recorded before daily aggregate history became available."
      : " Lifetime counters are stored separately from daily history even when their current totals agree.";
  const partialText = history.partialStart
    ? ` The first stored daily aggregate is ${formatUsageDate(history.availableStart)}, after the selected period begins.`
    : "";
  const options = USAGE_WINDOW_OPTIONS.map((item) => `<option value="${item.id}" ${item.id === history.id ? "selected" : ""}>${panel._e(item.label)}</option>`).join("");

  return `<style>
      .usage-range-card{display:flex;align-items:end;justify-content:space-between;gap:20px}.usage-range-copy{display:grid;gap:6px}.usage-range-copy h2,.usage-range-copy p{margin:0}.usage-range-copy p{color:var(--secondary-text-color)}.usage-range-control{min-width:190px}.usage-range-control select{width:100%;min-height:42px}.usage-history-note{line-height:1.5}.usage-history-note strong{display:block;margin-bottom:4px}
      @media(max-width:680px){.usage-range-card{display:grid}.usage-range-control{min-width:0;width:100%}}
    </style><div data-eoc-usage-warnings>${usageWarnings(panel, result)}</div>
    <section class="content-card usage-range-card"><div class="usage-range-copy"><h2>Usage period</h2><p>Totals, chart data, and model/provider/API-mode breakdowns use this same Home Assistant local-calendar period.</p><small>${panel._e(historyRangeLabel(history))}</small></div><label class="usage-range-control">History window<select id="usage-window">${options}</select></label></section>
    <section class="metric-grid compact">
      ${panel._metric(history.label, summary.total_tokens || 0, cachedMeta(summary.cached_input_tokens))}
      ${panel._metric("Input tokens", summary.input_tokens || 0, `${formatUsageNumber(summary.output_tokens || 0)} output`)}
      ${panel._metric("Provider requests", summary.api_request_count || 0, `${formatUsageNumber(summary.run_count || 0)} conversation runs`)}
      ${panel._metric("Lifetime tokens", lifetime.total_tokens || 0, "Separate cumulative counter")}
      ${panel._metric("Latest response", latest?.total_tokens ?? "—", latest ? cachedMeta(latest.cached_input_tokens) : "Retained run detail")}
    </section>
    <section class="content-card"><div class="card-heading"><h2>Tokens by ${chartByMonth ? "month" : "recorded day"}</h2><div class="chart-legend" aria-label="Token categories"><span><i class="legend-swatch uncached"></i>Uncached</span><span><i class="legend-swatch cached"></i>Cached input</span></div></div><div class="chart" aria-label="Token usage for the selected period; cached input tokens are included within each total">${buckets.map((bucket) => renderUsageBar(panel, bucket, chartMax)).join("") || panel._empty("No daily usage is recorded in this period.")}</div>${chartAxis}<p class="chart-note"><strong>Cached input</strong> is input recognised as cached by the provider. It is included in total tokens and may be billed at a lower rate.</p></section>
    <section class="notice usage-history-note"><strong>Daily aggregate history: ${panel._e(availableText)}</strong><p>The selected period is ${panel._e(selectedText)}.${panel._e(partialText)}${panel._e(gapText)} “All available” therefore means all stored daily aggregate history, not necessarily the same value as lifetime usage.</p></section>
    <div data-eoc-usage-diagnostics>${renderUsageDiagnostics(panel, result)}</div>
    <section class="content-card"><div class="card-heading"><div><h2>Recent runs</h2><p>This table uses retained run detail and is not expanded by the selected aggregate-history period.</p></div></div><div class="table"><table><thead><tr>${["Completed", "Total", "Cached input", "Uncached", "Requests", "Duration", "Result"].map((header) => `<th>${header}</th>`).join("")}</tr></thead><tbody data-eoc-usage-runs>${usageRecentRows(panel, result)}</tbody></table></div></section>
    ${panel._data?.is_admin ? `<section class="content-card"><div class="card-heading"><h2>Manage usage history</h2></div><div class="section-actions"><button type="button" class="secondary inline-route" data-page="usage-maintenance" data-subsection="retention">Configure retention</button><button type="button" id="clear-details" class="danger secondary-danger">Clear recent details</button></div><small>Daily, monthly, selected-period, and lifetime aggregates are never removed by detail pruning.</small></section>` : ""}`;
}

export function bindUsageDiagnostics(panel) {
  const root = panel.shadowRoot;
  const window = root.querySelector("#usage-window");
  if (window) window.onchange = async (event) => {
    const previous = normalizeUsageWindow(panel._usageHistoryWindow || DEFAULT_USAGE_WINDOW);
    const next = normalizeUsageWindow(event.target.value);
    if (next === previous) return;
    const agentId = panel._agentId;
    const generation = panel._usageWindowGeneration = (panel._usageWindowGeneration || 0) + 1;
    event.target.disabled = true;
    try {
      const days = await loadUsageWindow(panel, next, resultToday(panel._result || {}, panel));
      if (generation !== panel._usageWindowGeneration || panel._agentId !== agentId || panel._viewKey?.() !== "usage-maintenance/usage") return;
      panel._usageHistoryWindow = next;
      panel._result = {...(panel._result || {}), days};
      panel._render();
    } catch (err) {
      if (generation === panel._usageWindowGeneration && panel._agentId === agentId && panel._viewKey?.() === "usage-maintenance/usage") {
        event.target.disabled = false;
        event.target.value = previous;
        panel._toast?.(`Unable to load usage history: ${err?.message || String(err)}`, true);
      }
    }
  };
  root.querySelectorAll(".close-usage-requests").forEach((button) => {
    button.onclick = () => {
      panel._usageRequestGeneration = (panel._usageRequestGeneration || 0) + 1;
      root.querySelector("#usage-request-dialog")?.close();
    };
  });
  const dialog = root.querySelector("#usage-request-dialog");
  if (dialog) dialog.oncancel = (event) => {
    event.preventDefault();
    panel._usageRequestGeneration = (panel._usageRequestGeneration || 0) + 1;
    dialog.close();
  };
  root.querySelectorAll(".usage-run-details").forEach((button) => {
    button.onclick = async () => {
      const dialog = root.querySelector("#usage-request-dialog");
      const body = root.querySelector("#usage-request-body");
      if (!dialog || !body) return;
      const generation = panel._usageRequestGeneration = (panel._usageRequestGeneration || 0) + 1;
      body.innerHTML = panel._loading();
      dialog.showModal();
      try {
        const response = await panel._call("usage", "requests", {run_id: button.dataset.usageRunId, limit: 100});
        if (!dialog.open || generation !== panel._usageRequestGeneration) return;
        body.innerHTML = renderRequestDetails(panel, response?.requests || []);
      } catch (err) {
        if (dialog.open && generation === panel._usageRequestGeneration) body.innerHTML = `<div class="error" role="alert">${panel._e(err.message || String(err))}</div>`;
      }
    };
  });
}
