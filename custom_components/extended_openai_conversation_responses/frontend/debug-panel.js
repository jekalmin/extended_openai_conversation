import {DEBUG_PROVIDER_PAGE_LIMIT, debugPageText, providerPageLabel, providerPageMeta, sessionLabel} from "./debug-management.js";

const WS_TYPE = "extended_openai_conversation_responses/request_debug";

export class ExtendedOpenAIDebugPanel extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({mode: "open"});
    this._hass = null;
    this._agents = [];
    this._agent = null;
    this._status = null;
    this._runs = [];
    this._loading = false;
    this._error = "";
    this._toastTimer = null;
  }

  set hass(value) {
    const first = !this._hass;
    this._hass = value;
    if (first) this._loadAgents();
  }

  _e(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  async _call(action, extra = {}) {
    return await this._hass.callWS({
      type: WS_TYPE,
      action,
      ...(this._agent ? {
        entry_id: this._agent.entry_id,
        subentry_id: this._agent.subentry_id,
      } : {}),
      ...extra,
    });
  }

  async _loadAgents() {
    this._loading = true;
    this._render();
    try {
      const result = await this._call("agents");
      this._agents = result.agents || [];
      const saved = (this.hasAttribute("embedded") && this._managementAgentId)
        || localStorage.getItem("extended-openai-debug-agent");
      this._agent = this._agents.find((item) => item.subentry_id === saved)
        || this._agents[0]
        || null;
      if (this._agent) {
        localStorage.setItem("extended-openai-debug-agent", this._agent.subentry_id);
        await this._loadRuns();
      }
      this._error = "";
    } catch (err) {
      this._error = err.message || String(err);
    } finally {
      this._loading = false;
      this._render();
    }
  }

  async _loadRuns() {
    if (!this._agent) return;
    const result = await this._call("runs");
    this._status = result;
    this._runs = result.runs || [];
  }

  async _selectAgent(subentryId) {
    this._agent = this._agents.find((item) => item.subentry_id === subentryId) || null;
    if (!this._agent) return;
    localStorage.setItem("extended-openai-debug-agent", this._agent.subentry_id);
    this._loading = true;
    this._render();
    try {
      await this._loadRuns();
      this._error = "";
    } catch (err) {
      this._error = err.message || String(err);
    } finally {
      this._loading = false;
      this._render();
    }
  }

  async _configure(extra) {
    try {
      this._status = await this._call("configure", extra);
      await this._loadRuns();
      this._render();
      this._toast(this._status.enabled ? "Request debugging enabled" : "Request debugging disabled");
    } catch (err) {
      this._toast(`Unable to update debugging: ${err.message || String(err)}`, true);
    }
  }

  async _clear() {
    if (!confirm("Clear all captured debug runs for this agent?")) return;
    try {
      await this._call("clear", {confirm: true});
      await this._loadRuns();
      this._render();
      this._toast("Debug captures cleared");
    } catch (err) {
      this._toast(`Unable to clear debug captures: ${err.message || String(err)}`, true);
    }
  }

  async _getRun(debugId, providerOffset = 0) {
    return await this._call("get", {
      debug_id: debugId,
      ...(this.hasAttribute("embedded") ? {
        provider_offset: Math.max(0, Number(providerOffset) || 0),
        provider_limit: DEBUG_PROVIDER_PAGE_LIMIT,
      } : {}),
    });
  }

  async _viewRun(debugId, providerOffset = 0) {
    const dialog = this.shadowRoot.querySelector("#debug-dialog");
    const title = this.shadowRoot.querySelector("#debug-dialog-title");
    const body = this.shadowRoot.querySelector("#debug-json");
    const copy = this.shadowRoot.querySelector("#copy-debug-log");
    const previous = this.shadowRoot.querySelector("#debug-provider-previous");
    const next = this.shadowRoot.querySelector("#debug-provider-next");
    const status = this.shadowRoot.querySelector("#debug-provider-status");
    const token = (this._eocDebugRunLoadToken || 0) + 1;
    this._eocDebugRunLoadToken = token;
    this._eocDebugId = debugId;
    title.textContent = "Loading debug run…";
    body.textContent = "Loading…";
    copy.disabled = true;
    copy.dataset.text = "";
    if (previous) previous.disabled = true;
    if (next) next.disabled = true;
    if (!dialog.open) dialog.showModal();
    try {
      const result = await this._getRun(debugId, providerOffset);
      if (this._eocDebugRunLoadToken !== token || !dialog.open) return;
      const trace = result.trace || {};
      const meta = providerPageMeta(trace);
      const text = this.hasAttribute("embedded") ? debugPageText(trace) : result.copy_text;
      this._eocDebugProviderOffset = Math.max(0, Number(meta.offset) || 0);
      this._eocDebugProviderMeta = meta;
      title.textContent = `Debug run ${debugId.slice(0, 8)}`;
      body.textContent = text;
      copy.dataset.text = text;
      copy.disabled = false;
      if (status) {
        status.textContent = `${providerPageLabel(trace)}${trace.management_projection?.truncated ? " · bounded/truncated management view" : ""}`;
      }
      if (previous) {
        previous.disabled = this._eocDebugProviderOffset === 0;
        previous.dataset.offset = String(Math.max(0, this._eocDebugProviderOffset - DEBUG_PROVIDER_PAGE_LIMIT));
      }
      if (next) {
        next.disabled = !meta.has_more;
        next.dataset.offset = String(meta.next_offset ?? (this._eocDebugProviderOffset + (Number(meta.returned) || 0)));
      }
    } catch (err) {
      if (this._eocDebugRunLoadToken !== token || !dialog.open) return;
      title.textContent = "Unable to load debug run";
      body.textContent = err.message || String(err);
      if (status) status.textContent = "";
    }
  }

  async _copyRun(debugId) {
    try {
      const result = await this._getRun(debugId, 0);
      const embedded = this.hasAttribute("embedded");
      await this._copyText(embedded ? debugPageText(result.trace) : result.copy_text);
      this._toast(embedded ? "First debug page copied" : "Entire debug log copied");
    } catch (err) {
      this._toast(`Unable to copy debug log: ${err.message || String(err)}`, true);
    }
  }

  async _copyText(text) {
    if (navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(text);
        return;
      } catch (_) {
        // Fall back for browsers/contexts that deny clipboard API access.
      }
    }
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    try {
      textarea.select();
      if (!document.execCommand("copy")) throw new Error("Clipboard access failed");
    } finally {
      textarea.remove();
    }
  }

  _formatDate(value) {
    if (!value) return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    try {
      return new Intl.DateTimeFormat(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        timeZone: this._hass?.config?.time_zone,
      }).format(date);
    } catch (_) {
      return date.toLocaleString();
    }
  }

  _number(value) {
    const number = Number(value);
    return Number.isFinite(number) ? Math.trunc(number).toLocaleString() : "—";
  }

  _duration(value) {
    const ms = Number(value);
    if (!Number.isFinite(ms)) return "—";
    return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(ms >= 10000 ? 1 : 2)} s`;
  }

  _continuityLabel(run) {
    if (this.hasAttribute("embedded")) return sessionLabel(run);
    if (run.continuity_resumed === true) return "Restored";
    if (run.incoming_conversation_id
      && run.resolved_conversation_id
      && run.incoming_conversation_id === run.resolved_conversation_id) {
      return "Continuing";
    }
    return run.resolved_conversation_id ? "New" : "—";
  }

  _toast(message, error = false) {
    const toast = this.shadowRoot.querySelector("#toast");
    if (!toast) return;
    toast.textContent = message;
    toast.className = `toast visible${error ? " error" : ""}`;
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => { toast.className = "toast"; }, 3500);
  }

  _styles() {
    return `<style>
      :host{display:block;color:var(--primary-text-color);background:var(--primary-background-color);min-height:100vh;box-sizing:border-box}
      :host([embedded]){min-height:0;background:transparent}
      :host([embedded]) main{max-width:none;padding:0;min-height:0}
      [hidden]{display:none!important}
      #debug-provider-status{margin-right:auto}
      *{box-sizing:border-box}main{max-width:1450px;margin:0 auto;padding:24px;display:grid;gap:18px}
      h1,h2,p{margin:0}.heading{display:flex;justify-content:space-between;gap:20px;align-items:end}.heading p{margin-top:6px;color:var(--secondary-text-color);line-height:1.5}.page-intro{display:grid;gap:7px;max-width:780px}.page-intro h1,.page-intro p{margin:0}.page-intro p{color:var(--secondary-text-color);line-height:1.5}.debug-intro{display:flex;align-items:end;justify-content:space-between;gap:20px}
      .card-heading{display:flex;align-items:start;justify-content:space-between;gap:24px;margin:0 0 20px}.card-heading h2{margin:0;font-size:19px;font-weight:600;line-height:1.3}.card-heading p{margin:6px 0 0;color:var(--secondary-text-color);line-height:1.5}.card-heading + .table-wrap{margin-top:0}.card{background:var(--card-background-color);border:1px solid var(--divider-color);border-radius:12px;padding:24px}.warning{border-left:4px solid var(--warning-color,#ff9800)}
      .warning p{margin-top:7px;line-height:1.55}.controls{display:grid;grid-template-columns:minmax(220px,1fr) auto auto auto;gap:14px;align-items:end}
      label{display:grid;gap:7px;font-size:13px;color:var(--secondary-text-color)}select,button{font:inherit;min-height:42px;border-radius:8px;border:1px solid var(--divider-color);background:var(--card-background-color);color:var(--primary-text-color);padding:8px 12px}
      button{cursor:pointer;color:var(--primary-color);font-weight:600}button.primary{background:var(--primary-color);color:var(--text-primary-color,#fff);border-color:var(--primary-color)}button.danger{color:var(--error-color)}button:disabled{opacity:.55;cursor:default}
      .switch-row{display:flex;align-items:center;gap:10px;min-height:42px}.switch-row input{width:20px;height:20px}.status{font-size:13px;color:var(--secondary-text-color);margin-top:12px}
      .table-wrap{overflow:auto;margin-top:14px}table{width:100%;border-collapse:collapse;min-width:1050px}th,td{text-align:left;padding:11px 10px;border-bottom:1px solid var(--divider-color);vertical-align:middle}th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--secondary-text-color)}td{font-size:14px}.mono{font-family:var(--code-font-family,monospace);font-size:12px}.actions{display:flex;gap:7px;white-space:nowrap}.actions button{min-height:34px;padding:5px 9px;font-size:13px}
      .empty{padding:30px;text-align:center;color:var(--secondary-text-color)}.error-box{padding:13px;border-radius:8px;background:color-mix(in srgb,var(--error-color) 12%,transparent);color:var(--error-color)}
      dialog{width:min(1200px,94vw);height:min(850px,90vh);border:1px solid var(--divider-color);border-radius:12px;background:var(--card-background-color);color:var(--primary-text-color);padding:0}dialog::backdrop{background:rgba(0,0,0,.45)}.dialog-head,.dialog-foot{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:15px 18px;border-bottom:1px solid var(--divider-color)}.dialog-foot{border-top:1px solid var(--divider-color);border-bottom:0;justify-content:flex-end}.dialog-body{height:calc(100% - 130px);overflow:auto;padding:0}pre{margin:0;padding:18px;white-space:pre-wrap;overflow-wrap:anywhere;font-family:var(--code-font-family,monospace);font-size:12px;line-height:1.5}
      .toast{position:fixed;right:24px;bottom:24px;z-index:10000;background:var(--primary-text-color);color:var(--primary-background-color);padding:11px 15px;border-radius:8px;opacity:0;pointer-events:none;transform:translateY(8px);transition:.15s}.toast.visible{opacity:1;transform:none}.toast.error{background:var(--error-color);color:white}
      @media(max-width:800px){main{padding:14px}.heading{display:grid}.controls{grid-template-columns:1fr 1fr}.controls label:first-child{grid-column:1/-1}}
    </style>`;
  }

  _render() {
    if (!this.shadowRoot) return;
    const embedded = this.hasAttribute("embedded");
    const status = this._status || {enabled:false,limit:10,count:0,allowed_limits:[5,10,25,50]};
    const agentOptions = this._agents.map((agent) => `<option value="${this._e(agent.subentry_id)}" ${agent.subentry_id === this._agent?.subentry_id ? "selected" : ""}>${this._e(agent.title)}</option>`).join("");
    const rows = this._runs.map((run) => `<tr>
      <td>${this._e(this._formatDate(run.completed_at || run.started_at))}</td>
      <td>${this._e(this._duration(run.duration_ms))}</td>
      <td>${this._number(run.input_tokens)}</td>
      <td>${this._number(run.cached_input_tokens)}</td>
      <td>${this._number(run.provider_request_count)}</td>
      <td>${run.first_text_ms == null ? "—" : this._e(this._duration(run.first_text_ms))}</td>
      <td class="mono" title="${this._e(run.resolved_conversation_id || "")}">${this._e((run.resolved_conversation_id || "—").slice(0,22))}${(run.resolved_conversation_id || "").length > 22 ? "…" : ""}</td>
      <td>${this._e(this._continuityLabel(run))}</td>
      <td>${run.successful === false ? this._e(run.error_type || "Failed") : "Success"}</td>
      <td><div class="actions"><button data-view="${this._e(run.debug_id)}">View</button><button data-copy="${this._e(run.debug_id)}">${embedded ? "Copy first page" : "Copy entire log"}</button></div></td>
    </tr>`).join("");

    this.shadowRoot.innerHTML = `${this._styles()}<main>
      <div class="debug-intro"><section class="page-intro"><h1>Request debugging</h1><p>Inspect exactly what Extended OpenAI assembled and how long each provider request took.</p></section><button id="refresh">Refresh</button></div>
      <section class="card warning"><strong>Debug captures can contain private data</strong><p>When enabled, Extended OpenAI keeps complete recent request material in memory, including the effective system prompt, conversation input, entity states, retrieved memories, tool schemas, provider events and tool results carried into later requests. Captures are bounded and disappear when Home Assistant restarts. Do not share a copied log without reviewing it first.</p></section>
      ${this._error ? `<div class="error-box">${this._e(this._error)}</div>` : ""}
      <section class="card"><div class="controls">
        <label ${embedded ? "hidden" : ""}>Conversation agent<select id="agent" ${this._loading ? "disabled" : ""}>${agentOptions}</select></label>
        <label>Capture limit<select id="limit" ${!this._agent ? "disabled" : ""}>${(status.allowed_limits || [5,10,25,50]).map((value) => `<option value="${value}" ${Number(status.limit) === Number(value) ? "selected" : ""}>${value} runs</option>`).join("")}</select></label>
        <label>Request debugging<span class="switch-row"><input id="enabled" type="checkbox" ${status.enabled ? "checked" : ""} ${!this._agent ? "disabled" : ""}><span>${status.enabled ? "Enabled" : "Disabled"}</span></span></label>
        <button id="clear" class="danger" ${!this._agent || !status.count ? "disabled" : ""}>Clear captures</button>
      </div><p class="status">${status.enabled ? `Capturing the next requests · ${this._number(status.count)} of ${this._number(status.limit)} slots currently used.` : "Capture is off. Normal usage history remains content-free."}</p></section>
      <section class="card"><div class="card-heading"><div><h2>Recent debug runs</h2><p>${embedded ? "Times are measured locally. First text is relative to provider request dispatch. Session handling describes how this run resolved conversation history. Prompt-cache hits can be shared across separate sessions and do not imply shared conversation history." : "Times are measured locally. First text is relative to provider request dispatch. Continuity shows whether the request started a new conversation, continued the incoming Home Assistant conversation, or restored integration-managed continuity."}</p></div></div><div class="table-wrap"><table><thead><tr><th>Completed</th><th>Run</th><th>Input</th><th>Cached</th><th>Requests</th><th>First text</th><th>Conversation ID</th><th>${embedded ? "Session handling" : "Continuity"}</th><th>Result</th><th>Debug log</th></tr></thead><tbody>${rows}</tbody></table>${rows ? "" : `<div class="empty">${status.enabled ? "No debug runs captured yet." : "Enable request debugging to capture future runs."}</div>`}</div></section>
    </main>
    <dialog id="debug-dialog"><div class="dialog-head"><h2 id="debug-dialog-title">Debug run</h2><button id="close-debug" aria-label="Close">Close</button></div><div class="dialog-body"><pre id="debug-json"></pre></div><div class="dialog-foot">${embedded ? `<span id="debug-provider-status" class="status"></span><button id="debug-provider-previous" disabled>Previous requests</button><button id="debug-provider-next" disabled>Next requests</button>` : ""}<button id="copy-debug-log" class="primary" disabled>${embedded ? "Copy visible page" : "Copy entire debug log"}</button></div></dialog>
    <div id="toast" class="toast"></div>`;

    this.shadowRoot.querySelector("#agent")?.addEventListener("change", (event) => this._selectAgent(event.target.value));
    this.shadowRoot.querySelector("#enabled")?.addEventListener("change", (event) => this._configure({enabled:event.target.checked}));
    this.shadowRoot.querySelector("#limit")?.addEventListener("change", (event) => this._configure({limit:Number(event.target.value)}));
    this.shadowRoot.querySelector("#clear")?.addEventListener("click", () => this._clear());
    this.shadowRoot.querySelector("#refresh")?.addEventListener("click", async () => { try { await this._loadRuns(); this._render(); } catch (err) { this._toast(err.message || String(err), true); } });
    this.shadowRoot.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => this._viewRun(button.dataset.view)));
    this.shadowRoot.querySelectorAll("[data-copy]").forEach((button) => button.addEventListener("click", () => this._copyRun(button.dataset.copy)));
    this.shadowRoot.querySelector("#close-debug")?.addEventListener("click", () => this.shadowRoot.querySelector("#debug-dialog")?.close());
    for (const direction of ["previous", "next"]) {
      const button = this.shadowRoot.querySelector(`#debug-provider-${direction}`);
      button?.addEventListener("click", () => {
        if (this._eocDebugId) void this._viewRun(this._eocDebugId, Number(button.dataset.offset) || 0);
      });
    }
    this.shadowRoot.querySelector("#copy-debug-log")?.addEventListener("click", async (event) => {
      try { await this._copyText(event.currentTarget.dataset.text || ""); this._toast(embedded ? "Visible debug page copied" : "Entire debug log copied"); }
      catch (err) { this._toast(`Unable to copy debug log: ${err.message || String(err)}`, true); }
    });
  }
}

if (!customElements.get("extended-openai-debug-panel")) {
  customElements.define("extended-openai-debug-panel", ExtendedOpenAIDebugPanel);
}
