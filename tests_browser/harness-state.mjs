const KEY = "extended-openai-browser-harness-state-v3";
const clone = (value) => structuredClone(value);
const now = () => new Date().toISOString();

function freshState() {
  return {
    agent: {
      entry_id: "entry-1", subentry_id: "agent-1", title: "Jarvis", provider: "OpenAI",
      model: "gpt-5-mini", function_count: 1, function_group_count: 1, memory_mode: "Manual", temporary_memory: "off",
      memory_count: 1, knowledge_source_count: 0, archive_enabled: true,
      guest_mode: {state: "inactive", has_home_assistant_exclusions: true},
    },
    scopes: [{scope_id: "user:test-user", scope_type: "user", display_name: "Test User", is_current_user: true, memory_count: 1, conversation_count: 2}],
    configuration: {
      title: "Jarvis", revision: "fixture-7",
      config: {
        chat_model: "gpt-5-mini", api_mode: "responses", max_tokens: 1200, temporary_memory: "off",
        web_search_context: "medium",
        max_function_calls_per_conversation: 8, function_tool_error_recovery: true,
        continue_conversation: "never",
        usage_request_retention_days: 30, usage_run_retention_days: 30,
        functions: [{spec: {name: "baseline_tool", description: "Baseline browser fixture Function Tool", parameters: {type: "object", properties: {}}}, function: {type: "script", sequence: []}, enabled: true}],
        function_groups: [{id: "baseline-group", name: "Baseline group", description: "Baseline browser fixture Function Group", loading_mode: "on_demand", functions: ["baseline_tool"], guest_allowed: false, enabled: true}],
      },
      options: {
        api_mode: [{value: "responses", label: "Responses"}, {value: "chat_completions", label: "Chat Completions"}],
        continue_conversation: [{value: "never", label: "Never"}, {value: "always", label: "Always"}],
        reasoning_effort: [{value: "low", label: "Low"}, {value: "medium", label: "Medium"}, {value: "high", label: "High"}],
        web_search_context: [{value: "low", label: "Low"}, {value: "medium", label: "Medium"}, {value: "high", label: "High"}],
        memory_retrieval_mode: [{value: "lexical", label: "Lightweight lexical"}, {value: "hybrid", label: "Hybrid semantic"}],
        usage_request_retention_days: [{value: 7, label: "7 days"}, {value: 30, label: "30 days"}],
        usage_run_retention_days: [{value: 7, label: "7 days"}, {value: 30, label: "30 days"}],
      },
      defaults: {}, model_capabilities: {}, local_handling: {supported: true, intents: [], pipeline_conflicts: []},
    },
    memories: [{revision: 1, memory_id: "memory-1", scope_id: "user:test-user", content: "Baseline browser fixture memory", category: "general", source: "manual", created_at: "2026-09-01T12:00:00Z", updated_at: "2026-09-01T12:00:00Z"}],
    knowledgeSources: [],
    guest: {config: {guest_mode_enabled: false, guest_web_search: false}, revision: "guest-1", legacy_policy: false},
    quiet: {config: {enabled: false, start: "22:00", end: "07:00", max_volume: 0.2, wake_sound: "off", overrides: {}}, active: false, satellites: []},
    requestRules: {
      revision: 3, defaults: {word_forms: true, wording_alternatives: true, fuzzy: false, fuzzy_threshold: 90}, wording_groups: [], groups: [], diagnostics: {},
      rules: [{id: "rule-1", name: "Baseline rule", enabled: true, phrases: ["baseline route"], match_type: "contains", action_type: "model_routing", action: {model: "gpt-5-mini", reasoning_effort: "", scope: "request", reset: false, success_response: "Updated"}, matching_behavior: "defaults", matching: {word_forms: true, wording_alternatives: true, fuzzy: false, fuzzy_threshold: 90}, order: 0}],
    },
    requestDebug: {enabled: false, limit: 10, count: 0, runs: []},
    toolYamls: {baseline_tool: "spec:\n  name: baseline_tool\n  description: Baseline browser fixture Function Tool\n  parameters:\n    type: object\n    properties: {}\nfunction:\n  type: script\n  sequence: []\n"},
    nextMemoryId: 2, nextKnowledgeId: 1, nextRuleId: 2, nextRuleGroupId: 1, failedConfigurationOnce: false,
  };
}

function load() {
  try { const raw = localStorage.getItem(KEY); if (raw) return JSON.parse(raw); } catch (_err) {}
  const state = freshState(); localStorage.setItem(KEY, JSON.stringify(state)); return state;
}

export function createStateBackend({partialOverview = false, failConfigurationOnce = false, seedConversations = false, seedUsageBoundary = false} = {}) {
  let state = load();
  const pendingToolYamls = new Map();
  const save = () => localStorage.setItem(KEY, JSON.stringify(state));
  if (seedConversations && !state.conversations) {
    state.conversations = {
      sessions: [{session_id: "session-1", scope_id: "user:test-user", title: "Kitchen project", last_message_at: "2026-09-01T12:00:00Z", turn_count: 1, scope_source: "Test User", turns: [{user_text: "What is the plan?", assistant_text: "Review the kitchen plan.", timestamp: "2026-09-01T12:00:00Z"}]}],
      active: [{key: "active-1", label: "Kitchen speaker", last_active: "2026-09-01T12:00:00Z", expires_at: "2026-10-01T12:00:00Z"}],
    };
    state.scopes[0].conversation_count = 1;
    save();
  }
  const counts = () => {
    state.agent.function_count = state.configuration.config.functions?.length || 0;
    state.agent.function_group_count = state.configuration.config.function_groups?.length || 0;
    state.agent.memory_count = state.memories.length;
    state.agent.knowledge_source_count = state.knowledgeSources?.length || 0;
    state.scopes[0].memory_count = state.memories.filter((m) => m.scope_id === state.scopes[0].scope_id).length;
    if (state.conversations) state.scopes[0].conversation_count = state.conversations.sessions.filter((s) => s.scope_id === state.scopes[0].scope_id).length;
  };
  const tools = () => ({functions: clone(state.configuration.config.functions || []), function_groups: clone(state.configuration.config.function_groups || []), references: {}, revision: state.configuration.revision});
  const normalizeRules = () => state.requestRules.rules.forEach((r, i) => { r.order = i; });
  const parseTool = (yaml) => {
    const text = String(yaml || "");
    const name = text.match(/^\s*name:\s*([^\n#]+)/m)?.[1]?.trim();
    const description = text.match(/^\s*description:\s*([^\n#]+)/m)?.[1]?.trim() || "Browser Function Tool";
    const type = (text.match(/(?:^|\n)function:\s*\n([\s\S]*)/m)?.[1] || "").match(/^\s*type:\s*([^\n#]+)/m)?.[1]?.trim() || "script";
    if (!name) return {valid: false, errors: [{message: "spec.name is required"}]};
    const config = {spec: {name, description, parameters: {type: "object", properties: {}}}, function: {type, sequence: []}, enabled: true};
    pendingToolYamls.set(name, text);
    return {valid: true, errors: [], name, type, config};
  };
  const backup = () => JSON.stringify({fixture_format: 1, title: state.configuration.title, created_at: now(), state: clone(state)}, null, 2);
  const inspect = (text) => {
    const doc = JSON.parse(text); if (!doc?.state?.configuration) throw new Error("Invalid browser fixture backup");
    const s = doc.state;
    return {title: doc.title || s.configuration.title, summary: {created_at: doc.created_at || now(), integration_version: "browser-fixture", request_rules: s.requestRules?.rules?.length || 0, persistent_memories: s.memories?.length || 0, temporary_memories: 0, knowledge_sources: 0, archive_sessions: 0, archive_turns: 0, usage_runs: 0, usage_requests: 0, guest_mode_scheduled: false}};
  };

  async function call(message) {
    const key = `${message.section || ""}/${message.action || ""}`;
    const requireFields = (fields) => {
      for (const [field, type] of fields) {
        const value = message[field];
        if (typeof value !== type || (type === "string" && !value.trim()) || (type === "object" && (!value || Array.isArray(value)))) {
          throw new Error(`Malformed EOAI management ${key}: ${field} must be ${type}`);
        }
      }
    };
    if (typeof message.section !== "string" || typeof message.action !== "string") {
      throw new Error(`Unsupported EOAI management request: section and action are required (${JSON.stringify(message)})`);
    }
    if (key === "overview/summary") {
      return {
        usage: {today: {total_tokens: 1234}, month: {total_tokens: 5678}, lifetime: {total_tokens: 9999}},
        conversations: {archive_enabled: true, archive_retention_days: 30},
        load_errors: partialOverview ? [{key: "knowledge", label: "Knowledge", message: "Knowledge fixture unavailable"}] : [],
      };
    }
    if (key === "overview/primary") {
      return {
        agent: clone(state.agent),
        usage: {},
        conversations: {archive_enabled: true, archive_retention_days: 30},
        load_errors: [],
        loading: {usage: true, memory: true, knowledge: true, guest_mode: true},
        setup_health: {
          provider_runtime: {client_loaded: true, provider: state.agent.provider, model: state.agent.model},
          function_tools: {usable_count: state.agent.function_count || 0, invalid_count: 0},
          prompt_state: "custom",
          exposed_entity_count: 2,
          memory: {mode: state.agent.memory_mode || "manual", available: false, loading: true},
          knowledge: {enabled: true, source_count: 0, available: false, loading: true},
          web_search: {enabled: false},
          can_manage: true,
          live_provider_tested: false,
        },
      };
    }
    if (key === "overview/detail") {
      if (message.kind === "usage") {
        return {kind: "usage", usage: {today: {total_tokens: 1234}, month: {total_tokens: 5678}, lifetime: {total_tokens: 9999}}, agent: {tokens_today: 1234}};
      }
      if (message.kind === "memory") {
        return {kind: "memory", agent: {memory_count: state.memories.length}, setup_health: {memory: {available: true, loading: false}}};
      }
      if (message.kind === "knowledge") {
        if (partialOverview) throw new Error("Knowledge fixture unavailable");
        const sourceCount = state.knowledgeSources?.length || 0;
        return {kind: "knowledge", agent: {knowledge_source_count: sourceCount}, setup_health: {knowledge: {source_count: sourceCount, available: true, loading: false}}};
      }
      if (message.kind === "guest_mode") {
        return {kind: "guest_mode", agent: {guest_mode: {state: "inactive", currently_active: false}}};
      }
      throw new Error(`Unsupported EOAI management overview detail kind: ${JSON.stringify(message.kind)}`);
    }
    if (key === "usage/summary") return seedUsageBoundary
      ? {today: {date: "2026-03-30", total_tokens: 42}, month: {total_tokens: 42}, lifetime: {total_tokens: 42}}
      : {today: {total_tokens: 1234}, month: {total_tokens: 5678}, lifetime: {total_tokens: 9999}};
    if (key === "diagnostics/test_agent") return {status: "passed", checks: [{name: "Model access", status: "passed", message: "Selected model is available"}]};
    if (key === "conversations/settings") return {archive_enabled: true, archive_retention_days: 30, archive_model_search_enabled: false};
    if (key === "knowledge/list") {
      if (partialOverview) throw new Error("Knowledge fixture unavailable");
      const sources = clone(state.knowledgeSources || []);
      const sourceCount = sources.length;
      return {sources, stats: {source_count: sourceCount}, feature_status: {state: state.configuration.config.knowledge_enabled === false ? "disabled" : sourceCount ? "available" : "empty", enabled: state.configuration.config.knowledge_enabled !== false, source_count: sourceCount}};
    }
    if (key === "knowledge/get") {
      const source = (state.knowledgeSources || []).find((item) => item.source_id === message.source_id);
      if (!source) throw new Error("Knowledge source not found");
      return {source: clone(source)};
    }
    if (key === "knowledge/create" || key === "knowledge/update") {
      requireFields([["title", "string"], ["content", "string"]]);
      if (key.endsWith("update")) requireFields([["source_id", "string"]]);
      state.knowledgeSources ||= [];
      let source;
      if (key.endsWith("create")) {
        source = {source_id: `source-${state.nextKnowledgeId++}`, title: message.title, description: message.description || "", content: message.content || "", character_count: String(message.content || "").length, enabled: message.enabled !== false, updated_at: now()};
        state.knowledgeSources.push(source);
      } else {
        source = state.knowledgeSources.find((item) => item.source_id === message.source_id);
        if (!source) throw new Error("Knowledge source not found");
        Object.assign(source, {title: message.title, description: message.description || "", content: message.content || "", character_count: String(message.content || "").length, enabled: message.enabled !== false, updated_at: now()});
      }
      counts(); save();
      const sourceCount = state.knowledgeSources.length;
      return {status: key.endsWith("create") ? "created" : "updated", source: clone(source), summary: clone(source), stats: {source_count: sourceCount}, feature_status: {state: state.configuration.config.knowledge_enabled === false ? "disabled" : "available", enabled: state.configuration.config.knowledge_enabled !== false, source_count: sourceCount}};
    }
    if (key === "knowledge/delete") {
      const before = (state.knowledgeSources || []).length;
      state.knowledgeSources = (state.knowledgeSources || []).filter((item) => item.source_id !== message.source_id);
      counts(); save();
      const sourceCount = state.knowledgeSources.length;
      return {deleted: before - sourceCount, stats: {source_count: sourceCount}, feature_status: {state: state.configuration.config.knowledge_enabled === false ? "disabled" : sourceCount ? "available" : "empty", enabled: state.configuration.config.knowledge_enabled !== false, source_count: sourceCount}};
    }
    if (key === "knowledge/set_enabled") {
      state.configuration.config.knowledge_enabled = message.enabled;
      state.configuration.revision = `${state.configuration.revision}x`;
      state.agent.knowledge_enabled = message.enabled;
      state.agent.feature_status = {
        ...(state.agent.feature_status || {}),
        knowledge: {state: message.enabled ? ((state.knowledgeSources?.length || 0) ? "available" : "empty") : "disabled", enabled: message.enabled, source_count: state.knowledgeSources?.length || 0},
      };
      save();
      return {
        revision: state.configuration.revision,
        knowledge_enabled: message.enabled,
        feature_status: clone(state.agent.feature_status.knowledge),
      };
    }
    if (key === "scopes/catalog") return {scopes: clone(state.scopes)};
    if (key === "guest_mode/get") return clone(state.guest);
    if (key === "guest_mode/details") return {
      policy: {guest_active:false, readable_entity_count:0, controllable_entity_count:0, configured_tool_count:0},
      knowledge_sources: [],
      functions: [],
      function_groups: [],
      domains: [],
    };
    if (key === "guest_mode/save_policy") {
      if (message.revision !== state.guest.revision) throw new Error("Saved data changed; reload before saving.");
      state.guest.config = clone(message.config); state.guest.revision += "x"; save(); return clone(state.guest);
    }
    if (key === "guest_mode/update" || key === "guest_mode/disable") return {status: {state: key.endsWith("disable") ? "inactive" : "active"}};
    if (key === "quiet_hours/get") return clone(state.quiet);
    if (key === "quiet_hours/update") { state.quiet.config = clone(message.config); save(); return clone(state.quiet); }
    if (key === "service_catalog/get") return {services: {}};

    if (key === "configuration/get") {
      const result = clone(state.configuration);
      delete result.local_handling;
      delete result.exposed_attribute_catalog;
      return result;
    }
    if (key === "configuration/retention_get") {
      const {title, revision, config, options} = state.configuration;
      return {title, revision, projection:"retention", config:{usage_request_retention_days:config.usage_request_retention_days, usage_run_retention_days:config.usage_run_retention_days}, options:{usage_request_retention_days:clone(options.usage_request_retention_days), usage_run_retention_days:clone(options.usage_run_retention_days)}};
    }
    if (key === "configuration/live_metadata") {
      const requested = new Set(message.metadata_keys || []);
      return {
        ...(requested.has("local_handling") ? {local_handling: clone(state.configuration.local_handling)} : {}),
        ...(requested.has("exposed_attribute_catalog") ? {
          exposed_attribute_catalog: {
            entities: [],
            saved_unexposed: [],
            identity_policy: "entity_registry",
          },
        } : {}),
      };
    }
    if (key === "configuration/validate") return {valid: true, errors: {}, model_capabilities: {}};
    if (key === "configuration/update" || key === "configuration/save") {
      if (!message.config || typeof message.config !== "object" || Array.isArray(message.config) || typeof message.revision !== "string") {
        throw new Error(`Malformed EOAI management ${key}: config object and revision are required`);
      }
      if (failConfigurationOnce && !state.failedConfigurationOnce) { state.failedConfigurationOnce = true; save(); throw new Error("Fixture rejected configuration save once"); }
      if (message.revision && message.revision !== state.configuration.revision) throw new Error("Saved data changed; reload before saving.");
      const updates = clone(message.config);
      for (const field of ["usage_request_retention_days", "usage_run_retention_days"]) if (field in updates) updates[field] = Number(updates[field]);
      state.configuration = {...state.configuration, title: message.title ?? state.configuration.title, revision: `${state.configuration.revision}x`, config: {...state.configuration.config, ...updates}};
      state.agent.title = state.configuration.title; state.agent.model = state.configuration.config.chat_model;
      state.agent.temporary_memory = state.configuration.config.temporary_memory; counts(); save();
      return {valid: true, errors: {}, ...clone(state.configuration), agent: {
        ...clone(state.agent),
        entry_id: message.entry_id || state.agent.entry_id,
        subentry_id: message.subentry_id || state.agent.subentry_id,
      }};
    }

    if (key === "memories/list") return {memories: clone(state.memories.filter((m) => !message.scope_id || m.scope_id === message.scope_id)), total: state.memories.length};
    if (key === "memories/temporary_list") return {memories: [], scope_id: message.scope_id, stats: {}};
    if (key === "memories/add") {
      requireFields([["scope_id", "string"], ["content", "string"]]);
      const memory = {revision: 1, memory_id: `memory-${state.nextMemoryId++}`, scope_id: message.scope_id, content: message.content, category: message.category || "general", source: "manual", created_at: now(), updated_at: now()};
      state.memories.push(memory); counts(); save(); return {status: "created", scope_id: memory.scope_id, memory: clone(memory)};
    }
    if (key === "memories/update") {
      requireFields([["scope_id", "string"], ["memory_id", "string"], ["content", "string"]]);
      const memory = state.memories.find((m) => m.memory_id === message.memory_id && m.scope_id === message.scope_id); if (!memory) throw new Error("Memory not found");
      memory.content = message.content; memory.category = message.category || "general";
      if (message.target_scope_id) memory.scope_id = message.target_scope_id;
      memory.revision = Number(memory.revision || 0) + 1; memory.updated_at = now(); counts(); save();
      return {status: "updated", scope_id: memory.scope_id, memory: clone(memory)};
    }
    if (key === "memories/delete") {
      const before = state.memories.length;
      state.memories = state.memories.filter((m) => !(m.memory_id === message.memory_id && m.scope_id === message.scope_id));
      counts(); save(); return {deleted: before - state.memories.length};
    }

    if (key === "request_rules/list") return clone(state.requestRules);
    if (key === "request_rules/rule_pack_export") {
      const selected=message.selection==="group" ? state.requestRules.rules.filter(rule=>rule.group_id===(message.group_id||null)) : message.selection==="selected" ? state.requestRules.rules.filter(rule=>message.rule_ids?.includes(rule.id)) : state.requestRules.rules;
      return {format:"extended_openai_request_rule_pack",version:1,groups:clone(state.requestRules.groups.filter(group=>selected.some(rule=>rule.group_id===group.id))),rules:clone(selected.map((rule,order)=>({...rule,order,continue_matching:rule.continue_matching??false,ai_input_mode:rule.ai_input_mode||"original",ai_input_capture:rule.ai_input_capture||null,conditions:rule.conditions||[],group_id:rule.group_id||null})))};
    }
    if (key === "request_rules/rule_pack_review" || key === "request_rules/rule_pack_import") {
      const pack=JSON.parse(message.pack);
      if(pack.format!=="extended_openai_request_rule_pack"||pack.version!==1||!Array.isArray(pack.rules))throw new Error("Invalid rule pack");
      const review={review_token:"fixture-review",count:pack.rules.length,ready:pack.rules.length,needs_attention:0,will_append:true,will_disable:true,new_groups:pack.groups.length,revision:state.requestRules.revision,rules:pack.rules.map(rule=>({name:rule.name,triggers:rule.phrases,group:pack.groups.find(group=>group.id===rule.group_id)?.name||"Ungrouped",action_type:rule.action_type,conditions:rule.conditions?.length||0,function_tools:[],entities:[],services:[],missing_dependencies:[],continue_to_ai:Boolean(rule.action?.continue_to_ai),continue_matching:Boolean(rule.continue_matching),ai_input_mode:rule.ai_input_mode||"original",ai_input_capture:rule.ai_input_capture||null,status:"ready"}))};
      if(key.endsWith("review"))return review;
      if(!message.confirm||message.review_token!=="fixture-review"||message.revision!==state.requestRules.revision)throw new Error("Review again");
      const imported=pack.rules.map(rule=>({...clone(rule),id:`rule-${state.nextRuleId++}`,enabled:false,order:state.requestRules.rules.length}));
      state.requestRules.rules.push(...imported);normalizeRules();state.requestRules.revision++;save();
      return {rules:clone(imported),groups:clone(state.requestRules.groups),revision:state.requestRules.revision,review};
    }
    if (key === "request_rules/create") { requireFields([["rule", "object"]]); const rule = {...clone(message.rule), id: `rule-${state.nextRuleId++}`}; state.requestRules.rules.push(rule); normalizeRules(); state.requestRules.revision++; save(); return {rule: clone(rule), revision: state.requestRules.revision}; }
    if (key === "request_rules/update") { requireFields([["rule_id", "string"], ["rule", "object"]]); const i = state.requestRules.rules.findIndex((r) => r.id === message.rule_id); if (i < 0) throw new Error("Request Rule not found"); state.requestRules.rules[i] = {...clone(message.rule), id: message.rule_id}; normalizeRules(); state.requestRules.revision++; save(); return {rule: clone(state.requestRules.rules[i]), revision: state.requestRules.revision}; }
    if (key === "request_rules/delete") { state.requestRules.rules = state.requestRules.rules.filter((r) => r.id !== message.rule_id); normalizeRules(); state.requestRules.revision++; save(); return {revision: state.requestRules.revision}; }
    if (key === "request_rules/duplicate") {
      if (message.revision !== state.requestRules.revision) throw new Error("Saved data changed; reload before saving.");
      const i = state.requestRules.rules.findIndex(r => r.id === message.rule_id);
      if (i < 0) throw new Error("Request Rule not found");
      const source = state.requestRules.rules[i], names = new Set(state.requestRules.rules.map(r => r.name.toLowerCase()));
      let name = `${source.name} copy`, n = 2;
      while (names.has(name.toLowerCase())) name = `${source.name} copy ${n++}`;
      const rule = {...clone(source), id: `rule-${state.nextRuleId++}`, name};
      state.requestRules.rules.splice(i + 1, 0, rule);
      normalizeRules(); state.requestRules.revision++; save(); return {rule: clone(rule), revision: state.requestRules.revision};
    }
    if (key === "request_rules/move") { const i = state.requestRules.rules.findIndex((r) => r.id === message.rule_id); let j = ({up:i-1,down:i+1,top:0,bottom:state.requestRules.rules.length-1})[message.direction]; if (message.direction === "before" || message.direction === "after") { j=state.requestRules.rules.findIndex((r) => r.id === message.target_rule_id); if (i < j) j--; if (message.direction === "after") j++; } if (i >= 0 && j >= 0 && j < state.requestRules.rules.length) state.requestRules.rules.splice(j,0,state.requestRules.rules.splice(i,1)[0]); normalizeRules(); state.requestRules.revision++; save(); const rule = state.requestRules.rules.find((r) => r.id === message.rule_id); return {rule: clone(rule), revision: state.requestRules.revision}; }
    if (key === "request_rules/groups") { if (message.revision !== state.requestRules.revision) throw new Error("Saved data changed; reload before saving."); state.requestRules.groups = clone(message.groups).map((group) => ({...group, id: group.id || `group-${state.nextRuleGroupId++}`})); const ids = new Set(state.requestRules.groups.map((group) => group.id)); state.requestRules.rules.forEach((rule) => {if(!ids.has(rule.group_id))rule.group_id=null;}); state.requestRules.revision++; save(); return {groups:clone(state.requestRules.groups),rules:clone(state.requestRules.rules),revision:state.requestRules.revision}; }
    if (key === "request_rules/settings") {
      if (message.revision !== state.requestRules.revision) throw new Error("Saved data changed; reload before saving.");
      state.requestRules.defaults = clone(message.defaults);
      state.requestRules.wording_groups = clone(message.wording_groups);
      state.requestRules.revision++;
      save();
      return {defaults: clone(state.requestRules.defaults), wording_groups: clone(state.requestRules.wording_groups), revision: state.requestRules.revision};
    }
    if (key === "request_rules/defaults") { if (message.revision !== state.requestRules.revision) throw new Error("Saved data changed; reload before saving."); state.requestRules.defaults = clone(message.defaults); state.requestRules.revision++; save(); return {defaults: clone(state.requestRules.defaults), revision: state.requestRules.revision}; }
    if (key === "request_rules/wording_groups") { if (message.revision !== state.requestRules.revision) throw new Error("Saved data changed; reload before saving."); state.requestRules.wording_groups = clone(message.wording_groups); state.requestRules.revision++; save(); return {wording_groups: clone(state.requestRules.wording_groups), revision: state.requestRules.revision}; }

    if (key === "tools/starter") return {yaml: "spec:\n  name: browser_tool\n  description: Browser Function Tool\n  parameters:\n    type: object\n    properties: {}\nfunction:\n  type: script\n  sequence: []\n"};
    if (key === "tools/built_in_catalog") return {functions: []};
    if (key === "tools/serialize") return {yaml: state.toolYamls[message.tool?.spec?.name] || ""};
    if (key === "tools/validate_yaml") return parseTool(message.yaml);
    if (key === "tools/save") {
      requireFields([["tool", "object"], ["revision", "string"]]);
      if (message.revision !== state.configuration.revision) throw new Error("Configuration changed in another tab. Reload the latest saved settings before saving.");
      
      const tool = clone(message.tool), list = state.configuration.config.functions || [], original = message.original_name;
      const i = list.findIndex((t) => t.spec?.name === (original || tool.spec?.name)); if (i >= 0) list[i] = tool; else list.push(tool);
      const name = tool.spec?.name;
      if (name && pendingToolYamls.has(name)) state.toolYamls[name] = pendingToolYamls.get(name);
      if (name) pendingToolYamls.delete(name);
      if (original && original !== tool.spec?.name) { for (const g of state.configuration.config.function_groups || []) g.functions = (g.functions || []).map((n) => n === original ? tool.spec.name : n); delete state.toolYamls[original]; pendingToolYamls.delete(original); }
      counts(); state.configuration.revision = `${state.configuration.revision}x`; save(); return tools();
    }
    if (key === "tools/delete") {
      if (message.revision !== state.configuration.revision) throw new Error("Configuration changed in another tab. Reload the latest saved settings before saving.");
       state.configuration.config.functions = (state.configuration.config.functions || []).filter((t) => t.spec?.name !== message.name); for (const g of state.configuration.config.function_groups || []) g.functions = (g.functions || []).filter((n) => n !== message.name); delete state.toolYamls[message.name]; pendingToolYamls.delete(message.name); counts(); state.configuration.revision = `${state.configuration.revision}x`; save(); return tools(); }
    if (key === "tools/set_enabled") {
      if (message.revision !== state.configuration.revision) throw new Error("Configuration changed in another tab. Reload the latest saved settings before saving.");
       const tool = (state.configuration.config.functions || []).find((t) => t.spec?.name === message.name); if (tool) tool.enabled = message.enabled; state.configuration.revision = `${state.configuration.revision}x`; save(); return tools(); }
    if (key === "tools/save_group") {
      if (message.revision !== state.configuration.revision) throw new Error("Configuration changed in another tab. Reload the latest saved settings before saving.");
       const group = clone(message.group), list = state.configuration.config.function_groups || [], original = message.original_id || group.id; for (const g of list) if (g.id !== original) g.functions = (g.functions || []).filter((n) => !(group.functions || []).includes(n)); const i = list.findIndex((g) => g.id === original); if (i >= 0) list[i] = group; else list.push(group); counts(); state.configuration.revision = `${state.configuration.revision}x`; save(); return tools(); }
    if (key === "tools/delete_group") {
      if (message.revision !== state.configuration.revision) throw new Error("Configuration changed in another tab. Reload the latest saved settings before saving.");
       state.configuration.config.function_groups = (state.configuration.config.function_groups || []).filter((g) => g.id !== message.group_id); counts(); state.configuration.revision = `${state.configuration.revision}x`; save(); return tools(); }
    if (key === "tools/validate_current") return {valid: true, errors: []};

    if (key === "backup/create") return {json: backup(), filename: "browser-fixture-full-backup.json"};
    if (key === "backup/inspect") return inspect(message.document);
    if (key === "backup/restore") { const doc = JSON.parse(message.document); if (!doc?.state?.configuration) throw new Error("Invalid browser fixture backup"); state = clone(doc.state); pendingToolYamls.clear(); counts(); save(); return {restored: true}; }

    if (key === "conversations/list" || key === "conversations/search") {
      requireFields([["scope_id", "string"]]);
      if (key === "conversations/search") requireFields([["query", "string"]]);
      const offset = Math.max(0, Number(message.offset) || 0);
      const limit = Math.max(1, Number(message.limit) || 50);
      const sessions = (state.conversations?.sessions || []).filter((item) => item.scope_id === message.scope_id);
      if (key === "conversations/search") {
        const results = sessions.flatMap((item) => item.turns.filter((turn) => `${turn.user_text} ${turn.assistant_text}`.toLowerCase().includes(message.query.toLowerCase())).map((turn) => ({...item, timestamp: turn.timestamp})));
        return {results: clone(results.slice(offset, offset + limit)), offset, limit, total: results.length, returned: results.slice(offset, offset + limit).length, has_more: offset + limit < results.length};
      }
      return {sessions: clone(sessions.slice(offset, offset + limit)), offset, limit, total: sessions.length, returned: sessions.slice(offset, offset + limit).length, has_more: offset + limit < sessions.length};
    }
    if (key === "conversations/active") return {active: clone(state.conversations?.active || [])};
    if (key === "conversations/get") {
      requireFields([["scope_id", "string"], ["session_id", "string"]]);
      const session = state.conversations?.sessions.find((item) => item.scope_id === message.scope_id && item.session_id === message.session_id);
      if (!session) throw new Error(`Unknown EOAI fixture conversation: ${message.session_id}`);
      const offset = Math.max(0, Number(message.start_turn) || 0);
      const limit = Math.max(1, Number(message.limit) || 20);
      return {session: {title: session.title}, turns: clone(session.turns.slice(offset, offset + limit)), offset, limit, total: session.turns.length, returned: session.turns.slice(offset, offset + limit).length, has_more: offset + limit < session.turns.length};
    }
    if (key === "conversations/delete") {
      requireFields([["scope_id", "string"], ["session_id", "string"]]);
      const sessions = state.conversations?.sessions || [];
      const remaining = sessions.filter((item) => item.scope_id !== message.scope_id || item.session_id !== message.session_id);
      if (remaining.length === sessions.length) throw new Error(`Unknown EOAI fixture conversation: ${message.session_id}`);
      state.conversations.sessions = remaining; counts(); save();
      return {deleted_sessions: 1};
    }
    if (key === "conversations/end_active") {
      requireFields([["continuity_key", "string"]]);
      const active = state.conversations?.active || [];
      const remaining = active.filter((item) => item.key !== message.continuity_key);
      if (remaining.length === active.length) throw new Error(`Unknown EOAI fixture active conversation: ${message.continuity_key}`);
      state.conversations.active = remaining; save();
      return {ended: true};
    }
    if (seedUsageBoundary && key === "usage/daily") {
      const date = "2026-03-30";
      return {days: message.start_date <= date && message.end_date >= date
        ? [{date, total_tokens: 42, input_tokens: 30, output_tokens: 12, run_count: 1, api_request_count: 1}]
        : []};
    }
    if (seedUsageBoundary && key === "usage/runs") return {runs: [{completed_at: "2026-03-29T23:30:00Z", total_tokens: 42, cached_input_tokens: 0, request_count: 1, duration_ms: 100, successful: true}]};
    const staticResponses = {"usage/daily": {days: []}, "usage/runs": {runs: []}, "usage/retention": {}};
    if (Object.hasOwn(staticResponses, key)) return clone(staticResponses[key]);
    throw new Error(`Unsupported EOAI management fixture request ${key}: ${JSON.stringify(message)}`);
  }

  async function debugCall(message) {
    if (message.action === "agents") return {agents: [clone(state.agent)]};
    if (message.action === "runs") return {...clone(state.requestDebug), allowed_limits: [5, 10, 25, 50]};
    if (message.action === "configure") {
      if (typeof message.enabled !== "boolean" && ![5, 10, 25, 50].includes(message.limit)) {
        throw new Error(`Malformed EOAI request-debug configuration: ${JSON.stringify(message)}`);
      }
      if (message.enabled !== undefined) state.requestDebug.enabled = message.enabled;
      if (message.limit !== undefined) state.requestDebug.limit = message.limit;
      save();
      return {...clone(state.requestDebug), allowed_limits: [5, 10, 25, 50]};
    }
    if (message.action === "clear") {
      if (message.confirm !== true) throw new Error("Request-debug clear requires confirmation");
      state.requestDebug.count = 0; state.requestDebug.runs = []; save();
      return {cleared: true};
    }
    throw new Error(`Unsupported EOAI request-debug fixture action: ${JSON.stringify(message)}`);
  }

  counts();
  return {call, debugCall, state: () => clone(state), reset: () => { state = freshState(); pendingToolYamls.clear(); save(); return clone(state); }, agent: () => clone(state.agent), scopes: () => clone(state.scopes)};
}
