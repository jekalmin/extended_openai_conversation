// Deterministic authoritative backend for unified Memory and Knowledge.
// Mutations are applied here, never optimistically in the panel under test.
export function createDataCollectionBackend(size = 100) {
  let revision = 1;
  let knowledgeEnabled = true;
  let nextId = size;
  const timestamp = () => `2026-09-01T12:${String(revision++).padStart(2, "0")}:00Z`;
  const state = {
    sources: Array.from({length: size}, (_, i) => ({source_id: `source-${i}`, title: `Source ${i}`, description: `Reference category ${i % 5}`, content: `Reference text ${i}`, character_count: 18, enabled: true, updated_at: "2026-09-01T12:00:00Z"})),
    memories: Array.from({length: size}, (_, i) => ({memory_id: `memory-${i}`, scope_id: "user:test-user", content: `Memory ${i}`, category: `category-${i % 5}`, scope: i % 2 ? "Shared household" : "Personal", importance: ["low", "normal", "high"][i % 3], source: "manual", revision: 1, updated_at: "2026-09-01T12:00:00Z"})),
    temporary: Array.from({length: 12}, (_, i) => ({memory_id: `temporary-${i}`, scope_id: "user:test-user", owner_scope_id: "user:test-user", content: `Temporary ${i}`, category: "general", source: "manual", expires_at: "2099-09-01T12:00:00Z", updated_at: "2026-09-01T12:00:00Z"})),
  };
  const calls = [];
  const clone = value => structuredClone(value);
  const scoped = (items, message) => items.filter(item => (!message.scope_id || item.scope_id === message.scope_id) && (!message.subentry_id || (item.subentry_id || "agent-1") === message.subentry_id));
  async function call(message) {
    calls.push(clone(message));
    const section = message.section || "memories";
    const action = message.action;
    if (section === "knowledge") {
      if (action === "list") return {sources: clone(state.sources)};
      if (action === "get") return {source: clone(state.sources.find(source => source.source_id === message.source_id))};
      if (action === "set_enabled") {
        if (typeof message.enabled !== "boolean") throw new Error("Knowledge availability requires a boolean enabled field");
        knowledgeEnabled = message.enabled;
        return {knowledge_enabled: knowledgeEnabled, revision: `collection-${revision++}`,
          feature_status: {state: knowledgeEnabled ? "available" : "disabled", enabled: knowledgeEnabled, source_count: state.sources.length}};
      }
      let source;
      if (action === "create") {
        source = {source_id: `source-${nextId++}`, title: message.title, description: message.description, content: message.content, character_count: message.content.length, enabled: message.enabled, updated_at: timestamp()};
        state.sources.push(source);
      } else if (action === "update") {
        source = state.sources.find(item => item.source_id === message.source_id);
        if (!source) throw new Error("Source not found");
        Object.assign(source, {title: message.title, description: message.description, content: message.content, character_count: message.content.length, enabled: message.enabled, updated_at: timestamp()});
      } else if (action === "delete") {
        const before = state.sources.length;
        state.sources = state.sources.filter(item => item.source_id !== message.source_id);
        return {deleted: before - state.sources.length, stats: {knowledge_source_count: state.sources.length},
          feature_status: {source_count: state.sources.length, enabled: true}};
      } else {
        throw new Error(`Unhandled Knowledge fixture action: ${action}`);
      }
      return {source: clone(source), summary: clone(source), stats: {knowledge_source_count: state.sources.length},
        feature_status: {source_count: state.sources.length, enabled: true}};
    }
    if (section !== "memories") throw new Error(`Unhandled fixture section: ${section}`);
    if (action === "list" || action === "search") {
      let items = scoped(state.memories, message);
      if (message.subentry_id === "second-agent") items = [{...state.memories[0], content: "Second agent memory"}];
      if (action === "search") items = items.filter(item => `${item.content} ${item.category} ${item.source}`.toLowerCase().includes(message.query.toLowerCase()));
      items = [...items].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
      const offset = message.offset || 0, limit = message.limit || 100;
      const next = offset + limit < items.length ? offset + limit : null;
      return {memories: clone(items.slice(offset, offset + limit)), total: items.length, has_more: next !== null, next_offset: next};
    }
    if (action === "temporary_list") return {memories: clone(scoped(state.temporary, message)), stats: {}};
    if (action === "temporary_delete") { const before = state.temporary.length; state.temporary = state.temporary.filter(item => item.memory_id !== message.memory_id); return {deleted: before - state.temporary.length}; }
    if (action === "temporary_clear") { if (!message.confirm) throw new Error("Confirmation required"); const before = state.temporary.length; state.temporary = state.temporary.filter(item => item.scope_id !== message.scope_id || (item.subentry_id || "agent-1") !== message.subentry_id); return {deleted: before - state.temporary.length}; }
    if (action === "temporary_update") {
      const item = state.temporary.find(item => item.memory_id === message.memory_id);
      Object.assign(item, {content: message.content, category: message.category, expires_at: message.expires_at, updated_at: timestamp()});
      return {memory: clone(item)};
    }
    if (action === "add") {
      const memory = {memory_id: `memory-${nextId++}`, scope_id: message.target_scope_id || message.scope_id || "user:test-user", content: message.content, category: message.category, scope: message.scope === "household" ? "Shared household" : "Personal", importance: message.importance || "normal", source: "manual", revision: 1, updated_at: timestamp()};
      state.memories.push(memory);
      return {status: "created", scope_id: memory.scope_id, memory: clone(memory)};
    }
    if (action === "update") {
      const memory = scoped(state.memories, message).find(item => item.memory_id === message.memory_id);
      if (!memory) throw new Error("Memory not found");
      if (message.expected_revision !== memory.revision) throw new Error("memory changed since it was loaded; reopen it before saving");
      if (message.target_scope_id) memory.scope_id = message.target_scope_id;
      if (message.refresh_confirmation) memory.last_confirmed_at = timestamp();
      for (const key of ["content", "category", "importance", "subject", "key", "valid_from"]) if (message[key] !== undefined) memory[key] = message[key];
      if (message.scope) memory.scope = message.scope === "household" ? "Shared household" : "Personal";
      for (const key of message.clear_fields || []) delete memory[key];
      memory.revision++; memory.updated_at = timestamp();
      return {status: "updated", scope_id: memory.scope_id, memory: clone(memory)};
    }
    if (action === "delete") { const before = state.memories.length; state.memories = state.memories.filter(item => item.memory_id !== message.memory_id); return {deleted: before - state.memories.length}; }
    if (action === "clear") {
      state.memories = state.memories.filter(item => (message.category && item.category !== message.category) || (message.scope && item.scope !== (message.scope === "household" ? "Shared household" : "Personal")));
      return {};
    }
    throw new Error(`Unhandled fixture action: ${action}`);
  }
  return {state, calls, call};
}
