const clone = (value) => structuredClone(value);
const encoder = new TextEncoder();
const decoder = new TextDecoder();

function bytesToBase64(bytes) {
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return btoa(binary);
}

function base64ToBytes(value) {
  const binary = atob(String(value || ""));
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function backupDocument(backend) {
  const state = backend.state();
  return {
    fixture_format: 1,
    title: state.configuration.title,
    created_at: new Date().toISOString(),
    state,
  };
}

function summary(document) {
  const state = document.state;
  return {
    created_at: document.created_at,
    integration_version: "browser-fixture",
    request_rules: state.requestRules?.rules?.length || 0,
    persistent_memories: state.memories?.length || 0,
    temporary_memories: 0,
    knowledge_sources: 0,
    archive_sessions: 0,
    archive_turns: 0,
    usage_runs: 0,
    usage_requests: 0,
    guest_mode_scheduled: false,
  };
}

function inspection(document) {
  if (!document?.state?.configuration) throw new Error("Invalid browser fixture backup");
  return {
    valid: true,
    source_kind: "full_backup",
    title: document.title || document.state.configuration.title,
    summary: summary(document),
    available_sections: ["configuration", "request_rules", "persistent_memory", "temporary_memory", "knowledge", "conversation_archive", "usage", "guest_mode"],
    preview: {preserved_sensitive_field_count: 0, missing_sensitive_field_count: 0},
  };
}

export function createBackupTransferBackend(backend) {
  let sequence = 1;
  const exports = new Map();
  const imports = new Map();

  const sessionId = (prefix) => `${prefix}-${sequence++}`;
  const importDocument = (session) => JSON.parse(decoder.decode(session.bytes));

  async function restore(document, sections) {
    const selected = new Set(sections || []);
    const saved = document.state;

    if (selected.has("configuration")) {
      await backend.call({
        section: "configuration",
        action: "save",
        title: saved.configuration.title,
        config: clone(saved.configuration.config),
        revision: backend.state().configuration.revision,
      });
    }

    if (selected.has("persistent_memory")) {
      const existing = await backend.call({section: "memories", action: "list", limit: 100});
      for (const memory of existing.memories || []) {
        await backend.call({section: "memories", action: "delete", memory_id: memory.memory_id, scope_id: memory.scope_id});
      }
      for (const memory of saved.memories || []) {
        await backend.call({section: "memories", action: "add", scope_id: memory.scope_id, content: memory.content, category: memory.category});
      }
    }

    if (selected.has("request_rules")) {
      const current = await backend.call({section: "request_rules", action: "list"});
      for (const rule of current.rules || []) {
        await backend.call({section: "request_rules", action: "delete", rule_id: rule.id});
      }
      for (const rule of [...(saved.requestRules?.rules || [])].sort((a, b) => (a.order || 0) - (b.order || 0))) {
        const next = clone(rule);
        delete next.id;
        await backend.call({section: "request_rules", action: "create", rule: next});
      }
    }
  }

  return async (message) => {
    const action = message.action;
    const data = message.data || {};

    if (action === "export_start") {
      const id = sessionId("export");
      const bytes = encoder.encode(JSON.stringify(backupDocument(backend)));
      exports.set(id, bytes);
      return {session_id: id, chunk_count: 1, size: bytes.length, filename: "browser-fixture-full-backup.zip", content_type: "application/zip"};
    }
    if (action === "export_chunk") {
      const bytes = exports.get(data.session_id);
      if (!bytes || data.index !== 0) throw new Error("Invalid browser fixture export session");
      return {session_id: data.session_id, index: 0, offset: 0, bytes: bytes.length, data: bytesToBase64(bytes)};
    }
    if (action === "export_cancel") {
      exports.delete(data.session_id);
      return {cancelled: true};
    }

    if (action === "import_start") {
      const id = sessionId("import");
      const size = Number(data.size || 0);
      if (!Number.isInteger(size) || size < 1) throw new Error("Invalid browser fixture import size");
      imports.set(id, {size, bytes: new Uint8Array(size), received: 0, nextIndex: 0});
      return {session_id: id, chunk_size: Math.max(size, 1), chunk_count: 1};
    }
    if (action === "import_chunk") {
      const session = imports.get(data.session_id);
      if (!session || data.index !== session.nextIndex) throw new Error("Invalid browser fixture import sequence");
      const chunk = base64ToBytes(data.data);
      session.bytes.set(chunk, session.received);
      session.received += chunk.length;
      session.nextIndex += 1;
      return {session_id: data.session_id, next_index: session.nextIndex, received: session.received};
    }
    if (action === "import_inspect") {
      const session = imports.get(data.session_id);
      if (!session || session.received !== session.size) throw new Error("Incomplete browser fixture import");
      session.previewToken = sessionId("preview");
      return {...inspection(importDocument(session)), preview_token: session.previewToken};
    }
    if (action === "import_restore") {
      const session = imports.get(data.session_id);
      if (!session || session.received !== session.size) throw new Error("Incomplete browser fixture import");
      if (!data.preview_token || data.preview_token !== session.previewToken) throw new Error("Stale browser fixture preview");
      await restore(importDocument(session), data.sections);
      imports.delete(data.session_id);
      return {restored: true};
    }
    if (action === "import_cancel") {
      imports.delete(data.session_id);
      return {cancelled: true};
    }

    throw new Error(`Unsupported browser fixture backup transfer action: ${action}`);
  };
}
