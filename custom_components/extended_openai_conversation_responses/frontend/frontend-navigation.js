export const NAVIGATION = [
  {id: "overview", label: "Overview", path: "/extended-openai/overview", sections: []},
  {id: "guide", label: "Guide", path: "/extended-openai/guide", sections: []},
  {id: "assistant", label: "Assistant", path: "/extended-openai/assistant/basics", sections: [
    {id: "basics", label: "Basics", description: "Name the assistant and choose its everyday model and response behavior."},
    {id: "model-responses", label: "Model & responses", description: "Fine-tune supported model response controls."},
    {id: "conversation", label: "Conversation", description: "Control how recent conversations are carried into later requests."},
    {id: "prompt-context", label: "Prompt & context", description: "Manage instructions and live Home Assistant context."},
    {id: "voice", label: "Voice & identity", description: "Choose identity and retained-data behavior for signed-in and unidentified voice requests."},
    {id: "speech", label: "Speech", description: "Clean assistant responses before they are spoken."},
  ]},
  {id: "capabilities", label: "Capabilities", path: "/extended-openai/capabilities/home-assistant", sections: [
    {id: "home-assistant", label: "Home Assistant & local handling", description: "Review Home Assistant access and choose which simple commands should stay local."},
    {id: "web-skills", label: "Web search & Skills", description: "Configure current-information search and installed instruction sets."},
    {id: "request-rules", label: "Request Rules", description: "Create fast local commands and route AI requests by phrase."},
    {id: "functions", label: "Functions", description: "Manage custom Function Tools and on-demand groups."},
    {id: "quiet-hours", label: "Quiet Hours", description: "Make Assist satellites quieter at set times each day and expose that period to Home Assistant automations."},
    {id: "guest-mode", label: "Guest Mode", description: "Limit what visitors can see, use, and remember."},
  ]},
  {id: "data-memory", label: "Data & Memory", path: "/extended-openai/data-memory/memory-settings", sections: [
    {id: "memory-settings", label: "Memory settings", description: "Control long-term, temporary, retrieval, and shared memory behavior."},
    {id: "memories", label: "Memories", description: "Review durable and automatically expiring memories."},
    {id: "knowledge", label: "Knowledge Library", description: "Control and maintain larger reference sources for on-demand search."},
    {id: "conversations", label: "Conversation history", description: "Review continuity and retained conversation archives."},
  ]},
  {id: "usage-maintenance", label: "Usage & Maintenance", path: "/extended-openai/usage-maintenance/usage", sections: [
    {id: "usage", label: "Usage", description: "Review token usage and recent runs."},
    {id: "request-debug", label: "Request debugging", description: "Capture and inspect complete recent provider requests."},
    {id: "backup-restore", label: "Backup & Restore", description: "Create or restore a private agent backup."},
    {id: "diagnostics", label: "Diagnostics", description: "Test the provider and inspect the selected agent."},
    {id: "retention", label: "Retention & maintenance", description: "Control detailed usage retention and cleanup."},
  ]},
];

const LEGACY_ROUTES = {
  configuration: ["assistant", "basics"],
  tools: ["capabilities", "functions"],
  guest: ["capabilities", "guest-mode"],
  memories: ["data-memory", "memories"],
  knowledge: ["data-memory", "knowledge"],
  conversations: ["data-memory", "conversations"],
  usage: ["usage-maintenance", "usage"],
  diagnostics: ["usage-maintenance", "diagnostics"],
  "assistant/advanced": ["capabilities", "web-skills"],
};

export function routeFromPath(pathname) {
  const parts = pathname.split("/").filter(Boolean);
  const marker = parts.lastIndexOf("extended-openai");
  const route = marker >= 0 ? parts.slice(marker + 1) : parts;
  if (!route.length) return {page: "overview", section: null, legacy: false};
  const legacyKey = route.length > 1 ? `${route[0]}/${route[1]}` : route[0];
  const legacy = LEGACY_ROUTES[legacyKey] || LEGACY_ROUTES[route[0]];
  if (legacy) {
    const [page, section] = legacy;
    return {page, section, legacy: true};
  }
  const page = NAVIGATION.find((item) => item.id === route[0]);
  if (!page) return {page: "overview", section: null, legacy: false};
  const section = page.sections.find((item) => item.id === route[1])?.id || page.sections[0]?.id || null;
  return {page: page.id, section, legacy: false};
}

export function routePath(page, section = null) {
  return `/extended-openai/${page}${section ? `/${section}` : ""}`;
}

export function pageMetadata(page) {
  return NAVIGATION.find((item) => item.id === page) || NAVIGATION[0];
}

export function shouldShowGlobalSettingsSearch() {
  return true;
}

export function focusManagementSetting(element) {
  if (!element) return;
  for (let parent = element.parentElement; parent; parent = parent.parentElement) {
    if (parent.tagName === "DETAILS") parent.open = true;
  }
  const field = element.closest?.("[data-setting]") || element;
  const selector = "input:not([type=hidden]),select,textarea,button,ha-user-picker,ha-entity-picker";
  const control = element.matches?.(selector) ? element : field.querySelector?.(selector);
  field.scrollIntoView?.({behavior:"smooth", block:"start"});
  if (control && !control.disabled) control.focus?.();
  else {
    field.tabIndex = -1;
    field.focus?.();
  }
}
