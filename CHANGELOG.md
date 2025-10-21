# Changelog

## 1.2.0 — 2025-10-20
### Added
- **GPT‑5 support** via **Responses API**, with `reasoning.effort` (`low`, `medium`, `high`).
- **Compat result** for Home Assistant Assist (`continue_conversation` + stable speech payload).
- **Config entry migration handler** to prevent “Migration handler not found”. :contentReference[oaicite:3]{index=3}
- **Single‑screen Options UI** (no more blank arrow pages / list schema errors). :contentReference[oaicite:4]{index=4}

### Changed
- Default to **non‑streaming** responses (streaming path will return later when fully hardened).
- Suppress `temperature/top_p` for GPT‑5 (they’re not supported on reasoning models).

### Fixed
- Removed legacy imports that referenced missing constants in some forks (`CONF_PAYLOAD_TEMPLATE`, `DEFAULT_MEMORY_WRITE_PATH`, `SERVICE_QUERY_IMAGE`). 
- Avoided blocking calls on the event loop for auth/client init by using HA’s shared HTTPX client.

### Known limits
- Dialog **history** is off (stateless per turn). Will be added behind a token budget.
- Tools & memory are scaffolded but **disabled by default**.
