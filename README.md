# Extended OpenAI Conversation (Home Assistant)

**A drop-in conversation agent for Home Assistant with first-class GPT‑5 support.**  
Works with Assist pipelines, supports OpenAI’s **Responses API** (reasoning models), and keeps memory/tools **optional** and **off by default** for reliability.

> This fork focuses on **stable voice UX** and **GPT‑5 compatibility**. The local memory/RAG layer is scaffolded but **disabled by default** until the core path is rock‑solid.

---

## Why use this?

- **GPT‑5** support via **Responses API**, including `reasoning.effort: low|medium|high`.
- Clean, deterministic prompt assembly; safe default tone.
- Plays nice with **Assist**: no websocket UI crashes, exposes `continue_conversation`.
- **No streaming surprises** (intentionally off until a robust streaming path lands).
- Memory/RAG **scaffolded** but **off** (easy to enable later).

---

## Installation

### Option A — HACS (recommended)
1. In HACS, add this repository as a **Custom repository** (Category: *Integration*).
2. Install **Extended OpenAI Conversation**.
3. **Restart Home Assistant.**
4. Go to **Settings → Devices & Services → Add Integration** → *Extended OpenAI Conversation*.
5. Enter your **OpenAI API key**. Leave Base URL as `https://api.openai.com/v1` unless using a compatible proxy.

### Option B — Manual
Copy `custom_components/extended_openai_conversation` into your HA `/config/custom_components` folder → **Restart** → add the integration as above.

---

## Assign to your Voice Assistant

1. **Settings → Voice assistants** → select your assistant.
2. Under **Conversation agent**, pick **Extended OpenAI Conversation**.
3. You may keep **Prefer handling commands locally** enabled—Assist will still short‑circuit simple HA intents locally.

---

## Options (what matters)

| Option | What it does | Recommended |
|---|---|---|
| **chat_model** | Model name | `gpt-5` |
| **Use Responses API** | Uses Responses API when supported | ✅ On |
| **Model strategy** | Auto/force chat/force responses | **Auto** |
| **Reasoning effort** | GPT‑5 planning depth | `low` or `medium` |
| **Max completion tokens** | Output length (Responses API) | `1024`–`2048` |
| **Temperature / Top‑P** | Sampling | **Ignored by GPT‑5**; used for non‑reasoning models |
| **Context threshold / Truncation** | When we include dialog history | Active once history is enabled |

**Notes**
- GPT‑5 **ignores** temperature/top‑p; use **Reasoning effort** instead.
- Streaming is intentionally disabled in this release to avoid partial speech in Assist.
- Tools/memory are off—router patterns won’t trigger anything yet.

---

## Current status

- ✅ GPT‑5 via Responses (non‑streaming)  
- ✅ Works with Assist; sets `continue_conversation` properly  
- ✅ Proper config‑entry migration handler (no more “Migration handler not found”)  
- ⏳ Dialog **history** and **memory/tools**: planned, off by default

---

## Troubleshooting

- **“Migration handler not found” banner**  
  Update to ≥1.2.0. We added a real `async_migrate_entry()` and use HA’s `async_update_entry` under the hood. :contentReference[oaicite:0]{index=0}

- **500 in Options / blank arrows**  
  Caused by an options schema that returned a raw list (voluptuous‑serialize can’t convert lists). Fixed by a single‑screen options UI. :contentReference[oaicite:1]{index=1}

- **ImportError referencing constants** (`CONF_PAYLOAD_TEMPLATE`, `DEFAULT_MEMORY_WRITE_PATH`, `SERVICE_QUERY_IMAGE`)  
  Old forks referenced constants not present in some copies of `const.py`. We removed those imports and made memory/tools optional. 

- **400 “Unsupported parameter: temperature”**  
  That occurs when sending `temperature` to a GPT‑5 reasoning call. We suppress sampling params for GPT‑5 and use `reasoning.effort` instead.

---

## Privacy

This integration sends prompts to OpenAI (or your configured compatible endpoint). No local memories are written unless you explicitly enable the memory layer later.

---

## Credits

Based on the original project by @jekalmin; this fork focuses on GPT‑5 compatibility and voice reliability.
