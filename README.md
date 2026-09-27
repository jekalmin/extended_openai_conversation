# Extended OpenAI Conversation (Responses)

Bring a more capable AI conversation agent to Home Assistant.

Extended OpenAI Conversation connects Home Assistant Assist to the OpenAI API, or a compatible provider, and adds richer Home Assistant control, memory, web search, local request handling, custom tools, voice features, privacy controls, diagnostics, and more.

It is designed to work well with a simple setup first, while keeping advanced features available when you need them.

> [!IMPORTANT]
> **A ChatGPT subscription does not include OpenAI API usage.**
>
> This integration uses the OpenAI API or another configured provider. API usage may be billed separately by that provider.

This project began as a fork of [jekalmin/extended_openai_conversation](https://github.com/jekalmin/extended_openai_conversation), but has since diverged substantially in features, architecture, configuration, testing, and user interface.

## Highlights

- **Home Assistant control** — work with entities exposed to Assist and call supported Home Assistant capabilities.
- **Responses API + Chat Completions** — use automatic API selection or choose a compatible mode explicitly.
- **Web Search** — let supported OpenAI models retrieve current public information when needed.
- **Conversation continuity & context management** — preserve useful conversational context while keeping long conversations bounded.
- **Temporary & Persistent Memory** — retain short-lived context with expiry or longer-term useful facts and preferences.
- **Knowledge Library** — give agents larger local reference material that can be searched on demand.
- **Request Rules & local handling** — build conditional local shortcuts, chain native Home Assistant actions and Function Tools, route selected AI requests, and safely share reusable rules. Native conversation responses override the rule acknowledgement; Stop and false conditions follow Home Assistant script flow. Continue to AI runs only after normal local completion. New rules start enabled, while imported rules start disabled. See the [Request Rules guide](docs/features/request-rules.md) for response and variable semantics.
- **Function Tools & Function Groups** — add custom capabilities and load large tool collections only when needed.
- **Delayed Function Tools** — persist eligible delayed work and re-check current authorization before it executes.
- **Voice Identity** — map Assist devices/satellites to Home Assistant users so retained data can follow the person speaking.
- **Voice follow-ups, speech cleanup & Quiet Hours** — improve spoken conversations and manage satellite behaviour overnight.
- **Broadcast** — send one-way spoken announcements to selected Assist satellites or the whole home.
- **Guest Mode** — apply a backend-enforced visitor policy that can only narrow normal access.
- **Conversation archive** — optionally retain searchable conversation history locally.
- **Skills** — load reusable instruction sets for an agent when needed.
- **AI Task support** — create model-backed AI Task agents.
- **Usage, request debugging, diagnostics & backups** — inspect operation and maintain or migrate individual agents.

You do **not** need to configure all of these features. A basic conversation agent can be set up with mostly default settings.

## Before you install

You will need:

- Home Assistant
- an OpenAI API key, or credentials for a compatible provider
- a Home Assistant Assist pipeline if you want to use Extended OpenAI as a conversation agent

For normal Home Assistant entity knowledge and control, expose the entities you want the assistant to use under **Settings → Voice assistants → Expose**.

> [!NOTE]
> Assist exposure is not an authorization bypass. The requesting Home Assistant user's permissions still apply, and Extended OpenAI policies such as Guest Mode can further restrict access.

## Installation

### HACS

1. Open **HACS** in Home Assistant.
2. Choose **Custom repositories**.
3. Add:

   `https://github.com/conorod1992/extended_openai_conversation`

4. Select **Integration** as the repository type.
5. Install **Extended OpenAI Conversation (Responses)**.
6. Restart Home Assistant.

### Manual

Copy:

`custom_components/extended_openai_conversation_responses`

into:

`<config>/custom_components/extended_openai_conversation_responses`

and restart Home Assistant.

See the [installation guide](docs/getting-started/installation.md) for the full walkthrough.

## First setup

After restarting Home Assistant:

1. Open **Settings → Devices & services → Add Integration**.
2. Add **Extended OpenAI Conversation (Responses)** and configure your provider credentials.
3. Open **Extended OpenAI** from the Home Assistant sidebar and select the conversation agent you want to configure.
4. Open **Settings → Voice assistants**, edit the Assist pipeline you want to use, and select **Extended OpenAI Conversation (Responses)** as its conversation agent.
5. Under **Settings → Voice assistants → Expose**, expose a small set of entities for an initial test.
6. Try a simple request such as:

   > What lights are on?

   or:

   > Turn off the kitchen light.

Once basic operation works, enable optional features one at a time.

For a fuller walkthrough, see [First setup](docs/getting-started/setup.md).

## Where configuration lives

Provider connection details are managed through Home Assistant's integration configuration. Conversation-agent behaviour is managed from the **Extended OpenAI** sidebar panel.

The management interface is organised around:

- **Assistant** — models, API behaviour, prompt/context, conversation, voice and speech
- **Capabilities** — Home Assistant access, Request Rules, Function Tools/Groups and Guest Mode
- **Data & Memory** — memories, Knowledge Library and conversation history
- **Usage & Maintenance** — usage, request debugging, backups, diagnostics and retention

The built-in **Guide** provides concise in-app orientation. The repository documentation under [`docs/`](docs/) is the canonical reference for feature behaviour and edge cases.

## Start with the right guide

| If you want to… | Read |
| --- | --- |
| Understand the main settings | [Configuration](docs/configuration.md) |
| Use Responses API features or a compatible provider | [Responses API](docs/features/responses-api.md) |
| Let the model search the web | [Web Search](docs/features/web-search.md) |
| Manage long conversations | [Conversation continuity](docs/features/conversation-continuity.md) and [Context management](docs/features/context-management.md) |
| Store durable facts/preferences | [Persistent Memory](docs/features/persistent-memory.md) |
| Store short-lived facts that expire | [Temporary Memory](docs/features/temporary-memory.md) |
| Give the agent larger reference material | [Knowledge Library](docs/features/knowledge-library.md) |
| Keep/search previous conversations | [Conversation Archive](docs/features/conversation-archive.md) |
| Handle predictable requests specially | [Request Rules](docs/features/request-rules.md) |
| Add or organise model-callable tools | [Functions](docs/functions/index.md) and [Function Groups](docs/features/function-groups.md) |
| Delay a configured Function Tool safely | [Delayed Function Tools](docs/features/delayed-function-tools.md) |
| Understand entity exposure and permissions | [Home Assistant access](docs/features/home-assistant-access.md) |
| Associate voice requests with individual users | [Voice Identity](docs/features/voice-identity.md) |
| Keep Assist listening for follow-up speech | [Voice follow-ups](docs/features/voice-followups.md) |
| Clean spoken output | [Speech processing](docs/features/speech-processing.md) |
| Reduce satellite volume overnight | [Quiet Hours](docs/features/quiet-hours.md) |
| Restrict visitor access | [Guest Mode](docs/features/guest-mode.md) |
| Inspect provider usage | [Usage statistics](docs/features/usage-statistics.md) |
| Inspect an individual request | [Request debugging](docs/features/request-debugging.md) |
| Refresh model capability metadata | [Model data](docs/features/model-data.md) |
| Back up or restore an agent | [Backup & Restore](docs/features/backup-restore.md) |
| Use AI Task | [AI Task](docs/features/ai-task.md) |
| Migrate from the original integration | [Migration](docs/migration.md) |
| Diagnose a problem | [Troubleshooting](docs/troubleshooting.md) |

## Permissions, privacy and data

Extended OpenAI combines local Home Assistant processing/storage with requests to the AI provider you configure.

Home Assistant Assist exposure defines the normal entity set available to the assistant, but authenticated Home Assistant user permissions still apply to protected actions. Extended OpenAI policy layers such as Guest Mode can make access more restrictive; they cannot grant permissions that Home Assistant denied.

When model processing is needed, relevant prompt/context, request text, available tool definitions, and retrieved memory or Knowledge content may be sent to the configured provider. Provider-side retention and processing are governed by that provider's policies.

Memories, Knowledge Library content, Request Rules, optional archived conversations, usage data, and related integration state are stored locally in Home Assistant. Features that complete a request locally can avoid a provider request for that request.

Full backups can contain private prompts, memories, Knowledge content, archived conversations and usage metadata. Treat exported backup files accordingly.

## Provider compatibility

The integration supports both **Responses API** and **Chat Completions**. For most users, **API mode: Auto** is the recommended starting point.

Some features depend on the selected model, API mode, or provider. For example, hosted Web Search and some multimodal/reasoning capabilities require a compatible Responses implementation. A provider that supports Chat Completions is not automatically compatible with `/v1/responses`.

Model capability metadata can be refreshed independently of an integration release, with bundled metadata retained as the fallback. See [Model data](docs/features/model-data.md).

## Reliability and testing

Contributors and coding agents should run `python scripts/fix.py` before final tests or hand-off. CI checks the committed source with `ruff check custom_components/` and `ruff format --check custom_components/`; it does not apply fixes.

The integration is exercised through unit/integration tests plus genuine Home Assistant acceptance tests. Selected critical flows are also tested through a real browser against Home Assistant, including upgrade-from-a-previous-release scenarios and cross-browser smoke coverage.

Green CI cannot prove that every provider, Home Assistant installation, entity combination, or custom Function Tool will behave identically, but the test suite is designed to cover the real runtime boundaries rather than only isolated helpers.

## Debug logging

If troubleshooting requires integration debug logs, add:

```yaml
logger:
  logs:
    custom_components.extended_openai_conversation_responses: debug
```

> [!CAUTION]
> Debug logs can be verbose and may contain information you do not want to retain. Avoid leaving debug logging enabled permanently.

## Documentation

The full user documentation is in [`docs/`](docs/). Start with:

- [Documentation overview](docs/index.md)
- [Installation](docs/getting-started/installation.md)
- [First setup](docs/getting-started/setup.md)
- [Configuration](docs/configuration.md)
- [Troubleshooting](docs/troubleshooting.md)

## Credits

Extended OpenAI Conversation (Responses) was originally forked from [jekalmin/extended_openai_conversation](https://github.com/jekalmin/extended_openai_conversation).

The upstream project's custom-function architecture and earlier Home Assistant/OpenAI integration work formed the foundation of this project.

AI-assisted development: OpenAI Codex is used in the development of this project.

Thanks to the upstream author and contributors, and to everyone contributing to Home Assistant, HACS, and the OpenAI ecosystem.
