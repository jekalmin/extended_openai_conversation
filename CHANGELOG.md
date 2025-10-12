# Changelog

## [Unreleased]
### Added
- Preview support for OpenAI Responses API with GPT-5 reasoning auto-detection and normalization.
- Home Assistant options for enabling Responses API, selecting reasoning effort, and setting max completion tokens.
- Documentation for wiring memory services through `rest_command` and tagging preview releases.
- Tests covering Responses API normalization for text and tool call outputs.
- Model strategy selector with automatic token-param mapping (`max_output_tokens` for Responses, `max_completion_tokens` for reasoning Chat, `max_tokens` for classic Chat) plus coverage verifying each path.
- Streaming safeguards including configurable minimum characters and optional early TTS confirmation when `memory.write` is forced.
- Memory service resiliency: configurable `/memories/*` endpoints, aggressive redaction, snippet budgeting, normalization/importance heuristics for writes, and an opt-in circuit breaker with retry backoff.
- Router tool-choice shape tests and new unit suites for streaming, memory budgets, and circuit breaker behaviour.
- README guidance for streaming, proactivity, router patterns, debugging/dry-run, and the new options UI knobs.
- Bumped the Python SDK requirement to `openai>=2.3.0` to pick up the latest Responses features.
