# Changelog

## [Unreleased]

## [1.1.2] - 2024-06-19
### Fixed
- Fix config_flow import (MODEL_STRATEGY_AUTO), use public OpenAI exceptions, pin OpenAI 1.x dependency.

## [1.1.0] - 2024-05-28
### Added
- GPT-5 "Thinking" and Responses API flow with automatic reasoning detection, token budgeting, and router forcing support.
- Streaming controls with minimum-character thresholds, proactive speech confirmation, and compatibility fixes for TTS pipelines.
- Memory toolkit including `memory.write`/`memory.search` adapters, sanitisation heuristics, snippet budgeting, and circuit breaker safeguards.
- Expanded options UI covering Responses vs Chat, reasoning effort, proactivity budgets, telemetry controls, and memory service endpoints.
- Updated README guidance and new unit tests for router behaviours, streaming paths, and Responses normalization.
