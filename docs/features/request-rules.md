# Request Rules

Request Rules examine text received by Extended OpenAI Conversation before its normal provider request. They provide deterministic local voice shortcuts and AI-routing commands without asking a model to interpret the trigger.

Open **Extended OpenAI > Capabilities > Request Rules**. Rules, groups, matching defaults, and wording alternatives are stored locally per conversation agent and included in that agent's backup.

Request Rules only see text that Home Assistant routes to this conversation agent. If a native intent or sentence-trigger automation handles a sentence first, the rule is not involved.

Rules are checked in the order shown. By default, the first matching rule handles the request. Advanced rules can choose to continue checking later rules. An AI handoff, failed action, denied Guest Mode action, or rule evaluation error stops further matching. A false **Only when** condition simply skips that rule.

## Build a rule

The editor follows the way a command is usually designed:

1. Enter what you will say and choose how it matches.
2. Optionally add **Only when** conditions with Home Assistant's native condition editor.
3. Choose what should happen.
4. Choose what the assistant should say.
5. Open **Advanced matching and action configuration** only when the defaults or friendly action fields are not enough.

## Only when conditions

Rules with no conditions behave exactly as before. For a conditional rule, phrase or sentence-pattern matching runs first. Only if the text matches are its Home Assistant conditions evaluated. A false condition skips that rule and checks later text-matching rules in global priority order. This also applies to fuzzy fallback candidates. The first text match whose conditions pass is the next eligible rule. By default it handles the request and stops matching; a rule with **Continue matching after this rule** enabled can allow later eligible rules to run too.

The **Only when** editor uses Home Assistant's condition selector. For example, an Equals rule for `good night` can require a state condition that `input_boolean.bedtime` is `on`. You can use native nested `and`/`or`/`not`, numeric state, time, template, device, and other conditions offered by Home Assistant. Saved condition structures remain intact when the rule is edited.

If a text-matching condition cannot be evaluated, Request Rules stop safely for that request. No lower-priority rule runs, and the user receives a rule error. Match Preview reports the error. This avoids treating an uncertain higher-priority rule as false and executing another local action.

## Local commands

A **Local command** runs one or more Home Assistant actions or enabled ExtendedOpenAI functions in order. By default it consumes the request locally and does not make an OpenAI/API call. Select an action by its `domain.action` name, then choose an entity, device, or area through Home Assistant's native selectors. Fields published by the selected service, such as brightness, temperature, media, or a select option, appear as friendly controls when Home Assistant provides selector metadata.

Existing `target` and `data` values are preserved when a rule is edited. **Advanced JSON** is a lossless fallback for service data or target keys that the friendly editor does not expose.

If the native script fails or aborts, the remaining actions do not run and the configured failure response is returned. An action with `continue_on_error: true` may recover according to Home Assistant semantics; if the script then completes, the rule succeeds. While Guest Mode is active, the entire sequence is authorized before the first action runs; if one action is unavailable, none run.

Turn on **Continue to AI** when the local sequence should run first and the provider should then answer. **AI input** defaults to **Original request**; choose **Captured value** to send one named Sentence Pattern capture instead. A rule follows this order after Home Assistant runs its native action sequence:

1. A failed or aborted script, including **Stop** with `error: true` or a false top-level condition action, returns the rule's failure response. It never calls AI.
2. A final **Set conversation response** value returns that response, overriding the generic success response. It never calls AI. Home Assistant's latest effective setter wins; parallel setters can be timing dependent.
3. A successful **Stop** consumes the request locally. It returns a prior conversation response if one exists, otherwise the rule's success response.
4. After normal completion without a final conversation response or stop, **Continue to AI** calls the provider once when enabled.
5. Otherwise, the rule returns its success response.

A condition inside `if`, `choose`, `repeat`, or another nested sequence halts only the scope Home Assistant halts. A fatal wait timeout (`continue_on_timeout: false`) follows the failure path; `continue_on_timeout: true` can complete normally. Earlier successful side effects are not rolled back if a later step fails. Native `variables` and service `response_variable` values remain available to later actions in the script, but are not automatically interpolated in the separate Request Rule success response. Only request captures and dedicated Function Tool result aliases are supported there.

### Example: fast local script

- Phrase: `good night`
- Match: **Equals**
- Action: `script.turn_on` targeting the Goodnight script
- Response: `Good night`

## AI routing

An **AI routing** rule can override the model, reasoning effort, or both. Matching and request flow are separate choices: **Equals**, **Starts with**, **Ends with**, **Contains**, and **ExtendedOpenAI sentence pattern** decide only whether a rule matches.

Use **Continue to AI** to choose what happens after a routing rule matches:

- **On** applies the routing settings and sends the selected AI input to the provider. The default is the unchanged original request. **This request only** applies the selected model/reasoning only to that provider call; **Rest of this conversation** applies it to that call and later requests in the same conversation.
- **Off** treats the phrase as a standalone routing command. ExtendedOpenAI acknowledges it locally and does not send that command to the provider. Because there is no provider request to modify, standalone commands use **Rest of this conversation** scope.

Routing precedence for a provider request is:

`single-request override > conversation override > configured agent default`

Conversation overrides use the current conversation/continuity identity and expire with it. They do not change the saved agent configuration.

A reset follows the same explicit flow choice. With **Continue to AI** on and **This request only**, the configured agent defaults apply to that one provider call while any saved conversation override remains in place for the next normal request. A **Rest of this conversation** reset clears the saved conversation override.

Existing routing rules created before this option was added retain their previous behaviour when first loaded: Equals and Sentence pattern rules remain standalone, while Starts with, Ends with, and Contains rules continue to the provider. Once normalized, that choice is stored explicitly and no longer changes when the match type changes.

Reasoning effort is validated against the model that will actually receive it, including a model supplied by another active conversation override. Model-specific values are supported where the model allows them; for example, GPT-6 Astra routing can use `xhigh` and `max` in addition to the normal reasoning levels.

## Matching modes

### Text matching

**Equals**, **Starts with**, **Ends with**, and **Contains** are case-insensitive and normalize punctuation and whitespace. Rules can inherit the defaults or customize:

- **Normalize word forms** handles conservative English forms such as `light/lights`.
- **Wording alternatives** map different ways of saying the same thing to a main phrase. The seeded alternatives preserve the previous built-in behavior, such as `switch on` to `turn on` and `television` to `tv`. Alternatives can be added, edited, and removed. Ambiguous duplicate phrases are rejected.
- **Fuzzy matching** tolerates small speech-recognition differences only after strict matching fails. Conservative, Normal, and Tolerant correspond to progressively lower thresholds. Sensitivity is unavailable when fuzzy matching is off.

Strict matching always wins over fuzzy matching. Deterministic rules are evaluated from top to bottom in the order shown on the Request Rules screen. The first eligible strict match handles the request by default, regardless of match type. A rule with **Continue matching** on can let later strict matches run. Fuzzy matching is considered only after strict candidates are exhausted. When no strict rule matched, the first fuzzy candidate is chosen by its existing score-based ranking; continuation then checks only later-priority fuzzy rules in global order.

Eligibility is checked before a candidate runs: a candidate whose **Only when** conditions are false is skipped, while conditions are never evaluated for rules whose text did not match. An eligible rule then follows the normal stop-or-continue behavior described above.

For example, if an earlier **Contains** rule and a later **Equals** rule both match the same request, the earlier rule wins. Move the Equals rule above it when that more specific case should take priority.

### ExtendedOpenAI sentence patterns

Choose **ExtendedOpenAI sentence pattern** when parts of a command can vary. The editor provides **Optional**, **Choice**, **Variable**, and **Number range** insertion helpers; they build the same raw syntax shown below, which remains fully editable. This is a small syntax owned by ExtendedOpenAI; it is not a promise of compatibility with Home Assistant/Hassil grammar. Patterns are parsed and compiled when rules are loaded or saved, then matched by a bounded finite-state matcher.

Sentence patterns are case-insensitive but otherwise use their own exact grammar path. Fuzzy matching, wording alternatives, and word-form normalization do not apply.

Literal spelling and constrained choices use the same Unicode and whitespace normalization as the input. Sentence-ending punctuation (`.`, `!`, `?`, `。`, and `؟`, including fullwidth equivalents) may follow a complete command. Punctuation explicitly written in a pattern still has to match; punctuation in the middle of a command is literal.

Supported syntax:

| Syntax | Meaning | Example |
| --- | --- | --- |
| plain text | Required literal text | `turn the light off` |
| `[text]` | Optional text | `turn [the] light off` |
| `[one|two]` | Optional alternatives | `[please|kindly] turn it off` |
| `(one|two)` | Required alternatives | `(turn|switch) the light off` |
| `{name}` | Non-empty free-text capture | `remember {fact}` |
| `{name=one|two}` | Constrained capture | `set {room=kitchen|bedroom} lights on` |
| `{name=min..max}` | Integer capture in an inclusive range | `set brightness to {level=0..100}` |
| `\` | Escape a syntax character so it is literal | `say \(hello\)` |

Optional and alternative blocks may be embedded inside a word, for example `light[s]` matches both `light` and `lights`.

A constrained capture both validates and captures the selected value. For example:

```text
[please ](turn|switch) {room=kitchen|bedroom|office} light[s] on
```

`please switch bedroom lights on` captures `room = bedroom`. A value outside the listed choices does not match.

Escapes also work inside constrained choices: `{value=a\|b|c}` accepts the literal values `a|b` and `c`, not separate `a` and `b` choices. Choices that become identical after Unicode, whitespace, and case normalization are rejected.

Numeric captures accept integer digits (including a leading `+` or `-` where the configured range allows it) and validate the value without expanding the range into thousands of alternatives:

```text
set brightness to {level=0..100}
```

`set brightness to 73` captures `level = 73`; `set brightness to 173` does not match.

Free-text captures may contain multiple words. The matcher chooses the shortest capture that still allows the complete pattern to succeed. For example:

```text
add {item} to {list_name}
```

`add semi skimmed milk to weekly shopping` captures `item = semi skimmed milk` and `list_name = weekly shopping`.

Free-text captures must be separated by required literal or constrained text. Ambiguous patterns such as `do {first} {second}` are rejected. A pattern also needs at least one required literal, constrained capture, or numeric capture, so a bare `{anything}` cannot intercept every request.

All phrase variants in one rule must expose the same capture names. Captured values can be selected as **Value from request** in supported action/function fields or inserted into responses with simple `{name}` substitution. This substitution is deterministic and does not execute Jinja.

Named expansion syntax such as `<device>`, Hassil permutations using `;`, predefined Hassil slot lists, arbitrary regular expressions, lookarounds, backreferences, and repetition operators are not part of the ExtendedOpenAI sentence-pattern language.

### Matching bounds

Sentence-pattern matching is deliberately bounded. The grammar is compiled to a finite-state representation instead of recursively backtracking through every optional/alternative combination, so repeated optional terms do not create exponential work.

The current safety limits are:

- 200 characters per saved phrase;
- 25 phrase variants per rule;
- 500 Request Rules per agent;
- 8 levels of sentence-pattern nesting;
- 8 captures per pattern, of which at most 4 may be free-text captures;
- 32 choices per constrained capture, with each choice limited to 80 characters;
- 512 compiled states per pattern;
- 50,000 enabled compiled sentence-pattern states per agent;
- 2,048 characters and 256 words in a live or Match Preview input, checked before and (for sentence matching) after Unicode normalization and case folding; and
- a shared aggregate matcher-work budget across every sentence pattern considered for one request.

The same parser, compiled matcher, input bounds, winner-selection rules, and aggregate work budget are used by live requests and Match Preview. Matching itself runs outside Home Assistant's main event loop. Each match reads one complete configuration snapshot, even if rules or defaults are being saved concurrently. Compiled patterns are immutable and reused through a bounded cache.

Deterministic candidates are evaluated in displayed rule order. Matching stops after an ordinary matched rule or an AI handoff; a continued rule lets later candidates run. An unresolved sentence-pattern candidate that could determine the next match causes evaluation to fail safely if it exceeds the work budget.

If a live request is larger than the matching limit or the aggregate work budget cannot safely complete, **no Request Rule action runs** and the original request continues through the normal AI path. The matcher never treats an interrupted higher-priority rule as a failed match and then executes a lower-priority local action. Match Preview instead reports the limit as an error.

If an existing stored sentence pattern is no longer accepted by the documented grammar or its safety bounds, it is preserved rather than silently deleted. The Request Rules screen marks that rule inactive with an actionable diagnostic so it can be edited or removed.

Inactive legacy rules can be disabled and repaired individually without blocking unrelated changes. Backups preserve them too; restoring a backup applies the same activation limits and diagnostics as loading stored rules. New or re-enabled rules must satisfy the current grammar and limits.

## Function actions

A rule can call an enabled Function Tool directly without sending the request to the AI provider. The function selector comes from the current agent's configured tools. Selecting one shows common string, number, integer, boolean, enum, and simple-array inputs from its existing schema. Each input can be a fixed value or a captured request value.

For example, use `Show {entity_id} attributes`, select the existing `get_attributes` function, and set `entity_id` to **Value from request → entity_id**. Direct execution uses the same implementation, argument validation, current enabled state, Guest Mode policy, and entity-access checks as a model-initiated call.

To use a Function Tool's return value, set an optional **Result alias** on its action step. For `get_device_battery(device={device})`, name the result `battery`; an object result such as `{"name":"Kitchen tablet","level":62}` can then be used in the local success response as `{battery.name} is at {battery.level}%`. A scalar string, number, boolean, or null can be referenced as `{battery}`. Nested object paths and simple numeric array indexes such as `{battery.items.0.name}` are supported. Missing paths fail the local rule and use its failure response; placeholders are never spoken unresolved.

Request captures and Function results stay separate. `{device}` remains a request capture, while `{battery.device}` reads a field from the Function result. A Function result cannot overwrite a capture. Each result-producing call needs a unique editable alias, even when the same Function Tool is called twice. Aliases use letters, digits, and underscores, start with a letter or underscore, and cannot use reserved names such as `request`, `conversation`, or `system`. The editor suggests an alias from the Function Tool name; the saved step ID remains stable when you rename the alias. References to an earlier alias in simple `{alias.path}` form are updated on rename. Removing or moving a producing step ahead of its consumers is rejected on save.

Function execution success is determined by the execution outcome, not the truthiness of its result: `false`, `0`, empty string, and null are valid values. Retained results are limited to 16 KiB and eight levels of nesting. The safe Match Preview lists the Function Tool and alias but never runs it or invents a result. Live Test can run it after the normal explicit confirmation. **Function results are not automatically sent to the AI provider**, including when Continue to AI is on; the provider receives the selected AI input.

## Groups and priority

Groups help organize and filter rules. They do not affect priority. Use **Show** to view all rules, Ungrouped rules, or one group. Filtered rules keep their global priority numbers. Switch to **All rules** to change priority. Search and group filters only change what is visible.

In **All rules**, drag a card or use **Move up**, **Move down**, **Move to top**, and **Move to bottom** to change the one global sequence. Reordering is unavailable while search or a group filter hides rules. Changing a rule's group or deleting a group preserves the global order; deleted-group rules become Ungrouped. Groups and order are included in backups. The server checks the current revision when saving a reorder, so an older tab cannot silently overwrite a newer order.

### Continue matching

Open a rule's advanced settings to enable **Continue matching after this rule**. After a local sequence succeeds, later matching rules can run in priority order. A later rule with this option off stops normally. A routing command can also apply its model or reasoning choice and let later rules run. If later routing rules change the same setting, the later matching choice wins.

**Continue to AI** always ends rule checking. The provider is called once, after successful local actions where applicable. A failed local action or Function Tool, Guest Mode denial, or condition evaluation error stops the chain safely. Function results remain within their own rule. Match Preview shows the rules that would match and where checking would stop, without running actions or calling the provider.

### AI input

For any rule with **Continue to AI** on, **Original request** sends the user's full request, as existing rules do. **Captured value** sends only the chosen named value from a Sentence Pattern trigger. For example, `deep think {question}` can send `why is the sky blue?` from `deep think why is the sky blue?`.

The selected capture must be present on every trigger variant and every possible match of those variants. If editing a trigger removes it, the editor keeps the selection visible and requires you to fix the trigger or switch back to **Original request**. Match Preview shows the exact user text that would be sent to the provider. It does not run local actions or call the provider.

### Rule Sharing

Open **Rule Sharing** at the bottom of Request Rules to export all rules, one group, or selected rules. A rule pack contains only selected rules, their organizational groups, and the effective matching settings owned by those rules. It does not contain provider credentials, Function Tool definitions, memories, knowledge, archive data, unrelated agent settings, or agent-wide wording alternatives. The pack has an explicit format and version for compatibility.

Import first shows **Review Rule Pack**, including rules, actions, conditions, Function Tool references, and unavailable resources. Nothing executes during review. Confirming appends the rules after your existing rules in their exported relative order and leaves every imported rule **disabled**. Newly created rules start enabled and can be toggled from the Request Rules list. Enable and test each rule after checking its dependencies. Groups with the same name are reused; other groups receive new internal IDs. Rule Sharing never changes the destination agent's wording alternatives; if a shared rule's author relies on additional alternative phrases, those remain a separate setup choice. A conflicting or invalid pack is rejected without replacing existing rules. Full agent backup and selective transfer remain separate capabilities.

## Request Rules compared with native automations

Use a Request Rule when:

- the phrase belongs specifically to this Extended OpenAI conversation agent;
- a stable command should bypass the AI/API call;
- the phrase should change model or reasoning routing; or
- you want the rule and its response managed with the agent.

Use a native Home Assistant sentence-trigger automation when:

- the command should work independently of this integration or conversation agent;
- it needs automation triggers, conditions, templates, variables, traces, or modes;
- it should be owned alongside the rest of your Home Assistant automations; or
- native Assist handling should take priority before text reaches an AI agent.

The two approaches can coexist, but their sentence-pattern languages are separate. Avoid giving both the same effective phrase unless their routing priority is intentional.

## Security and limits

Request Rules are not semantic understanding and do not learn from history. They do not use embeddings, a general thesaurus, automatic suggestions, or speaker authentication. Use Equals or a narrow sentence pattern for locks, alarms, garage-style covers, and other sensitive actions. The editor warns about obvious sensitive action domains when tolerant text matching is active.

Home Assistant service validation still applies. Guest Mode authorization uses the same backend enforcement as model-initiated Home Assistant tools and never partially executes a rejected sequence.
