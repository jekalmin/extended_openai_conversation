# Custom functions

Custom functions extend the conversation agent with tools beyond its included Home Assistant capabilities.

You do **not** need to define a custom function just to control an ordinary exposed Home Assistant entity. Start by exposing entities to Assist and using the included Functions configuration.

Custom functions are for cases where the model needs a new action or data source.

## When to create one

Examples include:

- wrapping a Home Assistant script as a natural-language tool
- returning calculated or templated data
- calling an external REST API
- scraping a web page
- chaining several operations together
- performing a carefully constrained database query

## How a function is structured

A function entry has two main parts:

1. **`spec`** — the tool schema given to the model: its name, description, parameters, and required fields.
2. **`function`** — how Home Assistant should implement the tool call.

A minimal template example:

```yaml
- spec:
    name: describe_room
    description: Get a short description of a room from Home Assistant data.
    parameters:
      type: object
      properties:
        room:
          type: string
          description: The Home Assistant area name.
      required:
        - room
  function:
    type: template
    value_template: >-
      The requested room is {{ room }}.
```

The schema should describe only inputs the model genuinely needs to choose.

If a Function Tool is edited, disabled, deleted, or removed from an available Function Group while a provider request is outstanding, a call from that old request is rejected. EOAI does not bind arguments generated for the old definition to a new implementation with the same name. The next request advertises the current definition. HA-owned tools keep their request-round schema and still pass through Home Assistant's live permission and schema checks before execution.

## Optional function groups

Large function collections can be organized into **Always available** and **Load when needed** groups from **Capabilities → Functions**. On-demand groups initially send only a compact name and description; the model loads their complete schemas through the existing tool loop when a task needs them. Function YAML remains unchanged, and ungrouped functions keep the previous always-available behaviour.

See [Function groups](../features/function-groups.md) for lifecycle, compatibility, UI, diagnostics, and token-use details.

## Function types

The integration supports several implementation types:

- **native** — built-in functionality implemented by the integration
- **script** — runs a Home Assistant action sequence
- **template** — returns a templated value
- **rest** — requests data from an HTTP endpoint
- **scrape** — extracts data from a web page
- **composite** — chains multiple function implementations
- **sqlite** — performs a database query
- **bash** — runs explicitly enabled shell commands with Home Assistant privileges
- **read_file**, **write_file**, **edit_file** — access bounded files under allowed directories

See [Function types and examples](function-types.md), [Native Function Tools](native-details.md), [File Function Tools](file-tools.md), [Bash Function Tool](bash.md), [SQLite Function Tool](sqlite.md), and [Scrape Function Tool](scrape.md) for details.

Use **Add LLM Tools** to select capabilities supplied by Home Assistant or installed
services, including registered MCP APIs. These are live references, with the same
enabled controls and Function Groups, rather than editable Function YAML. Bulk
addition saves only the individual tools selected now; future source tools are not
automatically exposed. See [Home Assistant LLM Tools](home-assistant-llm-tools.md).

## Prefer ordinary Home Assistant features where possible

Before writing a large custom function, consider whether the cleanest solution is:

1. create a Home Assistant script with normal UI/YAML tools
2. expose that script or wrap it with a small function definition
3. let the assistant supply only the parameters that need natural-language interpretation

This usually keeps the Home Assistant logic easier to test and avoids embedding too much automation logic inside an LLM tool schema.

## Security

Treat function definitions as permissions granted to the model.

- expose only actions and data that are necessary
- validate model-provided values before using them in sensitive operations
- avoid giving a model unrestricted database or network access when a constrained tool will work
- use Home Assistant's existing permission/exposure mechanisms where possible

SQLite deserves particular care: read-only access prevents writes, but unrestricted generated queries can still read more information than intended. Prefer fixed or constrained queries when privacy matters.

## Reserved delay parameter

Function specifications can use the reserved `delay` parameter to schedule execution after a model-supplied delay.

Use this only for tools where delayed execution is appropriate and understandable to the user.

## Existing examples

The repository's [`examples`](https://github.com/conorod1992/extended_openai_conversation/tree/develop/examples) directory contains additional working patterns. Treat examples as starting points and adapt entity IDs, services, validation, and permissions to your own Home Assistant installation.
