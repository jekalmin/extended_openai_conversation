"""Constants for the Extended OpenAI Conversation integration."""

DOMAIN = "extended_openai_conversation"
DEFAULT_NAME = "Extended OpenAI Conversation"
DEFAULT_CONVERSATION_NAME = "Extended OpenAI Conversation"
DEFAULT_AI_TASK_NAME = "Extended OpenAI AI Task"

CONF_ORGANIZATION = "organization"
CONF_BASE_URL = "base_url"
DEFAULT_CONF_BASE_URL = "https://api.openai.com/v1"
CONF_API_VERSION = "api_version"
CONF_SKIP_AUTHENTICATION = "skip_authentication"
DEFAULT_SKIP_AUTHENTICATION = False
CONF_API_PROVIDER = "api_provider"
API_PROVIDERS = [
    {"key": "openai", "label": "OpenAI"},
    {"key": "azure", "label": "Azure OpenAI"},
]
DEFAULT_API_PROVIDER = API_PROVIDERS[0]["key"]

EVENT_AUTOMATION_REGISTERED = "automation_registered_via_extended_openai_conversation"
EVENT_CONVERSATION_FINISHED = "extended_openai_conversation.conversation.finished"

CONF_PROMPT = "prompt"
DEFAULT_PROMPT = """You are a voice assistant for Home Assistant.

Answer in plain text only.
Respond naturally as a voice assistant.
Prefer a single sentence; use up to 2-3 sentences only when truly necessary.
Do not use parentheses or symbolic notation; integrate clarifications naturally using words.

For smart home interactions, follow this decision flow strictly:

0. Action classification
   Distinguish between two types of actions:
   A. Information retrieval
      - These do NOT change any device state
      - Execute immediately when user intent is clear
      - Only ask for clarification if the request is genuinely ambiguous
   B. State-changing actions
      - These DO change device state
      - Execute immediately when user explicitly specifies the device and the exact action with clear values
      - Require confirmation only when the request is ambiguous or lacks specific values
      - If you have already proposed an action and the user responds with a specification or refinement, treat this as explicit confirmation and execute immediately

1. Intent understanding
   Determine whether the user is requesting information retrieval or a state-changing action.
   Consider conversation context: if you recently proposed an action, the user's response may be confirming, refining, or rejecting that proposal.

2. Immediate execution
   Execute immediately when:
   - User requests information retrieval with clear intent
   - User explicitly specifies both the device and the exact action or target value for state-changing actions
   - User responds to your proposal with a clear confirmation or specification
   After successful execution, provide brief confirmation and stop.

3. Use provided state information intelligently
   The current state of all devices is already provided in the CSV tables below.
   For information available in the provided CSV:
   - Always use this information directly
   - Do NOT use tools to retrieve information that is already provided
   For information NOT available in the provided CSV:
   - If your response requires additional data beyond what is provided in the CSV, use available tools to retrieve that information
   - Only retrieve additional information when necessary

4. Context-aware proposal logic
   When the user's intent requires clarification or lacks specific values:
   a) Examine the provided CSV to identify devices relevant to the user's intent and their current states
   b) Determine if the intent can be satisfied by changing device states shown in the CSV:
      - If the required action is a simple state change but the target is ambiguous, propose a specific option
   c) If the relevant devices are already in an appropriate state for the intent, or if proposing a meaningful adjustment requires knowing current parameter values:
      - Retrieve the relevant adjustable parameters using available tools
      - Use the current parameter values to propose a contextually appropriate adjustment
      - The proposal should be relative to the current value, not an arbitrary target
      - If parameters are already at their limits for the user's goal, inform the user
   d) Propose one minimal and reasonable adjustment based on complete information
   e) Always end with a confirmation question; never trigger execution

5. State reference in confirmation questions
   When asking for confirmation:
   - Determine whether to include specific numeric values based on everyday familiarity:
     * If the unit or value is something people routinely use in daily conversation and can intuitively understand without specialized knowledge, include the number
     * If the unit is technical, abstract, or rarely discussed in everyday settings, use relative descriptive language instead without mentioning specific values
   - For binary states: Omit the current state as the proposed action implies it
   - Keep confirmation questions concise and natural
   - Propose a single concrete action and await explicit user approval

When referring to the smart home state,
use only the information provided below or retrieved through allowed tools.

For general knowledge questions not related to the home,
answer truthfully using internal knowledge only.

Current Time: {{now()}}
Current Area: {{area_id(current_device_id)}}

An overview of the areas and the available devices:
{%- set area_entities = namespace(mapping={}) %}
{%- for entity in extended_openai.exposed_entities() %}
    {%- set current_area_id = area_id(entity.entity_id) or "etc" %}
    {%- set entities = (area_entities.mapping.get(current_area_id) or []) + [entity] %}
    {%- set area_entities.mapping = dict(area_entities.mapping, **{current_area_id: entities}) -%}
{%- endfor %}

{%- for current_area_id, entities in area_entities.mapping.items() %}

  {%- if current_area_id == "etc" %}
  Etc:
  {%- else %}
  {{area_name(current_area_id)}}:
  {%- endif %}
```csv
    entity_id,name,state,aliases
    {%- for entity in entities %}
    {{ entity.entity_id }},{{ entity.name }},{{ entity.state }},{{ entity.aliases | join('/') }}
    {%- endfor %}
```
{%- endfor %}

{{user_input.extra_system_prompt | default('', true)}}
"""
CONF_CHAT_MODEL = "chat_model"
DEFAULT_CHAT_MODEL = "gpt-5-mini"

MODEL_PARAMETER_SUPPORT = (
    {"pattern": r"^gpt-5-(mini|nano)", "unsupported_params": {"top_p"}},
)

MODEL_TOKEN_PARAMETER_SUPPORT = (
    {
        "pattern": r"(^|-)(gpt-4o|gpt-5|o1|o3|o4)",
        "token_param": "max_completion_tokens",
    },
)
DEFAULT_TOKEN_PARAM = "max_tokens"
CONF_MAX_TOKENS = "max_tokens"
DEFAULT_MAX_TOKENS = 500
CONF_TOP_P = "top_p"
DEFAULT_TOP_P = 1
CONF_TEMPERATURE = "temperature"
DEFAULT_TEMPERATURE = 0.5
CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION = "max_function_calls_per_conversation"
DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION = 3
CONF_FUNCTIONS = "functions"
DEFAULT_CONF_FUNCTIONS = [
    {
        "spec": {
            "name": "execute_services",
            "description": "Execute service of devices in Home Assistant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "delay": {
                        "type": "object",
                        "description": "Time to wait before execution",
                        "properties": {
                            "hours": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "minutes": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "seconds": {
                                "type": "integer",
                                "minimum": 0,
                            },
                        },
                    },
                    "list": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "domain": {
                                    "type": "string",
                                    "description": "The domain of the service.",
                                },
                                "service": {
                                    "type": "string",
                                    "description": "The service to be called",
                                },
                                "service_data": {
                                    "type": "object",
                                    "description": "The service data object to indicate what to control.",
                                    "properties": {
                                        "entity_id": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "description": "The entity_id retrieved from available devices. It must start with domain, followed by dot character.",
                                            },
                                        },
                                        "area_id": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "description": "The id retrieved from areas. You can specify only area_id without entity_id to act on all entities in that area",
                                            },
                                        },
                                    },
                                },
                            },
                            "required": ["domain", "service", "service_data"],
                        },
                    },
                },
            },
        },
        "function": {"type": "native", "name": "execute_service"},
    },
    {
        "spec": {
            "name": "get_attributes",
            "description": "Get attributes of entity or multiple entities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "array",
                        "description": "entity_id of entity or multiple entities",
                        "items": {"type": "string"},
                    }
                },
                "required": ["entity_id"],
            },
        },
        "function": {
            "type": "template",
            "value_template": "```csv\nentity,attributes\n{%for entity in entity_id%}\n{{entity}},{{states[entity].attributes}}\n{%endfor%}\n```",
        },
    },
]
CONF_CONTEXT_THRESHOLD = "context_threshold"
DEFAULT_CONTEXT_THRESHOLD = 13000
CONTEXT_TRUNCATE_STRATEGIES = [{"key": "clear", "label": "Clear All Messages"}]
CONF_CONTEXT_TRUNCATE_STRATEGY = "context_truncate_strategy"
DEFAULT_CONTEXT_TRUNCATE_STRATEGY = CONTEXT_TRUNCATE_STRATEGIES[0]["key"]

# Service Tier options (for GPT-5 models)
CONF_SERVICE_TIER = "service_tier"
DEFAULT_SERVICE_TIER = "flex"
SERVICE_TIER_OPTIONS = ["auto", "default", "flex", "priority"]

# Reasoning Effort options (for o1, o3, o4, gpt-5 models)
CONF_REASONING_EFFORT = "reasoning_effort"
DEFAULT_REASONING_EFFORT = "low"
REASONING_EFFORT_OPTIONS = ["low", "medium", "high"]

SERVICE_QUERY_IMAGE = "query_image"

CONF_PAYLOAD_TEMPLATE = "payload_template"

# Advanced Options
CONF_ADVANCED_OPTIONS = "advanced_options"
DEFAULT_ADVANCED_OPTIONS = False

# Model-specific parameter configurations
# Default configuration for standard models (gpt-4, gpt-4o, etc.)
DEFAULT_MODEL_CONFIG = {
    "supports_top_p": True,
    "supports_temperature": True,
    "supports_max_tokens": True,
    "supports_max_completion_tokens": False,
    "supports_reasoning_effort": False,
    "supports_service_tier": False,
}

# Pattern-based model configurations
# Each entry: {"pattern": regex_string, "config": config_dict}
# Patterns are matched in order; first match wins
MODEL_CONFIG_PATTERNS = [
    # Reasoning models (o1, o3, o4, gpt-5, etc.)
    {
        "pattern": r"^o[1-4]|^gpt-5",
        "config": {
            "supports_top_p": False,
            "supports_temperature": False,
            "supports_max_tokens": False,
            "supports_max_completion_tokens": True,
            "supports_reasoning_effort": True,
            "supports_service_tier": True,
        },
    },
]

# AI Task default options (simpler than conversation - no prompt, just model/token settings)
DEFAULT_AI_TASK_OPTIONS = {
    CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
    CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
    CONF_ADVANCED_OPTIONS: DEFAULT_ADVANCED_OPTIONS,
}

# Skill System Constants
CONF_SKILLS = "skills"
DEFAULT_SKILLS_DIRECTORY = "skills"
SKILL_FILE_NAME = "SKILL.md"

# Skill Services
SERVICE_RELOAD_SKILLS = "reload_skills"
