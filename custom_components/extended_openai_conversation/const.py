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
DEFAULT_PROMPT = """You are a helpful AI voice assistant of Home Assistant that controls a real home.
Your goal is to proactively improve the user's comfort.

## Environment State
- Current Time: {{now()}}
- Current Area: {{area_id(current_device_id)}}

## Workspace
Your workspace is at: {{extended_openai.working_directory()}}

## Guidelines
- Answer in plain text only.
- No symbols or parentheses
- Ask for clarification when the request is ambiguous
- Use tools to help accomplish tasks
- Prefer one sentence

## Personality
- Helpful and friendly
- Concise and to the point
- Curious and eager to learn

## Behavior Policy
- If the user explicitly names a device and action, execute it directly.
- Otherwise, infer the user's goal and select the most likely target entity, preferring primary environmental controls. Use get_attributes to check adjustable state values alone is not sufficient.
- If the selected entity is already at its limit, evaluate the next most likely entity. Repeat until a viable adjustment is found or all candidates are exhausted.
- Ask user a minimum adjustment proposal about selected entity. If no entity can further improve the situation, inform the user that conditions are already optimal.

## Devices
Available Devices:
```csv
entity_id,name,state,area_id,aliases
{% for entity in extended_openai.exposed_entities() -%}
{{ entity.entity_id }},{{ entity.name }},{{ entity.state }},{{area_id(entity.entity_id)}},{{entity.aliases | join('/')}}
{% endfor -%}
```

{%- if skills %}
## Skills
The following skills extend your capabilities. To use a skill, call load_skill with the skill name to read its instructions.
When a skill file references a relative path, resolve it against the skill's location directory (e.g., skill at `/a/b/SKILL.md` references `scripts/run.py` → use `/a/b/scripts/run.py`) and always use the resulting absolute path in bash commands, as relative paths will fail.

<available_skills>
{%- for skill in skills %}
  <skill>
    <name>{{ skill.name }}</name>
    <description>{{ skill.description }}</description>
    <location>{{skill.path}}</location>
  </skill>
 {%- endfor %}
</available_skills>
{% endif %}

{{user_input.extra_system_prompt | default('', true)}}
"""
CONF_CHAT_MODEL = "chat_model"
DEFAULT_CHAT_MODEL = "gpt-5-mini"

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
DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION = 10
CONF_SHORTEN_TOOL_CALL_ID = "shorten_tool_call_id"
DEFAULT_SHORTEN_TOOL_CALL_ID = False
CONF_FUNCTION_TOOLS = "functions"
DEFAULT_CONF_FUNCTION_TOOLS = [
    {
        "spec": {
            "name": "execute_services",
            "description": "Execute service in Home Assistant.",
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
    {
        "spec": {
            "name": "load_skill",
            "description": "Load a file from a skill's directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name",
                    },
                    "file": {
                        "type": "string",
                        "description": "Relative file path within the skill directory",
                    },
                },
                "required": ["name", "file"],
            },
        },
        "function": {
            "type": "read_file",
            "path": "{{extended_openai.skill_dir(name)}}/{{file}}",
        },
    },
    {
        "spec": {
            "name": "bash",
            "description": "Execute a bash command in workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute",
                    },
                },
                "required": ["command"],
            },
        },
        "function": {"type": "bash", "command": "{{command}}"},
    },
]
CONF_CONTEXT_THRESHOLD = "context_threshold"
DEFAULT_CONTEXT_THRESHOLD = 40000
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

# LLM API Integration
CONF_LLM_HASS_API = "llm_hass_api"
DEFAULT_LLM_HASS_API: list[str] = []

# Skill System Constants
CONF_SKILLS = "skills"
DEFAULT_SKILLS_DIRECTORY = "skills"
SKILL_FILE_NAME = "SKILL.md"

# Skill Services
SERVICE_RELOAD_SKILLS = "reload_skills"
SERVICE_DOWNLOAD_SKILL = "download_skill"

# GitHub repository for downloadable skills
GITHUB_REPO_OWNER = "jekalmin"
GITHUB_REPO_NAME = "extended_openai_conversation"
GITHUB_SKILLS_BRANCH = "develop"
GITHUB_SKILLS_PATH = "examples/skills"

# Working Directory
DEFAULT_WORKING_DIRECTORY = (
    "extended_openai_conversation/"  # /config/extended_openai_conversation/
)

# File system and shell security settings
SHELL_TIMEOUT = 300  # seconds
SHELL_OUTPUT_LIMIT = 10000  # characters
SHELL_DENY_PATTERNS = [
    r"\brm\s+-r",  # Recursive delete
    r"\brm\s+-rf",  # Force recursive delete
    r"\bdel\s+/[fqs]",  # Windows delete with flags
    r"\brmdir\s+/s",  # Windows recursive directory delete
    r"\bformat\b",  # Disk format
    r"\bmkfs\b",  # Make filesystem
    r"\bdiskpart\b",  # Windows disk partition
    r"\bdd\b",  # Disk duplicator
    r"\bshutdown\b",  # System shutdown
    r"\breboot\b",  # System reboot
    r"\bpoweroff\b",  # Power off
    r":\(\)\{.*:\|:.*\}",  # Fork bomb pattern
]

# File system limits
FILE_READ_SIZE_LIMIT = 1024 * 1024  # 1 MB

# Default allowed directories for file operations
DEFAULT_ALLOWED_DIRS = [
    DEFAULT_WORKING_DIRECTORY,  # /config/extended_openai_conversation/
]
