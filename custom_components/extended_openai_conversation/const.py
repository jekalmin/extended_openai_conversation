"""Constants for the Extended OpenAI Conversation integration."""

DOMAIN = "extended_openai_conversation"
DEFAULT_NAME = "Extended OpenAI Conversation"
EVENT_AUTOMATION_REGISTERED = "automation_registered_via_extended_openai_conversation"
CONF_PROMPT = "prompt"
DEFAULT_PROMPT = """I want you to act as smart home manager of Home Assistant.
I will provide information of smart home along with a question, you will truthfully make correction or answer using information provided in one sentence in everyday language.

Current Time: {{now()}}

Available Devices:
```csv
entity_id,name,state,aliases
{% for entity in exposed_entities -%}
{{ entity.entity_id }},{{ entity.name }},{{ entity.state }},{{entity.aliases | join('/')}}
{% endfor -%}
```

The current state of devices is provided in available devices.
Use the execute_service function only for requested action, not for current states.
Do not execute service without user's confirmation.
Do not restate or appreciate what user says, rather make a quick inquiry.
"""
CONF_CHAT_MODEL = "chat_model"
DEFAULT_CHAT_MODEL = "gpt-3.5-turbo"
CONF_MAX_TOKENS = "max_tokens"
DEFAULT_MAX_TOKENS = 150
CONF_TOP_P = "top_p"
DEFAULT_TOP_P = 1
CONF_TEMPERATURE = "temperature"
DEFAULT_TEMPERATURE = 0.5
CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION = "max_function_calls_per_conversation"
DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION = 1
CONF_FUNCTIONS = "functions"
DEFAULT_CONF_FUNCTIONS = [
    {
        "spec": {
            "name": "execute_service",
            "description": "Use this function to execute a service of devices in Home Assistant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "The domain of the service",
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
                                "type": "string",
                                "description": "The entity_id retrieved from available devices. It must start with domain, followed by dot character.",
                            }
                        },
                        "required": ["entity_id"],
                    },
                },
                "required": ["domain", "service", "service_data"],
            },
        },
        "function": {"type": "native", "name": "execute_service_single"},
    }
]
CONF_BASE_URL = "base_url"
DEFAULT_CONF_BASE_URL = "https://api.openai.com/v1"
