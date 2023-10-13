"""Constants for the Extended OpenAI Conversation integration."""

DOMAIN = "extended_openai_conversation"
DEFAULT_NAME = "Extended OpenAI Conversation"
EVENT_AUTOMATION_REGISTERED = "automation_registered_via_extended_openai_conversation"
CONF_PROMPT = "prompt"
DEFAULT_PROMPT = """This is smart home is controlled by Home Assistant.
Answer the user's question using a list of available devices in one or two sentences in everyday language.
A list of available devices in this smart home:

```csv
entity_id,name,state,aliases
{% for entity in exposed_entities -%}
{{ entity.entity_id }},{{ entity.name }},{{entity.state}},{{entity.aliases | join('/')}}
{% endfor -%}
```

If user asks for devices that are not available, answer the user's question about the world truthfully.
If the query requires the current state of device, answer the user's question using current state from the list of available devices. If device is not present in the list, reject the request.
If the query requires a call service, look for the device from the list. If device is not present in the list, reject the request.
Use comma separated everyday language rather than list structure to answer, and do not restate question.
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
            "name": "execute_services",
            "description": "Use this function to execute service of devices in Home Assistant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "list": {
                        "type": "array",
                        "items": {
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
                                    "description": """The service data object to indicate what to control. The key "entity_id" is required. The value of "entity_id" should be retrieved from the list of available devices.""",
                                },
                            },
                            "required": ["domain", "service", "service_data"],
                        },
                    }
                },
            },
        },
        "function": {"type": "native", "name": "execute_service"},
    }
]
