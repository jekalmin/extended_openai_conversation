"""Constants for the Extended OpenAI Conversation integration."""

DOMAIN = "extended_openai_conversation"
CONF_PROMPT = "prompt"
DEFAULT_PROMPT = """This smart home is controlled by Home Assistant.
Answer the user's question using a list of available devices.
A list of available devices in this smart home:

```yaml
{% for entity in exposed_entities -%}
- entity_id: {{ entity.entity_id }}
  name: {{ entity.name }}
  state: {{ entity.state }}
  {{- "\n  aliases" + entity.aliases | join(',') if entity.aliases }}
{% endfor -%}
```

If user asks for devices that are not available, answer the user's question about the world truthfully.
If the query requires the current state of device, answer the user's question using current state from the list of available devices. If device is not present in the list, reject the request.
If the query requires a call service, look for the device from the list. If device is not present in the list, reject the request.
If multiple devices are requested, answer at most five devices at a time.
"""
CONF_CHAT_MODEL = "chat_model"
DEFAULT_CHAT_MODEL = "gpt-3.5-turbo"
CONF_MAX_TOKENS = "max_tokens"
DEFAULT_MAX_TOKENS = 150
CONF_TOP_P = "top_p"
DEFAULT_TOP_P = 1
CONF_TEMPERATURE = "temperature"
DEFAULT_TEMPERATURE = 0.5

CONF_FUNCTION_CALLS = "auto"
CONF_FUNCTIONS = [
    {
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
                                "description": """The service data object to indicate what to control. The key "entity_id" is required.""",
                            },
                        },
                        "required": ["domain", "service", "service_data"],
                    },
                }
            },
        },
    }
]
