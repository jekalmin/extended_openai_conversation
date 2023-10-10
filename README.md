# Extended OpenAI Conversation

Derived from [OpenAI Conversation](https://www.home-assistant.io/integrations/openai_conversation/) with a new feature of calling service added.


## How it works
Extended OpenAI Conversation uses OpenAI API's feature of [function calling](https://platform.openai.com/docs/guides/gpt/function-calling) to call service of Home Assistant.

Since "gpt-3.5-turbo" model already knows how to call service of Home Assistant in general, you just have to let model know what devices you have by [exposing entities](https://github.com/jekalmin/extended_openai_conversation#preparation)

## Installation
1. Copy `extended_openai_conversation` folder into `<config directory>/custom_components`
2. Restart Home Assistant
3. Go to Settings > Devices & Services.
4. In the bottom right corner, select the Add Integration button.
5. Follow the instructions on screen to complete the setup (API Key is required).
    - [Generating an API Key](https://www.home-assistant.io/integrations/openai_conversation/#generate-an-api-key)
6. Go to Settings > Voice Assistants.
7. Click to edit Assistant (named "Home Assistant" by default).
8. Select "Extended OpenAI Conversation" from "Conversation agent" tab.
    <details>

    <summary>guide image</summary>
    <img width="500" alt="스크린샷 2023-10-07 오후 6 15 29" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/0849d241-0b82-47f6-9956-fdb82d678aca">

    </details>

## Preparation
After installed, you need to expose entities from "http://{your-home-assistant}/config/voice-assistants/expose".

## Examples
### 1. Turn on single entity
https://github.com/jekalmin/extended_openai_conversation/assets/2917984/938dee95-8907-44fd-9fb8-dc8cd559fea2

### 2. Turn on multiple entities
https://github.com/jekalmin/extended_openai_conversation/assets/2917984/528f5965-94a7-4cbe-908a-e24f7bbb0a93

### 3. Hook with custom notify function
https://github.com/jekalmin/extended_openai_conversation/assets/2917984/4a575ee7-0188-41eb-b2db-6eab61499a99


## Customize
### Options
By clicking a button from Edit Assist, Options can be customized.<br/>
Options include [OpenAI Conversation](https://www.home-assistant.io/integrations/openai_conversation/) options and two new options. 


- `Maximum Function Calls Per Conversation`: limit the number of function calls in a single conversation.
(Sometimes function is called over and over again, possibly running into infinite loop) 
- `Functions`: A list of mappings of function spec to function.
  - `spec`: Function which would be passed to [functions](https://platform.openai.com/docs/api-reference/chat/create#chat/create-functions) of [chat API](https://platform.openai.com/docs/api-reference/chat/create).
  - `function`: function that will be called.


| Edit Assist                                                                                                                                  | Options                                                                                                                                                                       |
|----------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img width="608" alt="1" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/bb394cd4-5790-4ac9-9311-dbcab0fcca56"> | <img width="591" alt="스크린샷 2023-10-10 오후 10 53 57" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/431e4bc5-87a0-4d7b-8da0-6273f955877f"> |


### Functions
Below is a default configuration of functions.

```yaml
- spec:
    name: execute_services
    description: Use this function to execute service of devices in Home Assistant.
    parameters:
      type: object
      properties:
        list:
          type: array
          items:
            type: object
            properties:
              domain:
                type: string
                description: The domain of the service
              service:
                type: string
                description: The service to be called
              service_data:
                type: object
                description: The service data object to indicate what to control.
                  The key "entity_id" is required. The value of "entity_id" should be retrieved from a list of available devices.
            required:
            - domain
            - service
            - service_data
  function:
    type: predefined
    name: execute_service
```

This is an example of configuration of functions.

#### Example 1.
```yaml
- spec:
    name: get_current_weather
    description: Get the current weather in a given location
    parameters:
      type: object
      properties:
        location:
          type: string
          description: The city and state, e.g. San Francisco, CA
        unit:
          type: string
          enum:
          - celcius
          - farenheit
      required:
      - location
  function:
    type: template
    value_template: The temperature in {{ location }} is 25 {{unit}}
- spec:
    name: add_item_to_shopping_cart
    description: Add item to shopping cart
    parameters:
      type: object
      properties:
        item:
          type: string
          description: The item to be added to cart
      required:
      - item
  function:
    type: script
    sequence:
    - service: shopping_list.add_item
      data:
        name: '{{item}}'
```

Copy and paste above configuration into "Functions".

Then you will be able to let OpenAI call your function.

| get_current_weather                                                                                                                                                           | add_item_to_shopping_cart                                                                                                                                                     | 
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img width="391" alt="스크린샷 2023-10-07 오후 7 56 27" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/05e31ea5-daab-4759-b57d-9f5be546bac8"> | <img width="341" alt="스크린샷 2023-10-07 오후 7 54 56" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/89060728-4703-4e57-8423-354cdc47f0ee"> |

#### Example 2.

In order to accomplish "send it to Line" like [example3](https://github.com/jekalmin/extended_openai_conversation#3-hook-with-custom-notify-function), register a notify function like below.  

```yaml
- spec:
    name: send_message_to_line
    description: Use this function to send message to Line.
    parameters:
      type: object
      properties:
        message:
          type: string
          description: message you want to send
      required:
      - message
  function:
    type: script
    sequence:
    - service: script.notify_all
      data:
        message: "{{ message }}"
```

Supported function types are following:

  - `predefined`: Pre-defined function that is provided by "extended_openai_conversation".
    - Currently `name: execute_service` is only supported.
  - `script`: A list of services that will be called
  - `template`: The value to be returned from function.

## Logging
In order to monitor logs of API requests and responses, add following config to `configuration.yaml` file

```yaml
logger:
  logs:
    custom_components.extended_openai_conversation: info
```