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




## Services
### extended_openai_conversation.reload
- reload configuration file `<config directory>/extended_openai_conversation/functions.yaml`

| service data  | attribute |	required |	dataType |	description |
|---------------|-----------|-----------|-----------|---------------|


## Customize
### Options
By clicking a button from Edit Assist, Options can be customized.<br/>
Options are same as [OpenAI Conversation](https://www.home-assistant.io/integrations/openai_conversation/) options except for "Maximum function calls per conversation"

"Maximum function calls per conversation" is added to limit the number of function calls in a single conversation.<br/>
(Sometimes function is called over and over again, possibly running into infinite calls)

| Edit Assist                                                                                                                                  | Options                                                                                                                                                                       |
|----------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img width="608" alt="1" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/bb394cd4-5790-4ac9-9311-dbcab0fcca56"> | <img width="590" alt="스크린샷 2023-10-08 오후 2 15 17" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/2d686958-7a9a-4107-9904-eac7c2ffbbb4"> |


### Custom Functions
This is an example of configuration of custom functions.

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

Copy and paste above configuration into "Custom Functions" .

Then you will be able to let OpenAI call your function.

| get_current_weather                                                                                                                                                           | add_item_to_shopping_cart                                                                                                                                                     | 
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img width="391" alt="스크린샷 2023-10-07 오후 7 56 27" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/05e31ea5-daab-4759-b57d-9f5be546bac8"> | <img width="341" alt="스크린샷 2023-10-07 오후 7 54 56" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/89060728-4703-4e57-8423-354cdc47f0ee"> |

Supported function types are following:
  - script
  - template

## DEBUG
In order to monitor logs of API requests and responses, add following config to `configuration.yaml` file

```yaml
logger:
  logs:
    custom_components.extended_openai_conversation: info
```