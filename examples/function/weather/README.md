## Objective
- Get current weather and forecasts

<img width="300" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/177f416e-2194-4a10-a3f6-39a94da942ce">
<img width="300" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/3c861f37-3654-4f6b-bcbf-554f12189051">

## Prerequisite
Expose `weather.xxxxx` entity

## Function

### get_attributes
```yaml
- spec:
    name: get_attributes
    description: Get attributes of any home assistant entity
    parameters:
      type: object
      properties:
        entity_id:
          type: string
          description: entity_id
      required:
      - entity_id
  function:
    type: template
    value_template: "{{states[entity_id]}}"
```