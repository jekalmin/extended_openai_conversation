## Objective
- Get attributes of entity

<img width="300" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/5994c7a0-1370-4924-bed8-d2e77ec1d11d">


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