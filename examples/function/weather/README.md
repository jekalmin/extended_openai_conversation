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

Or if you want to get forecasts (tested with [Met.no](https://www.home-assistant.io/integrations/met) and [Météo France](https://www.home-assistant.io/integrations/meteo_france)):

### get_weather_forecasts
```yaml
- spec:
    name: get_weather_forecasts
    description: Get hourly and daily weather forecasts
    parameters:
      type: object
      properties:
        entity_id:
          type: string
        type:
          type: string
          enum: ["daily", "hourly"]
      required:
      - entity_id
      - type
  function:
    type: script
    sequence:
      - service: weather.get_forecasts
        data:
          type: "{{ type }}"
        target:
          entity_id: "{{ entity_id }}"
        response_variable: forecast_data
      - stop: ""
        response_variable: forecast_data
```
