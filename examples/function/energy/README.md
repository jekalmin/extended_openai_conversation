## Objective
- Get energy statistics

<img width="300" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/04ef6eaa-f0be-4cf2-ae53-b11aecf88c4d">
<img width="300" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/a38e88e2-c5e5-4db9-a7d3-b3ee2cecb8c2">

## Function

### get_energy_statistic_ids
```yaml
- spec:
    name: get_energy_statistic_ids
    description: Get statistics
    parameters:
      type: object
      properties:
        dummy:
          type: string
          description: Nothing
  function:
    type: composite
    sequence:
      - type: native
        name: get_energy
        response_variable: result
      - type: template
        value_template: "{{result.device_consumption | map(attribute='stat_consumption') | list}}"
```
### get_statistics
```yaml
- spec:
    name: get_statistics
    description: Get statistics
    parameters:
      type: object
      properties:
        start_time:
          type: string
          description: The start datetime
        end_time:
          type: string
          description: The end datetime
        statistic_ids:
          type: array
          items:
            type: string
            description: The statistic ids
        period:
          type: string
          description: The period
          enum:
            - day
            - week
            - month
      required:
        - start_time
        - end_time
        - statistic_ids
        - period
  function:
    type: native
    name: get_statistics
```