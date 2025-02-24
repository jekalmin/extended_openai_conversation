## Objective
This function is used to set a preset mode on a fan entity. Within this example the preset mode can be set to `off`, `low` or `high`.

## Function

### set_fan_preset_mode
```yaml
- spec:
    name: set_fan_preset_mode
    description: Use this function to set the preset mode of a fan to "off", "low" or "high".
    parameters:
      type: object
      properties:
        entity_id:
          type: string
          description: entity_id of the fan
        preset_mode:
          type: string
          description: preset mode you want to set
      required:
      - entity_id
      - preset_mode
  function:
    type: script
    sequence:
    - service: fan.set_preset_mode
      target: 
        entity_id: "{{ entity_id }}"
      data:
        preset_mode: "{{ preset_mode }}"
```