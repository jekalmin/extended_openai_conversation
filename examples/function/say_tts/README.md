## Objective

- say_tts will say a message on any text to speech device

## Function

### say_tts
```yaml
- spec:
    name: say_tts
    description: Say message on a text to speech device
    parameters:
      type: object
      properties:
        message:
          type: string
          description: message you want to say
        device:
          type: string
          description: entity_id of media_player tts device
      required:
      - message
      - device
  function:
    type: script
    sequence:
    - service: tts.cloud_say
      data:
        entity_id: "{{device}}"
        message: "{{message}}"
```
