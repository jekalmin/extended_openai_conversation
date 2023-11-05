## Objective
<img width="300" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/16dc4ca0-c823-4dfe-a2b7-1ba7623acc70">

## Function

### send_message_to_messenger
```yaml
- spec:
    name: send_message_to_messenger
    description: Use this function to send message to messenger.
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
    - service: notify.{YOUR_MESSENGER}
      data:
        message: "{{ message }}"
```