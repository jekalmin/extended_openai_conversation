## Objective

<img width="300" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/55f5fe7e-b1fd-43c9-bce6-ac92e203598f">

## Notice

Before adding automation, I highly recommend set notification on `automation_registered_via_extended_openai_conversation` event and create separate "Extended OpenAI Assistant" and "Assistant"

(Automation can be added even if conversation fails because of failure to get response message, not automation)

| Create Assistant                                                                                                                             | Notify on created                                                                                                                                                              |
|----------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img width="830" alt="1" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/b7030a46-9a4e-4ea8-a4ed-03d2eb3af0a9"> | <img width="1116" alt="스크린샷 2023-10-13 오후 6 01 40" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/7afa3709-1c1d-41d0-8847-70f2102d824f"> |


Copy and paste below configuration into "Functions"

## Function
### add_automation
```yaml
- spec:
    name: add_automation
    description: Use this function to add an automation in Home Assistant.
    parameters:
      type: object
      properties:
        automation_config:
          type: string
          description: A configuration for automation in a valid yaml format. Next line character should be \\n, not \n. Use devices from the list.
      required:
      - automation_config
  function:
    type: native
    name: add_automation
```

