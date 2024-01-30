

## Objective
- Call service via area_id

<img width="300" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/78a5c829-a6b4-4f76-8c68-9fe0ccdca2e8">


## Function

### execute_services
```yaml
- spec:
    name: execute_services
    description: Execute service of devices in Home Assistant.
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
                description: The domain of the service.
              service:
                type: string
                description: The service to be called
              service_data:
                type: object
                description: The service data object to indicate what to control.
                properties:
                  entity_id:
                    type: array
                    items:
                      type: string
                      description: The entity_id retrieved from available devices. It must start with domain, followed by dot character.
                  area_id:
                    type: array
                    items:
                      type: string
                      description: The id retrieved from areas. You can specify only area_id without entity_id to act on all entities in that area
            required:
            - domain
            - service
            - service_data
  function:
    type: native
    name: execute_service
```