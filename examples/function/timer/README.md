## Objective
- Add a delay to any functions. `spec.parameters.properties.delay` is a reserved parameter which makes a delayed function call. 

<img width="300" src="https://private-user-images.githubusercontent.com/2917984/530385902-7ce3f831-7bb2-40d5-9148-9abe3294134d.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjY3NjU2MjUsIm5iZiI6MTc2Njc2NTMyNSwicGF0aCI6Ii8yOTE3OTg0LzUzMDM4NTkwMi03Y2UzZjgzMS03YmIyLTQwZDUtOTE0OC05YWJlMzI5NDEzNGQucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI1MTIyNiUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTEyMjZUMTYwODQ1WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9NzBkZjNlNDY2NDA2Y2I4ZjU4YjE5N2QyYzEzNGE5NThjNmY1YzEyNGEzODJkNmM5ZTZjNTc3MDI4YmU2YmY3MiZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.mnOFnEv4wgN9vm43WnmRTHgeoxikifcVUSynDDNLEaE">


## Guide
```yaml
- spec:
    name: ...
    description: ...
    parameters:
      type: object
      properties:
        delay: # Add delay parameter to any function spec.
          type: object
          description: Time to wait before execution
          properties:
            hours:
              type: integer
              minimum: 0
            minutes:
              type: integer
              minimum: 0
            seconds:
              type: integer
              minimum: 0
        ...
  function:
    type: ...
```

## Function
### 1. execute_services
```yaml
- spec:
    name: execute_services
    description: Use this function to execute service of devices in Home Assistant.
    parameters:
      type: object
      properties:
        delay:
          type: object
          description: Time to wait before execution
          properties:
            hours:
              type: integer
              minimum: 0
            minutes:
              type: integer
              minimum: 0
            seconds:
              type: integer
              minimum: 0
        list:
          type: array
          items:
            type: object
            properties:
              domain:
                type: string
                description: The domain of the service
              service:
                type: string
                description: The service to be called
              service_data:
                type: object
                description: The service data object to indicate what to control.
                properties:
                  entity_id:
                    type: string
                    description: The entity_id retrieved from available devices. It
                      must start with domain, followed by dot character.
                required:
                - entity_id
            required:
            - domain
            - service
            - service_data
  function:
    type: native
    name: execute_service
```

### 2. add_item_to_list
```yaml
- spec:
    name: add_item_to_list
    description: Add item to a list
    parameters:
      type: object
      properties:
        delay:
          type: object
          description: Time to wait before execution
          properties:
            hours:
              type: integer
              minimum: 0
            minutes:
              type: integer
              minimum: 0
            seconds:
              type: integer
              minimum: 0
        item:
          type: string
          description: The item to be added to the list
        list:
          type: string
          description: the entity id of the list to update
          enum:
            - todo.shopping_list
            - todo.to_do
      required:
      - item
      - list
  function:
    type: script
    sequence:
    - service: todo.add_item
      data:
        item: '{{item}}'
      target:
        entity_id: '{{list}}'
```