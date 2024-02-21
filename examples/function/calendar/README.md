## Objective
<img width="300" alt="스크린샷 2023-10-31 오후 9 04 56" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/7a6c6925-a53e-4363-a93c-45f63951d41b">

## Function
### 1. get_events
```yaml
- spec:
    name: get_events
    description: Use this function to get list of calendar events.
    parameters:
      type: object
      properties:
        start_date_time:
          type: string
          description: The start date time in '%Y-%m-%dT%H:%M:%S%z' format
        end_date_time:
          type: string
          description: The end date time in '%Y-%m-%dT%H:%M:%S%z' format
      required:
      - start_date_time
      - end_date_time
  function:
    type: script
    sequence:
    - service: calendar.get_events
      data:
        start_date_time: "{{start_date_time}}"
        end_date_time: "{{end_date_time}}"
      target:
        entity_id:
        - calendar.[YourCalendarHere]
        - calendar.[MoreCalendarsArePossible]
      response_variable: _function_result
```

### 2. create_event
```yaml
- spec:
    name: create_event
    description: Adds a new calendar event.
    parameters:
      type: object
      properties:
        summary:
          type: string
          description: Defines the short summary or subject for the event.
        description:
          type: string
          description: A more complete description of the event than the one provided by the summary.
        start_date_time:
          type: string
          description: The date and time the event should start.
        end_date_time:
          type: string
          description: The date and time the event should end.
        location:
          type: string
          description: The location
      required:
      - summary
  function:
    type: script
    sequence:
      - service: calendar.create_event
        data:
          summary: "{{summary}}"
          description: "{{description}}"
          start_date_time: "{{start_date_time}}"
          end_date_time: "{{end_date_time}}"
          location: "{{location}}"
        target:
          entity_id: calendar.[YourCalendarHere]
```