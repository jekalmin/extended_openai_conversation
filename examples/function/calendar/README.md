## Objective
<img width="300" alt="스크린샷 2023-10-31 오후 9 04 56" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/7a6c6925-a53e-4363-a93c-45f63951d41b">

## Function
### get_events
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
