## Objective

<img width="300" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/32217f3d-10fc-4001-9028-717b1683573b">


## Function

### get_history
```yaml
- spec:
    name: get_history
    description: Retrieve historical data of specified entities.
    parameters:
      type: object
      properties:
        entity_ids:
          type: array
          items:
            type: string
            description: The entity id to filter.
        start_time:
          type: string
          description: Start of the history period in "%Y-%m-%dT%H:%M:%S%z".
        end_time:
          type: string
          description: End of the history period in "%Y-%m-%dT%H:%M:%S%z".
      required:
      - entity_ids
  function:
    type: composite
    sequence:
      - type: native
        name: get_history
        response_variable: history_result
      - type: template
        value_template: >-
          {% set ns = namespace(result = [], list = []) %}
          {% for item_list in history_result %}
              {% set ns.list = [] %}
              {% for item in item_list %}
                  {% set last_changed = item.last_changed | as_timestamp | timestamp_local if item.last_changed else None %}
                  {% set new_item = dict(item, last_changed=last_changed) %}
                  {% set ns.list = ns.list + [new_item] %}
              {% endfor %}
              {% set ns.result = ns.result + [ns.list] %}
          {% endfor %}
          {{ ns.result }}
```
