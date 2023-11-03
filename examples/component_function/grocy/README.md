## Requirement
Assume using [grocy](https://github.com/custom-components/grocy)

## Function
### get_today_chore
```yaml
- spec:
    name: get_today_chore
    description: Use this function to retrieve a list of chores due for today or before.
    parameters:
      type: object
      properties: {}
  function:
    type: template
    value_template: >-
      {% set now_date = now().date() %}
      {% set chores_data = state_attr('sensor.grocy_chores', 'chores') %}

      {% set overdue_chores = chores_data | selectattr('next_estimated_execution_time', 'string') | map(attribute='next_estimated_execution_time') | map('regex_replace', 'T\\d{2}:\\d{2}:\\d{2}', '') | select('lt', now_date | string) | list %}
      {% set chores_due_today = chores_data | selectattr('next_estimated_execution_time', 'string') | map(attribute='next_estimated_execution_time') | map('regex_replace', 'T\\d{2}:\\d{2}:\\d{2}', '') | select('equalto', now_date | string) | list %}
      {% set combined_chores = overdue_chores + chores_due_today %}

      ```csv
      name,last_tracked_time,next_estimated_execution_time
      {% for chore in combined_chores %}
      {{ chore['name'] | replace(",", " ") }},{{ chore['last_tracked_time'] }},{{ chore['next_estimated_execution_time'] }}
      {% endfor -%}
      ```
```

### execute_chore

```yaml
- spec:
    name: execute_chore
    description: Use this function to execute a chore in Home Assistant.
    parameters:
      type: object
      properties:
        chore_id:
          type: string
          description: The ID of the chore to be executed.
      required:
      - chore_id
  function:
    type: script
    sequence:
    - service: script.execute_chore
      data:
        chore_id: "{{ chore_id }}"
```

```yaml
script:
  execute_chore:
    alias: "Execute Chore"
    sequence:
      - service: grocy.execute_chore
        data:
          chore_id: "{{ chore_id }}"
```

### get_inventory_stock
```yaml
- spec:
    name: get_inventory_stock
    description: Use this function to retrieve the inventory entries data.
    parameters:
      type: object
      properties: {}
  function:
    type: template
    value_template: >-
      {% set data = states['sensor.grocy_stock'].attributes['products'] | list %}
      ```csv
      name,id,product_group_id,available_amount,amount_aggregated,amount_opened,amount_opened_aggregated,is_aggregated_amount,best_before_date
      {% for product in data -%}
      {{ product['name'] }},{{ product['id'] }},{{ product['product_group_id'] }},
      {%- if product['available_amount'] == 0 -%}
      {{ product['amount_aggregated'] }},
      {%- else -%}
      {{ product['available_amount'] }},
      {%- endif -%}
      {{ product['amount_aggregated'] }},{{ product['amount_opened'] }},{{ product['amount_opened_aggregated'] }},{{ product['is_aggregated_amount'] }},{{ product['best_before_date'] }}
      {% endfor -%}
      ```
```