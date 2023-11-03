## Requirement
Assume using [17track](https://www.home-assistant.io/integrations/seventeentrack)

## Function

### get_incoming_packages

```yaml
- spec:
    name: get_incoming_packages
    description: Use this function to retrieve information about incoming packages.
    parameters:
      type: object
      properties: {}
  function:
    type: template
    value_template: >-
      {% set ns = namespace(current_status=None) %}
      {% set statuses = {
          'expired': states('sensor.17track_packages_expired')|int,
          'undelivered': states('sensor.17track_packages_undelivered')|int,
          'delivered': states('sensor.17track_packages_delivered')|int,
          'ready_to_be_picked_up': states('sensor.17track_packages_ready_to_be_picked_up')|int,
          'returned': states('sensor.17track_packages_returned')|int,
          'in_transit': states('sensor.17track_packages_in_transit')|int,
          'not_found': states('sensor.17track_packages_not_found')|int
      } %}
      {% set priority_order = ['expired', 'undelivered', 'delivered', 'ready_to_be_picked_up', 'returned', 'in_transit', 'not_found'] %}
      {% set friendly_status = {
          'expired': '만료',
          'undelivered': '미배송',
          'delivered': '배송 완료',
          'ready_to_be_picked_up': '배송 출발',
          'returned': '반송',
          'in_transit': '배송 중',
          'not_found': '찾을 수 없음'
      } %}
      ```csv
      package status,package name,details
      {%- for status in priority_order %}
        {%- set current_status = status %}
        {%- set current_package_count = statuses[current_status] %}
        {%- if current_package_count > 0 %}
          {%- set package_details = state_attr('sensor.17track_packages_' + current_status, 'packages') %}
          {%- for package in package_details %}
            {{ friendly_status[current_status] }},{{ package.friendly_name }},{{ package.info_text | replace(",", ";") }} 
          {%- endfor %}
        {%- endif %}
      {%- endfor -%}
      ```
```