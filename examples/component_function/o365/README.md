## Requirement
Assume using [o365](https://github.com/PTST/O365-HomeAssistant)

## Function
### get_email_inbox
```yaml
- spec:
    name: get_email_inbox
    description: Use this function to retrieve the list of emails from the inbox.
    parameters:
      type: object
      properties: {}
  function:
    type: template
    value_template: >-
      {% set data = states['sensor.inbox'].attributes['data'] | list %}
      ```csv
      subject,received,to,sender,has_attachments,importance,is_read,body
      {% for email in data -%}
      "{{ email['subject'] }}","{{ email['received'] }}","{{ email['to'] | join(', ') }}","{{ email['sender'] }}",{{ email['has_attachments'] }},{{ email['importance'] }},{{ email['is_read'] }},"{{ email['body'] | replace('\n', ' ') | replace('"', '\\"') }}"
      {% endfor -%}
      ```
```