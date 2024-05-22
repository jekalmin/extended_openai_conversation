## Objective
- Search from Google

## Prerequisite
Needs Google API key and CX code.

## Function

### search_google

```yaml
- spec:
    name: search_google
    description: Execute Google Custom Search to find and refine relevant information. Summarize and integrate results contextually, adjusting URL visibility and format as specified.
    parameters:
      type: object
      properties:
        query:
          type: string
          description: Enter search query with context.
        results_count:
          type: integer
          description: Number of results to retrieve, default is 3.
      required:
      - query
  function:
    type: rest
    resource_template: "https://www.googleapis.com/customsearch/v1?key=[API-KEY]&cx=[CX-CODE]&q={{ query | urlencode }}&num={{ results_count | default(3) }}"
    value_template: >
      {%- set items = value_json['items'] if value_json['items'] is iterable else [] %}
      {%- if items -%}
        {{ items | tojson }}
      {%- else -%}
        No data found or data is not iterable.        
      {%- endif -%}
```
