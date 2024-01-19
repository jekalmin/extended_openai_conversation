## Objective
- Search from Google

## Prerequisite
Needs Google API Key

## Function

### search_google

```yaml
- spec:
    name: search_google
    description: Search Google using the Custom Search API.
    parameters:
      type: object
      properties:
        query:
          type: string
          description: The search query.
      required:
      - query
  function:
    type: rest
    resource_template: "https://www.googleapis.com/customsearch/v1?key=[GOOGLE_API_KEY]&cx=[GOOGLE_PROGRAMMING_SEARCH_ENGINE]:omuauf_lfve&q={{ query }}&num=3"
    value_template: >-
      {% if value_json.items %}
      ```csv
      title,link
      {% for item in value_json.items %}
      "{{ item.title | replace(',', ' ') }}","{{ item.link }}"
      {% endfor %}
      ```
      {% else %}
      No results found,
      {% endif %}
```