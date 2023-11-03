## Objective

## Requirement
Assume using [ytube_music_player](https://github.com/KoljaWindeler/ytube_music_player)

## Function

### search_music
```yaml
- spec:
    name: search_music
    description: Use this function to search music
    parameters:
      type: object
      properties:
        query:
          type: string
          description: The query
      required:
      - query
  function:
    type: composite
    sequence:
    - type: script
      sequence:
      - service: ytube_music_player.search
        data:
          entity_id: media_player.ytube_music_player
          query: "{{ query }}"
    - type: template
      value_template: >-
        media_content_type,media_content_id,title
        {% for media in state_attr('sensor.ytube_music_player_extra', 'search') -%}
          {{media.type}},{{media.id}},{{media.title}}
        {% endfor%}
```