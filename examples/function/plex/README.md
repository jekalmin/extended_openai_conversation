## Function

### search_plex
```yaml
- spec:
    name: search_plex
    description: Use this function to search for media in Plex.
    parameters:
      type: object
      properties:
        query:
          type: string
          description: The search query to look up media on Plex.
      required:
      - query
      - token
  function:
    type: rest
    resource_template: "https://YOUR.PLEX.SERVER.TLD/search?query={{query}}&X-Plex-Token=YOURPLEXTOKEN"
    value_template: >-
      ```csv
      title,year,director,type,key
      {% for metadata in value_json["MediaContainer"]["Metadata"] %}
        {{ metadata["title"]|replace(",", " ") }},
        {{ metadata["year"] }},
        {{ metadata["Director"][0]["tag"] if metadata["Director"] else "N/A" }},
        {{ metadata["type"] }},
        {{ metadata["key"] }}
      {% endfor -%}
      ```
```

### play_plex_media_in_apple_tv

```yaml
- spec:
    name: play_plex_media_in_apple_tv
    description: Use this function to play Plex media on an Apple TV.
    parameters:
      type: object
      properties:
        key:
          type: string
          description: The key of the media in Plex.
        entity_id:
          type: string
          description: The entity ID of the Apple TV in Home Assistant.
        type:
          type: string
          enum:
          - movie
          - show
          - episode
      required:
      - key
      - entity_id
      - type
  function:
    type: script
    sequence:
    - service: script.play_plex_media_on_apple_tv
      data:
        kind: "{{ kind }}"
        content_id: "{{ content_id }}"
        player: "{{ player }}"
```

```yaml
script:
  play_plex_media_on_apple_tv:
    alias: "Play Plex Media on Apple TV"
    sequence:
    - service: media_player.play_media
      data_template:
        media_content_type: "{{ type }}"
        media_content_id: "plex://MYSERVERID/{{ key }}"
      target:
        entity_id: "{{ entity_id }}"
```