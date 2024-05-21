## Objective
<img width="300" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/d5c9e0db-8d7c-4a7a-bc46-b043627ffec6">

## Prompt
Add following text in your prompt
````
Youtube Channels:
```csv
channel_id,channel_name
UCLkAepWjdylmXSltofFvsYQ,BANGTANTV
```
````

## Function
### play_youtube
#### webostv
```yaml
- spec:
    name: play_youtube
    description: Use this function to play Youtube.
    parameters:
      type: object
      properties:
        video_id:
          type: string
          description: The video id.
      required:
      - video_id
  function:
    type: script
    sequence:
    - service: webostv.command
      data:
        entity_id: media_player.{YOUR_WEBOSTV}
        command: system.launcher/launch
        payload:
          id: youtube.leanback.v4
          contentId: "{{video_id}}"
    - delay:
        hours: 0
        minutes: 0
        seconds: 10
        milliseconds: 0
    - service: webostv.button
      data:
        entity_id: media_player.{YOUR_WEBOSTV}
        button: ENTER
```
#### Apple TV
```yaml
- spec:
    name: play_youtube_on_apple_tv
    description: Use this function to play YouTube content on a specified Apple TV.
    parameters:
      type: object
      properties:
        kind:
          type: string
          enum:
            - video
            - channel
            - playlist
          description: The type of YouTube content.
        content_id:
          type: string
          description: ID of the YouTube content (can be videoId, channelId, or playlistId).
        entity_id:
          type: string
          description: entity_id of Apple TV.
      required:
      - kind
      - content_id
      - entity_id
  function:
    type: script
    sequence:
    - service: script.play_youtube_on_apple_tv
      data:
        kind: "{{ kind }}"
        content_id: "{{ content_id }}"
        player: "{{ player }}"
```

```yaml
script:
  play_youtube_on_apple_tv:
    alias: "Play YouTube on Apple TV"
    sequence:
    - service: media_player.play_media
      data_template:
        media_content_type: url
        media_content_id: >-
          {% if kind == 'video' %}
            youtube://www.youtube.com/watch?v={{content_id}}
          {% elif kind == 'channel' %}
            youtube://www.youtube.com/channel/{{content_id}}
          {% else %} 
            youtube://www.youtube.com/playlist?list={{content_id}}
          {% endif %}
      target:
        entity_id: "{{ entity_id }}"
```

#### Android TV
```yaml
- spec:
    name: play_youtube_on_android_tv
    description: Use this function to play YouTube content on a specified Android TV.
    parameters:
      type: object
      properties:
        kind:
          type: string
          enum:
            - video
            - channel
            - playlist
          description: The type of YouTube content.
        content_id:
          type: string
          description: ID of the YouTube content (can be videoId, channelId, or playlistId).
        player:
          type: string
          description: media_player entity.
      required:
      - kind
      - content_id
      - player
  function:
    type: script
    sequence:
    - service: script.play_youtube_on_android_tv
      data:
        kind: "{{ kind }}"
        content_id: "{{ content_id }}"
        player: "{{ player }}"
```

```yaml
script:
  play_youtube_on_android_tv:
    alias: "Play YouTube on Android TV"
    sequence:
    - service: remote.turn_on
      data:
        activity: >-
          {% if kind == 'video' %}
            https://www.youtube.com/watch?v={{content_id}}
          {% elif kind == 'channel' %}
            https://www.youtube.com/channel/{{content_id}}
          {% else %}  {# playlist kind #}
            https://www.youtube.com/playlist?list={{content_id}}
          {% endif %}
      target:
        entity_id: "{{ player }}"
```

### get_recent_youtube
```yaml
- spec:
    name: get_recent_youtube_videos
    description: Use this function to get recent videos of youtube.
    parameters:
      type: object
      properties:
        channel_id:
          type: string
          description: The channel id of Youtube
      required:
      - channel_id
  function:
    type: rest
    resource_template: "https://www.youtube.com/feeds/videos.xml?channel_id={{channel_id}}"
    value_template: >-
      ```csv
      video_id,title
      {% for item in value_json["feed"]["entry"] %}
        {{item["yt:videoId"]}},{{item["title"][0:10]}}
      {% endfor -%}
      ```
```

### search_youtube
- Replace "YOUROWNSUPERSECRETYOUTUBEAPIV3KEY" with your API Key

```yaml
- spec:
    name: search_youtube
    description: Use this function to search for YouTube videos, channels, or playlists based on a query.
    parameters:
      type: object
      properties:
        query:
          type: string
          description: The search query to look up on YouTube.
        type:
          type: string
          enum:
            - video
            - channel
            - playlist
          default: video
          description: The type of content to search for on YouTube.
      required:
      - query
      - type
  function:
    type: rest
    resource_template: "https://www.googleapis.com/youtube/v3/search?part=snippet&q={{query}}&type={{type}}&key={YOUROWNSUPERSECRETYOUTUBEAPIV3KEY}"
    value_template: >-
      ```csv
      kind,id,title
      {% for item in value_json["items"] %}
        {{item["id"]["kind"]|replace("youtube#", "")}},{{item["id"][type + "Id"]}},{{item["snippet"]["title"]|replace(",", " ")|truncate(50, True, "...")}}
      {% endfor -%}
      ```
```
