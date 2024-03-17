## Objective

This example provides three functions:

- query_image: generic query image
- camera_snapshot: generic take camera snapshot and store in /media
- camera_query: a combined take camera snapshot and query image

Use the first two functions instead of the combined camera_query
function for more flexiblity as they could be used independantly. You
may need to grant file system access via home assistant configuration
of allowlist_external_dirs to /media or your choosen directory.

## Function

### query_image
```yaml
- spec:
    name: query_image
    description: Get description of items or scene from an image
    parameters:
      type: object
      properties:
        url:
          type: string
          description: path or url for image
      required:
      - url
  function:
    type: composite
    sequence:
      - type: script
        sequence:
          - service: extended_openai_conversation.query_image
            data:
              model: gpt-4-vision-preview
              prompt: What's in this image?
              images:
                - url: "{{url}}"
              max_tokens: 500
              config_entry: YOUR_CONFIG_ENTRY
            response_variable: _function_result
        response_variable: image_result
      - type: template
        value_template: "{{image_result.choices[0].message.content}}"
```

### camera_snapshot
```yaml
- spec:
    name: camera_snapshot
    description: Generate an image from a camera
    parameters:
      type: object
      properties:
        entity_id:
          type: string
          description: Camera entity
        filename:
          type: string
          description: full path and name of file to generate. Please name it as /media/camera_entity_latest.jpg
      required:
      - item
  function:
    type: script
    sequence:
    - service: camera.snapshot
      target:
        entity_id: "{{entity_id}}"
      data:
        filename: '{{filename}}'
```

### camera_query
```yaml
- spec:
    name: camera_query
    description: Get a description of items or scene from a camera
    parameters:
      type: object
      properties:
        entity_id:
          type: string
          description: Camera entity
        filename:
          type: string
          description: full path and name of file to generate. Please name it as /media/camera_entity_latest.jpg
      required:
      - item
  function:
    type: composite
    sequence:
      - type: script
        sequence:
          - service: camera.snapshot
            target:
              entity_id: "{{entity_id}}"
            data:
              filename: '{{filename}}'
          - service: extended_openai_conversation.query_image
            data:
              model: gpt-4-vision-preview
              prompt: What's in this image?
              images:
                - url: "{{filename}}"
              max_tokens: 500
              config_entry: YOUR_CONFIG_ENTRY
            response_variable: _function_result
        response_variable: image_result
      - type: template
        value_template: "{{image_result.choices[0].message.content}}"
```
