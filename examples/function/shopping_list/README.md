## Objective
<img width="300" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/89060728-4703-4e57-8423-354cdc47f0ee">

## Function

### add_item_to_list
```yaml
- spec:
    name: add_item_to_list
    description: Add item to a list
    parameters:
      type: object
      properties:
        item:
          type: string
          description: The item to be added to the list
        list:
          type: string
          description: the entity id of the list to update
          enum:
            - todo.shopping_list
            - todo.to_do
      required:
      - item
      - list
  function:
    type: script
    sequence:
    - service: todo.add_item
      data:
        item: '{{item}}'
      target:
        entity_id: '{{list}}'
```

### remove_item_from_list
```yaml
- spec:
    name: remove_item_from_list
    description: Check an item off a list
    parameters:
      type: object
      properties:
        item:
          type: string
          description: The item to be removed from the list
        list:
          type: string
          description: the entity id of the list to update
          enum:
            - todo.shopping_list
            - todo.to_do
      required:
      - item
      - list
  function:
    type: script
    sequence:
    - service: todo.update_item
      data:
        item: '{{item}}'
        status: 'completed'
      target:
        entity_id: '{{list}}'
```

### get_items_from_list
```yaml
- spec:
    name: get_items_from_list
    description: Read back items from a list
    parameters:
      type: object
      properties:
        list:
          type: string
          description: the entity id of the list to update
          enum:
            - todo.shopping_list
            - todo.to_do
      required:
      - list
  function:
    type: script
    sequence:
    - service: todo.get_items
      data:
        status: 'needs_action'
      target:
        entity_id: '{{list}}'
      response_variable: _function_result
```
