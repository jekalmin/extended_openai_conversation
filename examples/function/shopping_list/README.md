## Objective
<img width="300" src="https://github.com/jekalmin/extended_openai_conversation/assets/2917984/89060728-4703-4e57-8423-354cdc47f0ee">

## Prompt
### add the following to prompts and fill in your list entity ids
```For lists: Ensure you are differentiating and using one of the following as the list parameter: todo.shopping_list for modifying the "shopping list" and todo.to_do for modification to the "to-do list" ```


## Function

### add_item_to_shopping_cart
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
