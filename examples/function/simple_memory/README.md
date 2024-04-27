## Objective

Implements a simple short term, long-term, and archive memory for the LLM to use.

## About


Giving the LLM a memory allows it to store and retrieve information across conversations.
I've found it great for a number of use cases including
 - allowing it to remember how to perform certain operations that differ from its training
 - remembering facts about people or things, thereby reducing infomation you encode in your prompt
 - can be used to set (remember) reminders that it should recall during conversation later
 - reduce amount of tokens consumed; allowing it to use cheaper models while maintaining better results

## Approach

This is a functional proof of concept split into a set of function calls that invoke a sensor via event triggers.

The memories are implemented as dictionaries hidden inside a HA sensor. The LLM is told that these are dictornaries, and should encode a concise description in the key, followed by more information in the value.

The memories are separated into short-term, long-term and archive.  You don't need to have all three, but so far I've found it helps me and the LLM manage memories.  That is,

 - short term memory is intended for items we shouldn't need to remember for long.
 - long term memory is intended for items we need to remember, such as reoccuring memories, but are unlikely to needed forever.
 - the archive memory is intended for things you want to keep forever.

The intent with these three is that the keys, and perhaps values, for both short-term and long-term should fit in token space.  These could be passed in with the prompt or you could let the LLM probe them on its own.

The archive memory can only be referenced by key.  In this case the LLM will query for a list of keys, and can then retrieve a memory based upon a  key it likes.

### Notes

 - If you want to keep it simple to start you could drop the archive or even long-term memory.
 - It might be worth playing around with the function description fields below to help instruct the LLM on how to use (this may reduce prompt text).
 - This is intended as a proof of concept. I'll probably implement it as a python module in the future as this will be cleaner and can use a proper db backend.

# Installation

There are three parts to install
 - prompt text
 - function specs
 - Home Assistant template sensor

## Prompt

I add this to my prompt.  Although I haven't tried, it could be made much more concise or perhaps avoided altogether by encoding information in the sensor or function call descriptions.

```text
sensor.memory is our memory storage bank. Use the memory sensor to store and retrieve short_term, long_term or archived memories.
The short_term and long_term memories can be read via the sensor's attributes.
Consult the memory sensor before answering queries as it may contain relevant information.
Please check short_term memories now for anything we should act upon. You can let me know we should discuss those as part of the response to my query.
Please also store in the appropriate memory any information you may need to recall later and include details. You can use it to remember summaries of conversations or any important events, reminders, and details.
If any short_term memories such as reminders that have past that we have discussed, feel free to clean them up.
Keep the short_term and long_term with a limited set of entries that are good to manage the day-to-day and need instant recall. For other memories, use the chronicle.

check memory first before answering.
```

## Function

Put this in your extended openAI configuration:


```yaml
- spec:
    name: set_memory
    description: Use this function to remember something in the form of a key value pair, that will appear in future prompt.
    parameters:
      type: object
      properties:
        key:
          type: string
          description: short key to refer to this memory
        value:
          type: string
          description: the information to remember
        memory_location:
          type: string
          description: set it to match the attribute where you want it stored, for example set to long_term or short_term and it will place the memory under that attribute.
        due_timestamp:
          type: string
          default: ""
          description: optional iso format timestamp that you can use to indicate an important time related to this memory.
      required:
      - key
      - value
      - memory_location
  function:
    type: script
    sequence:
    - event: set_memory
      event_data:
        key: "{{key}}"
        value: "{{value}}"
        memory_location: "{{memory_location}}"
        due_timestamp: "{{due_timestamp}}"
- spec:
    name: forget_memory
    description: Use this function to delete a memory stored as an attribute in the memory sensor by its key.
    parameters:
      type: object
      properties:
        key:
          type: string
          description: short key to refer to this memory
        memory_location:
          type: string
          description: set it to match the attribute where it is stored, for example set to long_term or short_term and it will remove the memory under that attribute.
      required:
      - key
      - memory_location
  function:
    type: script
    sequence:
    - event: remove_memory
      event_data:
        key: "{{key}}"
        memory_location: "{{memory_location}}"
- spec:
    name: retrieve_memories_from_chronicle
    description: Use this function to retrieve a subset of memories by their key value. The result goes into retrieved_chronicle_memories attribute of memory sensor. To use this function first look at chronicle_available_keys or similar to get a list of keys and choose appropriate ones for your query.
    parameters:
      type: object
      properties:
        keys:
          type: string
          description: a string containing a jinja2 list of one or more keys that are present in chronicle_available_keys. Each key to refers to a memory.
      required:
      - keys
  function:
    type: script
    sequence:
    - event: retrieve_from_chronicle_by_keys
      event_data:
        key_list: "{{keys}}"

- spec:
    name: engrave_chronicle_memory
    description: Use this function to remember something in the form of a key value pair in our chronicle memory archive.
    parameters:
      type: object
      properties:
        key:
          type: string
          description: short key to refer to this memory
        value:
          type: string
          description: the information to remember
      required:
      - key
      - value
  function:
    type: script
    sequence:
    - event: engrave_memory
      event_data:
        key: "{{key}}"
        value: "{{value}}"
- spec:
    name: erase_chronicle_memory
    description: Use this function to erase a memory stored in the chronicle memory archive by passing its key
    parameters:
      type: object
      properties:
        key:
          type: string
          description: short key to refer to this memory
      required:
      - key
  function:
    type: script
    sequence:
    - event: erase_memory
      event_data:
        key: "{{key}}"
```


## configuration.yaml

Put this in your Home Assistant yaml as a template sensor.

### search_google

```yaml
template:
  - trigger:
      - platform: event
        event_type: engrave_memory
      - platform: event
        event_type: erase_memory
    sensor:
      - unique_id: 3b4c2b87-2cf1-4912-8791-0a17a2780fff
        name: Chronicle Cache Backing Store
        state: >
          {% set current = this.attributes.get('backing_store', {}) %}
          {{ current.keys() | length }}
        attributes:
          description: archive store for memories. Use engrave_memory and erase_memory to store. Do not look at the attributes of this sensor as they could be very big. Instead access via the memory sensor's retrieve_from_chronicle_by_keys and its retrieved_chronicle_memories attribute.
          backing_store: >
            {%- macro store_chronical(destination,event) -%}
              {%- set my_dict = this.attributes.get(destination, {}) -%}
              {%- if event.event_type == 'engrave_memory' -%}
                {%- set new = {event.data.key: {'value': event.data.value, 'last_written_timestamp': now().isoformat()}} -%}
                {{ dict(my_dict, **new) }}
              {%- elif event.event_type == 'erase_memory' -%}
                {{ dict(my_dict.items() | rejectattr('0', 'eq', event.data.key)) }}
              {%- else -%}
                {{my_dict}}
              {%- endif -%}
            {%- endmacro -%}
            {{store_chronical('backing_store',trigger.event)}}

  - trigger:
      - platform: event
        event_type: set_memory
      - platform: event
        event_type: remove_memory
      - platform: event
        event_type: retrieve_from_chronicle_by_keys
    sensor:
      - unique_id: 9d37c876-eaec-4c43-a4c0-c493847161a2
        name: Memory
        state: >
          {% set current = this.attributes.get('short_term', {}) %}
          {{ current.keys() | length }}
        attributes:
          description: store memories as a key value pair. Add or remove here by calling the set_memory event service with optional memory_location can be set to short_term (default) or long_term. Memories from both of these are immediately available via the attributes. You can optionally set due_timestamp in isoformat. Short-term memory is for tasks and reminders that are relevant for the day or coming week—things you need to act on soon. Long-term memory holds information that's important over the next few months or has no specific due date but you'll need to recall eventually. The chronicle memory archives significant events, detailed information, and milestones you want to preserve for years or reflect on in the future, without cluttering your active reminders and tasks. Use the chronicle memory for important events, milestones, or detailed information that you want to preserve over the long term and may not need to recall immediately but is significant for historical reference or future reflection. To check for useful topics or memories in the chronical memory you must first consult the chronicle_available_keys attribute to know what key or keys to retrieve.
          # note from 2023.4 can define these macros globally to reduce duplication
          short_term: >
            {%- macro store_memory(destination,event,destination_default) -%}
              {%- set current = this.attributes.get(destination, {}) -%}
              {%- if event.data.get('memory_location', destination_default) == destination -%}
                {%- if event.event_type == 'set_memory' -%}
                  {%- if event.data.get('due_timestamp', false) -%}
                    {%- set new = {event.data.key: {'value': event.data.value, 'last_written_timestamp': now().isoformat(), 'due_timestamp': event.data.due_timestamp}} -%}
                  {%- else -%}
                    {%- set new = {event.data.key: {'value': event.data.value, 'last_written_timestamp': now().isoformat()}} -%}
                  {%- endif -%}
                  {{ dict(current, **new) }}
                {%- elif event.event_type == 'remove_memory' -%}
                  {{ dict(current.items() | rejectattr('0', 'eq', event.data.key)) }}
                {%- else -%}
                  {{current}}
                {%- endif -%}
              {%- elif current is defined and current is mapping -%}
                {{current}}
              {%- endif -%}
            {%- endmacro -%}
            {{ store_memory('short_term',trigger.event,'short_term') }}
          long_term: >
            {%- macro store_memory(destination,event,destination_default) -%}
              {%- set current = this.attributes.get(destination, {}) -%}
              {%- if event.data.get('memory_location', destination_default) == destination -%}
                {%- if event.event_type == 'set_memory' -%}
                  {%- if event.data.get('due_timestamp', false) -%}
                    {%- set new = {event.data.key: {'value': event.data.value, 'last_written_timestamp': now().isoformat(), 'due_timestamp': event.data.due_timestamp}} -%}
                  {%- else -%}
                    {%- set new = {event.data.key: {'value': event.data.value, 'last_written_timestamp': now().isoformat()}} -%}
                  {%- endif -%}
                  {{ dict(current, **new) }}
                {%- elif event.event_type == 'remove_memory' -%}
                  {{ dict(current.items() | rejectattr('0', 'eq', event.data.key)) }}
                {%- else -%}
                  {{current}}
                {%- endif -%}
              {%- elif current is defined and current is mapping -%}
                {{current}}
              {%- endif -%}
            {%- endmacro -%}
            {{store_memory('long_term',trigger.event,'short_term')}}
          chronicle_available_keys: >
            {%- macro get_keys(destination) -%}
              {%- set backing = state_attr('sensor.chronicle_cache_backing_store',destination) -%}
              {%- if backing is defined and backing is mapping -%}
                {{ backing.keys() | list }}
              {%- else -%}
                {}
              {%- endif -%}
            {%- endmacro -%}
            {{ get_keys('backing_store') }}
          retrieved_chronicle_memories: >
            {%- macro get_memories_by_keys(my_dict, keys) -%}
              {%- set ns = namespace() %}
              {%- set ns.results = [] %}
              {%- for key in keys if key in my_dict -%}
                {%- set ns.results = ns.results + [(key,my_dict[key])] -%}
              {%- endfor -%}
              {{ dict.from_keys(ns.results) }}
            {%- endmacro -%}
            {%- set current = this.attributes.get('retrieved_chronical_memories', {}) -%}
            {%- set backing = state_attr('sensor.chronicle_cache_backing_store','backing_store') -%}
            {%- if backing is defined and backing is mapping and trigger.event.event_type == 'retrieve_from_chronicle_by_keys' -%}
              {%- set keys = trigger.event.data.get('key_list', []) -%}
              {{ get_memories_by_keys(backing,keys) }}
            {%- else -%}
              {{current}}
            {%- endif -%}
```