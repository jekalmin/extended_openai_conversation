## Objective
- Map user_id to friendly user name string

When the option to pass the current user to OpenAI via the user
message context is enabled, we actually pass the user_id rather than a
friendly user name as OpenAI has limitations on characters it accepts.
This function can be used to resolve the user's name, without
limitation on acceptable characters.

## Function

### get_user_from_user_id
```yaml
- spec:
    name: get_user_from_user_id
    description: Retrieve users name from the supplied user id hash
    parameters:
      type: object
      properties:
        user_id:
          type: string
          description: user_id
      required:
      - user_id
  function:
    type: native
    name: get_user_from_user_id
```