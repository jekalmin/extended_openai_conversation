## Objective
- Use file read/write functions to implement features such as persistent memory.


## Functions

### write_to_file
```yaml
- spec:
    name: write_to_file
    description: write the provided content to the specified file. A new file will be created if one does not exist. 
    parameters:
      type: object
      properties:
        filename:
          type: string
          description: a filename consists of 2-32 alphanumerical characters and underscore.
        content:
          type: string
          description: the content to be written.
        open_mode:
          type: string
          description: optional file mode ('w', 'a', 'w+', 'a+'). Defaults to 'w'
        required:
        - filename
        - content
  function:
    type: native
    name: write_to_file
```

### read_from_file
```yaml
- spec:
    name: read_from_file
    description: read text from the specified file.
    parameters:
      type: object
      properties:
        filename:
          type: string
          description: a filename consists of 2-32 alphanumerical characters.
        required:
        - filename
  function:
    type: native
    name: read_from_file
```

## Sample prompts
```
When explicitly requested, help user remember things by appending a new row (fields "timestamp" and "content") using write_to_file function to the file "assistant_memo.csv". Retrieve later by reading from one or more files with the read_from_file function. 

Example: 
User: help me remember that I have taken vitamin today.
(You should call function write_to_file("assistant_memo.csv", "2024-12-28 08:00:05.109809-08:00,\"user has taken vitamin today\"\n", "a")
User: help me remember that I just took vitamin.
(You should call function write_to_file("assistant_memo.csv", "2024-12-28 21:57:01.103210-08:00,\"user just took vitamin\"\n", "a")
User: how many times have I taken vitamin today?
You: Yes, you have taken Vitamin twice today at 8:00 and 21:57 respectively.
```