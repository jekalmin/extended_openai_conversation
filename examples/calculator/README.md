## Objective
- Calculate a simple math expression and return the calculation result as a number.


## Functions

### write_to_file
```yaml
- spec:
    name: calculate
    description: calculate a math expression and return the calculation result as a number.
    strict: true
    parameters:
      type: object
      additionalProperties: false
      properties:
        expression:
          type: string
          description: a legal Python expression that contains only common math operators (+, -, *, /, **, %), numbers, and brackets.
      required:
        - expression
  function:
    type: native
    name: calculate
```

## Example
### Prompt
...
When the inquiry implies math calculation (e.g., "how much in total did I make over the past week?"), always call function calculate to get the result. You can do that by formulate the calculation as a Python math expression and evaluate it by calling function "calculate". Do not ask user to confirm if calculation is needed.
```
### Sample Conversation 1
```
User:
What's 247 divided by 4, and then powered by 5?

Assistant:
The result of 247 divided by 4, and then raised to the power of 5, is approximately 897,810,767.58.
```

### Sample Conversation 2
```
User:
How much milk did the baby drink today?

Assistant:
I found the following entries in the memo regarding how much the baby drank today:

- 130 ml at 00:30
- 125 ml at 05:00
- 110 ml at 13:00
- 125 ml at 06:00
- 130 ml at 21:00

I added these amounts together:

130 + 125 + 110 + 125 + 130 = 620 ml.

That's how I reached the total of 620 milliliters.
```