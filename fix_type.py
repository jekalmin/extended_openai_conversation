with open('custom_components/extended_openai_conversation/helpers.py', 'r') as f:
    content = f.read()

# Instead of just type: ignore, we'll cast it if possible, but actually `type: ignore` is exactly what HA Core does for this.
