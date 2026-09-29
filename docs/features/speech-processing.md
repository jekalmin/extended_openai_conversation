# Speech processing

Configure spoken-output cleanup from **Extended OpenAI → Assistant → Speech**.

Speech processing creates a cleaner version of the assistant response for text-to-speech without replacing the original model response stored in conversation history, archives, ChatLog and later conversation context.

## Built-in cleanup

The built-in cleanup can remove content that is useful on screen but awkward when spoken, including:

- Markdown links and citation-style text
- formatting markers
- bare URLs

It works with Responses, Chat Completions and compatible providers. It also keeps track of text across streamed chunks, so a Markdown link or URL split across several pieces can still be removed before it is spoken.

Home Assistant currently uses the same progressive stream for live visual updates and TTS. This means the live on-screen stream also receives the speech-safe text, while the original model response is still retained for history and future conversation context.

For Responses API replies, structured URL citation information is still retained with the original response even when the spoken version is cleaned.

## Custom replacements

Advanced replacement rules use Python regular expressions and run in order after the full response is complete. Built-in Markdown and bare-URL cleanup runs first, so replacements see the cleaned text. Turn off **Remove bare URLs** if a custom rule needs to match a URL.

```yaml
- pattern: '\\[[0-9]+\\]'
  replacement: ''
- pattern: '\\bHA\\b'
  replacement: 'Home Assistant'
```

Because a custom regular expression may depend on text that has not arrived yet, configuring any custom replacement disables progressive TTS for that response. Extended OpenAI waits for the completed answer, applies the normal cleanup, and then runs the custom replacements.

Custom processing has safety limits. If a saved rule is invalid, takes too long, fails, receives too much input, or produces excessive output, Extended OpenAI falls back to the cleaned speech text from before the custom replacements. It will not speak a partially modified result.

Use **Preview spoken text** in the Speech section to check the completed-response pipeline without making a provider request.

## What speech processing does not change

Speech cleanup does not rewrite the assistant's original response for future model context. It only changes the version intended to be spoken aloud.
