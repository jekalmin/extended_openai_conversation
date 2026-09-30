"""Spoken-response post-processing that preserves the original assistant text."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
import logging
import re
from typing import Any

from .const import (
    CONF_SPEECH_PROCESSING_ENABLED,
    CONF_SPEECH_REGEX_REPLACEMENTS,
    CONF_SPEECH_STRIP_MARKDOWN,
    CONF_SPEECH_STRIP_URLS,
    DEFAULT_SPEECH_PROCESSING_ENABLED,
    DEFAULT_SPEECH_REGEX_REPLACEMENTS,
    DEFAULT_SPEECH_STRIP_MARKDOWN,
    DEFAULT_SPEECH_STRIP_URLS,
)

_LOGGER = logging.getLogger(__name__)

_HEADING_OR_QUOTE = re.compile(r"(?m)^\s{0,3}(?:#{1,6}|>)\s+")
_LIST_MARKER = re.compile(r"(?m)^\s*(?:[-+*]|\d+[.)])\s+")
_EMPHASIS = re.compile(r"(?<!\w)(?:\*\*|__|~~)|(?:\*\*|__|~~)(?!\w)")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_EMPTY_PARENS = re.compile(r"\(\s*\)")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?])")
_WHITESPACE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")

_URL_PREFIXES = ("http://", "https://", "www.")
_WWW_ADDRESS = re.compile(
    r"www\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+",
    re.IGNORECASE,
)
_FORMAT_MARKERS = ("**", "__", "~~")

DEFAULT_STREAMING_BUFFER_LIMIT = 4096
_FORMAT_PREFIX_LIMIT = 32


class StreamingSpeechSanitizer:
    """Incrementally remove Markdown links, bare URLs, and speech formatting.

    Normal prose is released immediately. Only an unresolved construct, trailing
    whitespace, or short line-start prefix is retained. Malformed constructs are
    released verbatim after ``max_buffer_chars`` so provider output cannot make the
    TTS stream wait without bound.
    """

    def __init__(
        self,
        *,
        markdown: bool = True,
        urls: bool = True,
        max_buffer_chars: int = DEFAULT_STREAMING_BUFFER_LIMIT,
    ) -> None:
        if max_buffer_chars < 32:
            raise ValueError("max_buffer_chars must be at least 32")
        self.markdown = markdown
        self.urls = urls
        self.max_buffer_chars = max_buffer_chars
        self.suppressed_constructs = 0
        self.max_buffered_chars = 0
        self._buffer = ""
        self._format_buffer = ""
        self._line_start = True
        self._needs_separator = False
        self._last_output = ""

    @property
    def buffered_chars(self) -> int:
        """Return the current unresolved character count."""
        return len(self._buffer) + len(self._format_buffer)

    def feed(self, delta: str) -> str:
        """Consume one arbitrary provider delta and return speech-safe text."""
        if not delta:
            return ""
        self._buffer += delta
        self.max_buffered_chars = max(self.max_buffered_chars, self.buffered_chars)
        link_safe = self._drain_links(final=False)
        result = self._format(link_safe, final=False)
        self.max_buffered_chars = max(self.max_buffered_chars, self.buffered_chars)
        return result

    def finish(self) -> str:
        """Resolve and return the remaining tail at end of stream."""
        link_safe = self._drain_links(final=True)
        if self._needs_separator:
            self._needs_separator = False
        return self._format(link_safe, final=True)

    def _emit(self, output: list[str], text: str) -> None:
        if not text:
            return
        if self._needs_separator:
            if text.isspace():
                return
            self._needs_separator = False
            if (
                self._last_output
                and not self._last_output[-1].isspace()
                and not text[0].isspace()
                and text[0] not in ",.;:!?)]}"
            ):
                output.append(" ")
                self._last_output += " "
        output.append(text)
        self._last_output += text
        if len(self._last_output) > 4:
            self._last_output = self._last_output[-4:]

    def _suppress(self, output: list[str], end: int) -> int:
        while output and output[-1] and output[-1][-1].isspace():
            output[-1] = output[-1].rstrip()
            if not output[-1]:
                output.pop()
        self._last_output = self._last_output.rstrip()
        self._needs_separator = True
        self.suppressed_constructs += 1
        return end

    def _hold_preceding_whitespace(self, output: list[str], index: int) -> int:
        """Move locally emitted whitespace back into the unresolved input tail."""
        held = 0
        while output and output[-1].isspace():
            output.pop()
            held += 1
        if held:
            self._last_output = self._last_output[:-held]
        return index - held

    def _drain_links(self, *, final: bool) -> str:
        text = self._buffer
        output: list[str] = []
        index = 0

        while index < len(text):
            char = text[index]

            if self.markdown and char == "(":
                link_start = index + 1
                if link_start < len(text) and text[link_start] == "!":
                    link_start += 1
                if link_start < len(text) and text[link_start] == "[":
                    status, link_end = self._markdown_link_end(text, link_start, final)
                    if status == "incomplete":
                        index = self._hold_preceding_whitespace(output, index)
                        break
                    if status == "complete":
                        if link_end == len(text) and not final:
                            index = self._hold_preceding_whitespace(output, index)
                            break
                        end = link_end + (
                            link_end < len(text) and text[link_end] == ")"
                        )
                        index = self._suppress(output, end)
                        continue
                elif link_start >= len(text) and not final:
                    index = self._hold_preceding_whitespace(output, index)
                    break

            markdown_start = index
            if self.markdown and char == "!":
                if index + 1 == len(text) and not final:
                    break
                if index + 1 < len(text) and text[index + 1] == "[":
                    markdown_start += 1
            if self.markdown and text[markdown_start : markdown_start + 1] == "[":
                status, link_end = self._markdown_link_end(text, markdown_start, final)
                if status == "incomplete":
                    index = self._hold_preceding_whitespace(output, index)
                    break
                if status == "complete":
                    index = self._suppress(output, link_end)
                    continue

            if self.urls and char.lower() in "hw":
                remaining = text[index:].lower()
                matching_prefix = next(
                    (
                        prefix
                        for prefix in _URL_PREFIXES
                        if remaining.startswith(prefix)
                    ),
                    None,
                )
                if (
                    matching_prefix is None
                    and not final
                    and any(prefix.startswith(remaining) for prefix in _URL_PREFIXES)
                ):
                    index = self._hold_preceding_whitespace(output, index)
                    break
                preceding = (
                    text[index - 1]
                    if index
                    else (self._last_output[-1:] if matching_prefix == "www." else "")
                )
                if matching_prefix is not None and not (
                    preceding.isalnum() or preceding == "@"
                ):
                    end = index + len(matching_prefix)
                    while (
                        end < len(text)
                        and not text[end].isspace()
                        and text[end] not in "<>"
                    ):
                        end += 1
                    if end == len(text) and not final:
                        index = self._hold_preceding_whitespace(output, index)
                        break
                    if matching_prefix == "www." and not _WWW_ADDRESS.match(
                        text[index:end]
                    ):
                        self._emit(output, char)
                        index += 1
                        continue
                    index = self._suppress(output, end)
                    continue

            if char.isspace() and not final and not text[index + 1 :]:
                break
            self._emit(output, char)
            index += 1

        self._buffer = text[index:]
        if not final and len(self._buffer) > self.max_buffer_chars:
            # A malformed construct or exceptionally long URL must not stall TTS.
            overflow = self._buffer
            self._buffer = ""
            self._emit(output, overflow)
            _LOGGER.debug(
                "Streaming speech sanitizer released malformed text at %d-char limit",
                self.max_buffer_chars,
            )
        elif final and self._buffer:
            self._emit(output, self._buffer)
            self._buffer = ""

        return "".join(output)

    @staticmethod
    def _markdown_link_end(text: str, start: int, final: bool) -> tuple[str, int]:
        label_end = text.find("]", start + 1)
        if label_end == -1:
            return ("no", start) if final else ("incomplete", start)
        if label_end + 1 >= len(text):
            return ("no", start) if final else ("incomplete", start)
        if text[label_end + 1] != "(":
            return "no", start

        url_start = label_end + 2
        if url_start >= len(text):
            return ("no", start) if final else ("incomplete", start)
        depth = 0
        for index in range(url_start, len(text)):
            char = text[index]
            if char.isspace() or char in "<>\n":
                return "no", start
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    return "complete", index + 1
                depth -= 1
        return ("no", start) if final else ("incomplete", start)

    def _format(self, text: str, *, final: bool) -> str:
        if not self.markdown:
            return text
        self._format_buffer += text
        output: list[str] = []
        index = 0

        while index < len(self._format_buffer):
            if self._line_start:
                action, count = self._line_prefix(self._format_buffer[index:], final)
                if action == "incomplete":
                    break
                if action == "strip":
                    index += count
                    self._line_start = False
                    continue
                self._line_start = False

            char = self._format_buffer[index]
            if char == "\n":
                output.append(char)
                index += 1
                self._line_start = True
                continue
            if char == "`":
                index += 1
                continue
            pair = self._format_buffer[index : index + 2]
            if pair in _FORMAT_MARKERS:
                if len(pair) < 2 and not final:
                    break
                if pair != "__" or (
                    index == 0
                    or not self._format_buffer[index - 1].isalnum()
                    or index + 2 == len(self._format_buffer)
                    or not self._format_buffer[index + 2].isalnum()
                ):
                    index += 2
                    continue
            if not final and index == len(self._format_buffer) - 1 and char in "*_~":
                break
            output.append(char)
            index += 1

        self._format_buffer = self._format_buffer[index:]
        if not final and len(self._format_buffer) > _FORMAT_PREFIX_LIMIT:
            output.append(self._format_buffer)
            self._format_buffer = ""
            self._line_start = False
        elif final and self._format_buffer:
            output.append(self._format_buffer.replace("`", ""))
            self._format_buffer = ""
        return "".join(output)

    @staticmethod
    def _line_prefix(text: str, final: bool) -> tuple[str, int]:
        whitespace = 0
        while whitespace < len(text) and text[whitespace] in " \t":
            whitespace += 1
        if whitespace == len(text):
            return ("none", 0) if final else ("incomplete", 0)
        marker = text[whitespace]
        end = whitespace + 1

        if marker == "#":
            while end < len(text) and text[end] == "#" and end - whitespace < 6:
                end += 1
            if end == len(text):
                return ("none", 0) if final else ("incomplete", 0)
            if text[end].isspace():
                while end < len(text) and text[end] in " \t":
                    end += 1
                return "strip", end
        elif marker in ">-+*":
            if end == len(text):
                return ("none", 0) if final else ("incomplete", 0)
            if text[end].isspace():
                while end < len(text) and text[end] in " \t":
                    end += 1
                return "strip", end
        elif marker.isdigit():
            while end < len(text) and text[end].isdigit():
                end += 1
            if end == len(text):
                return ("none", 0) if final else ("incomplete", 0)
            if text[end] in ".)":
                end += 1
                if end == len(text):
                    return ("none", 0) if final else ("incomplete", 0)
                if text[end].isspace():
                    while end < len(text) and text[end] in " \t":
                        end += 1
                    return "strip", end
        return "none", 0


def has_custom_speech_replacements(agent_config: Mapping[str, Any]) -> bool:
    """Return whether configured custom regex requires completed-response TTS."""
    if not agent_config.get(
        CONF_SPEECH_PROCESSING_ENABLED, DEFAULT_SPEECH_PROCESSING_ENABLED
    ):
        return False
    rules = agent_config.get(
        CONF_SPEECH_REGEX_REPLACEMENTS, DEFAULT_SPEECH_REGEX_REPLACEMENTS
    )
    return isinstance(rules, list) and bool(rules)


def streaming_speech_processing_enabled(agent_config: Mapping[str, Any]) -> bool:
    """Return whether built-in cleanup can safely run on progressive deltas."""
    return bool(
        agent_config.get(
            CONF_SPEECH_PROCESSING_ENABLED, DEFAULT_SPEECH_PROCESSING_ENABLED
        )
        and not has_custom_speech_replacements(agent_config)
        and (
            agent_config.get(CONF_SPEECH_STRIP_MARKDOWN, DEFAULT_SPEECH_STRIP_MARKDOWN)
            or agent_config.get(CONF_SPEECH_STRIP_URLS, DEFAULT_SPEECH_STRIP_URLS)
        )
    )


class _SpeechDeltaListener:
    """Sanitize only ChatLog listener deltas while ChatLog retains originals."""

    def __init__(
        self,
        listener: Callable[[Any, dict[str, Any]], None],
        agent_config: Mapping[str, Any],
    ) -> None:
        self._listener = listener
        self._agent_config = agent_config
        self._sanitizer = self._new_sanitizer()

    def _new_sanitizer(self) -> StreamingSpeechSanitizer:
        return StreamingSpeechSanitizer(
            markdown=self._agent_config.get(
                CONF_SPEECH_STRIP_MARKDOWN, DEFAULT_SPEECH_STRIP_MARKDOWN
            ),
            urls=self._agent_config.get(
                CONF_SPEECH_STRIP_URLS, DEFAULT_SPEECH_STRIP_URLS
            ),
        )

    def __call__(self, chat_log: Any, delta: dict[str, Any]) -> None:
        if "role" in delta:
            self.flush(chat_log)
            self._sanitizer = self._new_sanitizer()
            self._listener(chat_log, delta)
            return
        content = delta.get("content")
        if not isinstance(content, str):
            self._listener(chat_log, delta)
            return
        safe = self._sanitizer.feed(content)
        filtered = dict(delta)
        if safe:
            filtered["content"] = safe
        else:
            filtered.pop("content", None)
        if filtered:
            self._listener(chat_log, filtered)

    def flush(self, chat_log: Any) -> None:
        """Forward the resolved end-of-message tail."""
        if tail := self._sanitizer.finish():
            self._listener(chat_log, {"content": tail})

    def log_summary(self) -> None:
        """Log non-sensitive streaming diagnostics."""
        _LOGGER.debug(
            "Streaming speech cleanup suppressed %d constructs; maximum buffered chars: %d",
            self._sanitizer.suppressed_constructs,
            self._sanitizer.max_buffered_chars,
        )


@contextmanager
def async_streaming_speech_cleanup(chat_log: Any, agent_config: Mapping[str, Any]):
    """Wrap a ChatLog delta listener so only progressive output is sanitized."""
    listener = getattr(chat_log, "delta_listener", None)
    if listener is None or not streaming_speech_processing_enabled(agent_config):
        yield
        return
    wrapper = _SpeechDeltaListener(listener, agent_config)
    chat_log.delta_listener = wrapper
    _LOGGER.debug("Streaming speech sanitizer enabled")
    try:
        yield
    finally:
        try:
            wrapper.flush(chat_log)
            wrapper.log_summary()
        finally:
            chat_log.delta_listener = listener


def _built_in_cleanup(text: str, *, markdown: bool, urls: bool) -> str:
    sanitizer = StreamingSpeechSanitizer(markdown=markdown, urls=urls)
    text = sanitizer.feed(text) + sanitizer.finish()
    # Keep the completed-response regex cleanup as a conservative safety net and
    # to preserve the established formatting semantics used by Speech Preview.
    if markdown:
        text = _HEADING_OR_QUOTE.sub("", text)
        text = _LIST_MARKER.sub("", text)
        text = _INLINE_CODE.sub(r"\1", text)
        text = _EMPHASIS.sub("", text)
    return text


def _final_whitespace_cleanup(text: str) -> str:
    text = _EMPTY_PARENS.sub("", text)
    text = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n", text)
    return text.strip()


def process_speech_text(original_text: str, agent_config: Mapping[str, Any]) -> str:
    """Return TTS-only text using the deterministic completed-response pipeline."""
    if not agent_config.get(
        CONF_SPEECH_PROCESSING_ENABLED, DEFAULT_SPEECH_PROCESSING_ENABLED
    ):
        return original_text
    text = _built_in_cleanup(
        original_text,
        markdown=agent_config.get(
            CONF_SPEECH_STRIP_MARKDOWN, DEFAULT_SPEECH_STRIP_MARKDOWN
        ),
        urls=agent_config.get(CONF_SPEECH_STRIP_URLS, DEFAULT_SPEECH_STRIP_URLS),
    )
    rules = agent_config.get(
        CONF_SPEECH_REGEX_REPLACEMENTS, DEFAULT_SPEECH_REGEX_REPLACEMENTS
    )
    if isinstance(rules, list):
        for index, rule in enumerate(rules):
            try:
                if not isinstance(rule, dict):
                    raise TypeError("rule is not an object")
                text = re.sub(str(rule["pattern"]), str(rule["replacement"]), text)
            except KeyError, TypeError, re.error:
                _LOGGER.warning(
                    "Skipping invalid speech regex replacement %s", index, exc_info=True
                )
    return _final_whitespace_cleanup(text)
