"""Tests for TTS-only assistant response processing."""

import logging
from types import SimpleNamespace

import pytest

from custom_components.extended_openai_conversation_responses import speech
from custom_components.extended_openai_conversation_responses.conversation import (
    ExtendedOpenAIAgentEntity,
)
from custom_components.extended_openai_conversation_responses.speech import (
    DEFAULT_STREAMING_BUFFER_LIMIT,
    StreamingSpeechSanitizer,
    _SpeechDeltaListener,
    async_streaming_speech_cleanup,
    has_custom_speech_replacements,
    process_speech_text,
    streaming_speech_processing_enabled,
)
from homeassistant.components import conversation


def _config(**updates):
    return {
        "speech_processing_enabled": True,
        "speech_strip_markdown": True,
        "speech_strip_urls": True,
        "speech_regex_replacements": [],
        **updates,
    }


def _stream(chunks: list[str], **kwargs) -> str:
    sanitizer = StreamingSpeechSanitizer(**kwargs)
    return "".join([*(sanitizer.feed(chunk) for chunk in chunks), sanitizer.finish()])


def test_markdown_citation_and_bare_urls_are_removed_without_mutating_original() -> (
    None
):
    original = "Inside Iran ([apnews.com](https://apnews.com/article/example)) See https://example.com/news."
    spoken = process_speech_text(original, _config())
    assert spoken == "Inside Iran See"
    assert "apnews.com" in original
    assert "https://example.com" in original


def test_common_markdown_is_simplified_conservatively() -> None:
    original = "## Update\n- **Heating** is `on`\n- Temperature is 20 C"
    assert (
        process_speech_text(original, _config())
        == "Update\nHeating is on\nTemperature is 20 C"
    )


def test_custom_replacements_run_in_order_and_allow_empty_replacement() -> None:
    config = _config(
        speech_regex_replacements=[
            {"pattern": r"\[\d+\]", "replacement": ""},
            {"pattern": r"\bHA\b", "replacement": "Home Assistant"},
            {"pattern": "Home Assistant", "replacement": "the smart home"},
        ]
    )
    assert process_speech_text("HA [12] is ready", config) == "the smart home is ready"


def test_url_cleanup_precedes_custom_replacements() -> None:
    config = _config(
        speech_strip_urls=True,
        speech_regex_replacements=[
            {"pattern": r"https?://\S+", "replacement": "web link"}
        ],
    )
    assert process_speech_text("Read https://example.com now", config) == "Read now"


def test_runtime_invalid_regex_is_skipped_safely(caplog) -> None:
    config = _config(
        speech_strip_markdown=False,
        speech_strip_urls=False,
        speech_regex_replacements=[
            {"pattern": "[", "replacement": ""},
            {"pattern": "HA", "replacement": "Home Assistant"},
        ],
    )
    with caplog.at_level(logging.WARNING):
        assert process_speech_text("HA works", config) == "Home Assistant works"
    assert "Skipping invalid speech regex replacement" in caplog.text


def test_disabled_processing_and_ordinary_text_are_unchanged() -> None:
    text = "Ordinary spoken text."
    assert process_speech_text(text, {"speech_processing_enabled": False}) == text
    assert process_speech_text(text, _config()) == text


def test_streaming_citation_in_one_delta() -> None:
    assert (
        _stream(["The answer is 42. ", "([example.com](https://example.com/page))"])
        == "The answer is 42."
    )


def test_streaming_citation_split_before_outer_parenthesis() -> None:
    assert (
        _stream(
            [
                "The answer is 42. ([example.com](https://example.com/page)",
                ") Next sentence.",
            ]
        )
        == "The answer is 42. Next sentence."
    )


def test_streaming_link_split_at_every_position() -> None:
    link = "[example.com](https://example.com/page)"
    for split in range(len(link) + 1):
        assert _stream([link[:split], link[split:]]) == ""
    assert (
        _stream(["[", "example.com", "](", "https://", "example.com/", "page", ")"])
        == ""
    )


def test_streaming_outer_parentheses_split_separately() -> None:
    assert (
        _stream(
            ["Before ", "(", "[example.com](https://example.com/page)", ")", " after"]
        )
        == "Before after"
    )
    assert (
        _stream(["Before ([example.com](https://example.com/page", ")", ") After"])
        == "Before After"
    )


def test_streaming_image_link_matches_existing_cleanup_semantics() -> None:
    assert _stream(["Before ![alt text](https://example.com/image.png) after"]) == (
        "Before after"
    )


def test_streaming_preserves_word_boundaries_and_normal_brackets() -> None:
    assert _stream(["cease", "fire or peace agreement."]) == (
        "ceasefire or peace agreement."
    )
    assert _stream(["The value is [approximately ", "10]."]) == (
        "The value is [approximately 10]."
    )
    assert _stream(["This (ordinary parenthesis", ") survives."]) == (
        "This (ordinary parenthesis) survives."
    )


def test_streaming_malformed_markdown_degrades_with_bounded_memory() -> None:
    malformed = "[label](https://example.com/" + "x" * (
        DEFAULT_STREAMING_BUFFER_LIMIT + 10
    )
    sanitizer = StreamingSpeechSanitizer()
    emitted = sanitizer.feed(malformed)
    assert emitted == malformed
    assert sanitizer.buffered_chars == 0
    assert sanitizer.finish() == ""
    assert _stream(["Keep [unfinished](https://example.com"], urls=False) == (
        "Keep [unfinished](https://example.com"
    )


def test_streaming_bare_url_complete_split_and_punctuation() -> None:
    # Match the established completed-response behavior: terminal punctuation is
    # part of the bare URL match and is therefore suppressed with it.
    assert _stream(["See https://example.com/test. Next."]) == "See Next."
    assert _stream(["See https://exa", "mple.com/test! Then."]) == "See Then."


def test_streaming_multiple_adjacent_and_terminal_citations() -> None:
    citation_a = "([a.example](https://a.example/article))"
    citation_b = "([b.example](https://b.example/article))"
    assert _stream([f"Answer. {citation_a} {citation_b}"]) == "Answer."
    assert _stream([f"Answer. {citation_a}\n\nMore. {citation_b}"]) == ("Answer. More.")


def test_streaming_real_trace_shape() -> None:
    chunks = [
        "As of August 12, 2026, the war remains active with no durable cease",
        "fire or peace agreement.\n\nLarge-scale attacks continue. ",
        "([example.com](https://example.com/complete?utm_source=openai))\n\nOn the battlefield, progress is limited. ",
        "Allies are responding independently. ([example.org](https://example.org/split?utm_source=openai)",
        ")\n\nUkraine has increasingly adapted.",
    ]
    assert _stream(chunks) == (
        "As of August 12, 2026, the war remains active with no durable ceasefire "
        "or peace agreement.\n\nLarge-scale attacks continue. On the battlefield, "
        "progress is limited. Allies are responding independently. Ukraine has "
        "increasingly adapted."
    )


def test_streaming_common_markdown_formatting() -> None:
    assert (
        _stream(["# Head", "ing\n- **First** item\n2", ". `Second` item"])
        == "Heading\nFirst item\nSecond item"
    )


async def test_chat_log_retains_original_while_listener_gets_safe_deltas() -> None:
    heard: list[dict] = []
    chat_log = conversation.ChatLog(SimpleNamespace(data={}), "conversation-id")
    chat_log.delta_listener = lambda _chat_log, delta: heard.append(delta)

    async def stream():
        yield {"role": "assistant"}
        yield {"content": "Answer. ([example.com](https://example.com/page))"}

    with async_streaming_speech_cleanup(chat_log, _config()):
        contents = [
            content
            async for content in chat_log.async_add_delta_content_stream(
                "conversation.test", stream()
            )
        ]

    assert contents[-1].content == ("Answer. ([example.com](https://example.com/page))")
    assert heard == [{"role": "assistant"}, {"content": "Answer."}]


def test_custom_regex_requires_completed_response_processing() -> None:
    assert has_custom_speech_replacements(
        _config(
            speech_regex_replacements=[{"pattern": "begin.*end", "replacement": ""}]
        )
    )


def test_custom_regex_disables_agent_progressive_streaming() -> None:
    subentry = SimpleNamespace(
        subentry_id="agent-id",
        title="Agent",
        data=_config(
            speech_regex_replacements=[{"pattern": "begin.*end", "replacement": ""}]
        ),
    )
    entity = ExtendedOpenAIAgentEntity(SimpleNamespace(), subentry)
    assert entity.supports_streaming is False

    subentry.data = _config()
    entity = ExtendedOpenAIAgentEntity(SimpleNamespace(), subentry)
    assert entity.supports_streaming is True


def test_streaming_buffer_limit_rejects_too_small_values() -> None:
    with pytest.raises(ValueError, match="at least 32"):
        StreamingSpeechSanitizer(max_buffer_chars=31)


def test_empty_delta_and_terminal_separator_are_safe() -> None:
    sanitizer = StreamingSpeechSanitizer()
    assert sanitizer.feed("") == ""
    assert sanitizer.feed("Answer [source](https://example.com)") == "Answer"
    assert sanitizer.finish() == ""


def test_partial_url_prefix_is_buffered_until_resolved() -> None:
    sanitizer = StreamingSpeechSanitizer(markdown=False, urls=True)
    assert sanitizer.feed("See htt") == "See"
    assert sanitizer.feed("ps://example.com/path") == ""
    assert sanitizer.feed(" next") == " next"
    assert sanitizer.finish() == ""


def test_url_like_text_inside_identifier_is_not_suppressed() -> None:
    assert _stream(["mailboxhttps://example.com done"], markdown=False) == (
        "mailboxhttps://example.com done"
    )


def test_final_incomplete_markdown_constructs_are_preserved() -> None:
    assert _stream(["Keep [unfinished"], urls=False) == "Keep [unfinished"
    assert _stream(["Keep [label]"], urls=False) == "Keep [label]"
    assert _stream(["Keep [label]("], urls=False) == "Keep [label]("
    assert _stream(["Keep [label](bad url"], urls=False) == "Keep [label](bad url"


def test_nested_parentheses_in_markdown_url_are_suppressed() -> None:
    assert _stream(["Before [label](https://example.com/a(b)c) after"]) == (
        "Before after"
    )


def test_markdown_disabled_returns_text_without_format_buffering() -> None:
    sanitizer = StreamingSpeechSanitizer(markdown=False, urls=False)
    assert sanitizer.feed("**literal** `code`") == "**literal** `code`"
    assert sanitizer.finish() == ""


def test_line_prefix_boundaries_and_non_markers() -> None:
    sanitizer = StreamingSpeechSanitizer()
    assert sanitizer._line_prefix("   ", False) == ("incomplete", 0)
    assert sanitizer._line_prefix("   ", True) == ("none", 0)
    assert sanitizer._line_prefix("###", False) == ("incomplete", 0)
    assert sanitizer._line_prefix("###", True) == ("none", 0)
    assert sanitizer._line_prefix("-", False) == ("incomplete", 0)
    assert sanitizer._line_prefix("-", True) == ("none", 0)
    assert sanitizer._line_prefix("12", False) == ("incomplete", 0)
    assert sanitizer._line_prefix("12", True) == ("none", 0)
    assert sanitizer._line_prefix("12.", False) == ("incomplete", 0)
    assert sanitizer._line_prefix("12.", True) == ("none", 0)
    assert sanitizer._line_prefix("word", False) == ("none", 0)


def test_streaming_enablement_matrix() -> None:
    assert not streaming_speech_processing_enabled(
        _config(speech_processing_enabled=False)
    )
    assert not streaming_speech_processing_enabled(
        _config(
            speech_regex_replacements=[{"pattern": "x", "replacement": "y"}]
        )
    )
    assert not streaming_speech_processing_enabled(
        _config(speech_strip_markdown=False, speech_strip_urls=False)
    )
    assert streaming_speech_processing_enabled(
        _config(speech_strip_markdown=False, speech_strip_urls=True)
    )


def test_has_custom_replacements_rejects_disabled_or_non_list_rules() -> None:
    assert not has_custom_speech_replacements(
        _config(speech_processing_enabled=False, speech_regex_replacements=[{}])
    )
    assert not has_custom_speech_replacements(
        _config(speech_regex_replacements={"pattern": "x"})
    )


def test_delta_listener_forwards_role_non_string_and_metadata_only_deltas() -> None:
    heard: list[dict] = []
    listener = _SpeechDeltaListener(
        lambda _chat_log, delta: heard.append(delta), _config()
    )
    chat_log = object()

    listener(chat_log, {"role": "assistant"})
    listener(chat_log, {"content": 42})
    listener(chat_log, {"content": "[source](https://example.com)", "id": "chunk"})

    assert heard == [
        {"role": "assistant"},
        {"content": 42},
        {"id": "chunk"},
    ]


def test_delta_listener_flushes_buffered_tail_before_new_role() -> None:
    heard: list[dict] = []
    listener = _SpeechDeltaListener(
        lambda _chat_log, delta: heard.append(delta), _config()
    )
    chat_log = object()

    listener(chat_log, {"content": "unfinished "})
    listener(chat_log, {"role": "assistant"})

    assert "".join(delta.get("content", "") for delta in heard[:-1]) == "unfinished "
    assert heard[-1] == {"role": "assistant"}


def test_streaming_cleanup_bypasses_without_listener_or_when_disabled() -> None:
    chat_log = SimpleNamespace(delta_listener=None)
    with async_streaming_speech_cleanup(chat_log, _config()):
        assert chat_log.delta_listener is None

    original = lambda _chat_log, _delta: None
    chat_log = SimpleNamespace(delta_listener=original)
    with async_streaming_speech_cleanup(
        chat_log, _config(speech_processing_enabled=False)
    ):
        assert chat_log.delta_listener is original
    assert chat_log.delta_listener is original


def test_streaming_cleanup_restores_listener_after_exception() -> None:
    original = lambda _chat_log, _delta: None
    chat_log = SimpleNamespace(delta_listener=original)

    with pytest.raises(RuntimeError, match="boom"):
        with async_streaming_speech_cleanup(chat_log, _config()):
            assert chat_log.delta_listener is not original
            raise RuntimeError("boom")

    assert chat_log.delta_listener is original


def test_process_speech_text_skips_non_mapping_and_missing_key_rules(caplog) -> None:
    config = _config(
        speech_strip_markdown=False,
        speech_strip_urls=False,
        speech_regex_replacements=["not-a-rule", {"pattern": "missing replacement"}],
    )
    with caplog.at_level(logging.WARNING):
        assert process_speech_text("unchanged", config) == "unchanged"
    assert caplog.text.count("Skipping invalid speech regex replacement") == 2

    non_list = _config(
        speech_strip_markdown=False,
        speech_strip_urls=False,
        speech_regex_replacements={"pattern": "hello", "replacement": "goodbye"},
    )
    assert process_speech_text("hello   world", non_list) == "hello world"


def test_emit_separator_handles_whitespace_punctuation_and_plain_text() -> None:
    sanitizer = speech.StreamingSpeechSanitizer()
    output: list[str] = []

    sanitizer._emit(output, "")
    assert output == []

    sanitizer._last_output = "word"
    sanitizer._needs_separator = True
    sanitizer._emit(output, "   ")
    assert output == []
    assert sanitizer._needs_separator is True

    sanitizer._emit(output, ",")
    assert output == [","]
    assert sanitizer._needs_separator is False

    sanitizer._last_output = "word"
    sanitizer._needs_separator = True
    output.clear()
    sanitizer._emit(output, "next")
    assert output == [" ", "next"]


def test_suppress_and_hold_whitespace_cover_trailing_chunks() -> None:
    sanitizer = speech.StreamingSpeechSanitizer()
    output = ["text", "   "]
    sanitizer._last_output = "text   "

    end = sanitizer._suppress(output, 12)

    assert end == 12
    assert output == ["text"]
    assert sanitizer._last_output == "text"
    assert sanitizer._needs_separator is True
    assert sanitizer.suppressed_constructs == 1

    output = ["x", " ", "\t"]
    sanitizer._last_output = "x \t"
    assert sanitizer._hold_preceding_whitespace(output, 5) == 3
    assert output == ["x"]
    assert sanitizer._last_output == "x"


def test_outer_image_link_and_partial_outer_parenthesis_paths() -> None:
    assert _stream(["Before (![alt](https://example.com/a.png)) after"]) == "Before after"

    sanitizer = speech.StreamingSpeechSanitizer()
    assert sanitizer.feed("Before (") == "Before"
    assert sanitizer.finish() == " ("


def test_partial_url_prefix_and_identifier_paths() -> None:
    sanitizer = speech.StreamingSpeechSanitizer(markdown=False, urls=True)
    assert sanitizer.feed("prefix htt") == "prefix"
    assert sanitizer.feed("x next") == " httx next"
    assert sanitizer.finish() == ""

    assert _stream(["h", "ttp://example.com next"], markdown=False, urls=True) == "next"


def test_markdown_link_parser_final_and_nested_failure_paths() -> None:
    parser = speech.StreamingSpeechSanitizer._markdown_link_end

    assert parser("[label", 0, True) == ("no", 0)
    assert parser("[label]", 0, True) == ("no", 0)
    assert parser("[label](", 0, True) == ("no", 0)
    assert parser("[label]x", 0, True) == ("no", 0)
    assert parser("[label](bad url)", 0, False) == ("no", 0)
    assert parser("[label](https://example.com/a(b)c)", 0, False)[0] == "complete"


def test_format_handles_newline_backticks_markers_and_terminal_tail() -> None:
    sanitizer = speech.StreamingSpeechSanitizer(markdown=True, urls=False)
    assert sanitizer._format("plain\n`code` **bold** __word__", final=True) == (
        "plain\ncode bold word"
    )

    sanitizer = speech.StreamingSpeechSanitizer(markdown=True, urls=False)
    assert sanitizer._format("trailing*", final=False) == "trailing"
    assert sanitizer._format("", final=True) == "*"


def test_line_prefix_strips_supported_markers_and_rejects_near_misses() -> None:
    line_prefix = speech.StreamingSpeechSanitizer._line_prefix

    assert line_prefix("   ###   Heading", False) == ("strip", 9)
    assert line_prefix("\t>  Quote", False) == ("strip", 4)
    assert line_prefix("  -   Item", False) == ("strip", 6)
    assert line_prefix("12)  Item", False) == ("strip", 5)
    assert line_prefix("123x Item", False) == ("none", 0)
    assert line_prefix("####### Heading", False) == ("none", 0)
    assert line_prefix("-not-a-list", False) == ("none", 0)


def test_delta_listener_drops_content_only_suppressed_delta_and_flushes_tail() -> None:
    heard: list[dict] = []
    listener = speech._SpeechDeltaListener(
        lambda _chat_log, delta: heard.append(delta), _config()
    )
    chat_log = object()

    listener(chat_log, {"content": "[source](https://example.com)"})
    assert heard == []

    listener(chat_log, {"content": "tail "})
    listener.flush(chat_log)
    assert "".join(item.get("content", "") for item in heard) == "tail "


def test_streaming_cleanup_flush_failure_still_restores_original_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = lambda _chat_log, _delta: None
    chat_log = SimpleNamespace(delta_listener=original)

    def fail_flush(self, _chat_log):
        raise RuntimeError("flush failed")

    monkeypatch.setattr(speech._SpeechDeltaListener, "flush", fail_flush)

    with pytest.raises(RuntimeError, match="flush failed"):
        with speech.async_streaming_speech_cleanup(chat_log, _config()):
            assert chat_log.delta_listener is not original

    assert chat_log.delta_listener is original


def test_streaming_cleanup_summary_runs_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    original = lambda _chat_log, _delta: None
    chat_log = SimpleNamespace(delta_listener=original)
    calls: list[str] = []

    monkeypatch.setattr(
        speech._SpeechDeltaListener,
        "log_summary",
        lambda self: calls.append("summary"),
    )

    with speech.async_streaming_speech_cleanup(chat_log, _config()):
        assert chat_log.delta_listener is not original

    assert calls == ["summary"]
    assert chat_log.delta_listener is original


def test_process_speech_text_skips_bad_rule_types_and_bad_replacement(caplog) -> None:
    config = _config(
        speech_strip_markdown=False,
        speech_strip_urls=False,
        speech_regex_replacements=[
            None,
            {"replacement": "missing pattern"},
            {"pattern": "x", "replacement": r"\9"},
            {"pattern": "x", "replacement": "y"},
        ],
    )

    with caplog.at_level(logging.WARNING):
        assert speech.process_speech_text("x", config) == "y"

    assert caplog.text.count("Skipping invalid speech regex replacement") == 3


def test_final_whitespace_cleanup_collapses_empty_parens_and_spacing() -> None:
    assert speech._final_whitespace_cleanup("Hello ( )   , world\n\n\nnext") == (
        "Hello, world\nnext"
    )


def test_suppress_trims_trailing_space_without_dropping_nonempty_chunk() -> None:
    sanitizer = speech.StreamingSpeechSanitizer()
    output = ["text "]
    sanitizer._last_output = "text "

    assert sanitizer._suppress(output, 7) == 7
    assert output == ["text"]
    assert sanitizer._last_output == "text"


def test_lone_markdown_image_marker_waits_for_stream_completion() -> None:
    sanitizer = speech.StreamingSpeechSanitizer(markdown=True, urls=False)

    assert sanitizer.feed("Hello !") == "Hello "
    assert sanitizer.finish() == "!"


def test_format_releases_ambiguous_line_prefix_at_safety_limit() -> None:
    sanitizer = speech.StreamingSpeechSanitizer(markdown=True, urls=False)
    text = " " * (speech._FORMAT_PREFIX_LIMIT + 1)

    assert sanitizer._format(text, final=False) == text
    assert sanitizer._format_buffer == ""
    assert sanitizer._line_start is False


def test_line_prefix_finalizes_ambiguous_terminal_markers_as_plain_text() -> None:
    line_prefix = speech.StreamingSpeechSanitizer._line_prefix

    assert line_prefix("#", True) == ("none", 0)
    assert line_prefix("-", True) == ("none", 0)
    assert line_prefix("12", True) == ("none", 0)
    assert line_prefix("12)", True) == ("none", 0)
    assert line_prefix("   ", True) == ("none", 0)
