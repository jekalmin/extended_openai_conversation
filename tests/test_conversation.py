"""Tests for follow-up question detection in conversation.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from custom_components.extended_openai_conversation.conversation import (
    _is_follow_up_question,
)


@pytest.mark.parametrize(
    "text",
    [
        # English questions ending with ASCII question mark
        "Which light would you like to turn on?",
        "Do you want me to continue ? ",
        # English follow-up phrases without question mark
        "Which one should I control",
        "Would you like me to turn it on",
        # Chinese questions ending with full-width question mark
        "你想让我帮你控制家里的哪个设备呢？",
        "好的，是要打开客厅的灯带吗？",
        "请问需要调节到多少度？ ",
        # Chinese follow-up phrases without question mark
        "你想打开哪个房间的灯",
        "你要开灯还是关灯",
        "要不要我帮你关掉",
        "你需要我做什么",
        "想让我调到多少度",
    ],
)
def test_follow_up_question_detected(text: str) -> None:
    """Follow-up questions are detected in English and Chinese."""
    assert _is_follow_up_question(text) is True


@pytest.mark.parametrize(
    "text",
    [
        # Plain statements
        "好的，已经帮你打开客厅的灯。",
        "The living room light is now on.",
        "现在是晚上 11 点 46 分。",
        # Empty / whitespace
        "",
        "   ",
    ],
)
def test_statement_not_detected(text: str) -> None:
    """Plain statements do not trigger continued conversation."""
    assert _is_follow_up_question(text) is False
