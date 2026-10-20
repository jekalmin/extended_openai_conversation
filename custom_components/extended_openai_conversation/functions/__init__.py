"""Tools for Extended OpenAI Conversation."""

from __future__ import annotations

from ..exceptions import FunctionNotFound
from .base import Function
from .bash import BashFunction
from .composite import CompositeFunction
from .file import EditFileFunction, ReadFileFunction, WriteFileFunction
from .native import NativeFunction
from .parallel import ParallelFunction
from .script import ScriptFunction
from .sqlite import SqliteFunction
from .template import TemplateFunction
from .web import RestFunction, ScrapeFunction

__all__ = [
    "BashFunction",
    "CompositeFunction",
    "EditFileFunction",
    "Function",
    "NativeFunction",
    "ParallelFunction",
    "ReadFileFunction",
    "RestFunction",
    "ScrapeFunction",
    "ScriptFunction",
    "SqliteFunction",
    "TemplateFunction",
    "WriteFileFunction",
    "get_function",
]

FUNCTIONS: dict[str, Function] = {
    "native": NativeFunction(),
    "parallel": ParallelFunction(),
    "script": ScriptFunction(),
    "template": TemplateFunction(),
    "rest": RestFunction(),
    "scrape": ScrapeFunction(),
    "composite": CompositeFunction(),
    "sqlite": SqliteFunction(),
    "bash": BashFunction(),
    "read_file": ReadFileFunction(),
    "write_file": WriteFileFunction(),
    "edit_file": EditFileFunction(),
}


def get_function(function_type: str) -> Function:
    """Get function by function_config."""
    function = FUNCTIONS.get(function_type)
    if function is None:
        raise FunctionNotFound(function_type)
    return function
