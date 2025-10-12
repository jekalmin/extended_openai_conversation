import importlib.util
import sys
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "context_composer",
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "extended_openai_conversation"
    / "context_composer.py",
)
context_composer = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = context_composer
SPEC.loader.exec_module(context_composer)

compose_system_sections = context_composer.compose_system_sections


def test_compose_system_sections_respects_budgets():
    profile = "A" * 400
    scratchpad = "Previous answer with many details." * 20
    retrieved = "Snippet one. " * 80

    composition = compose_system_sections(
        profile,
        scratchpad,
        retrieved,
        budget_profile=50,
        budget_scratchpad=40,
        budget_retrieved=60,
    )

    assert composition.slices["profile"].tokens <= 50
    assert composition.slices["scratchpad"].tokens <= 40
    assert composition.slices["retrieved"].tokens <= 60
    assert "[Context" not in composition.content  # labels handled externally


def test_compose_system_sections_omits_empty_slices():
    composition = compose_system_sections(
        None,
        "",
        "Details here.",
        budget_profile=20,
        budget_scratchpad=20,
        budget_retrieved=20,
    )

    assert "Details here." in composition.content
    assert composition.slices["profile"].text == ""
    assert composition.slices["scratchpad"].text == ""
