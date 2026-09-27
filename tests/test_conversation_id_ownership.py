"""Conversation-ID ownership helper contracts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from custom_components.extended_openai_conversation_responses import (
    conversation_id_ownership as ownership,
)
from custom_components.extended_openai_conversation_responses.scope import user_scope


def _scope(user_id: str = "user-a"):
    return user_scope(user_id, source="test")


def test_claim_short_circuits_invalid_inputs_without_mutation() -> None:
    scope = _scope()
    hass = SimpleNamespace(data={})

    assert ownership.claim_conversation_id(
        hass, "agent", scope, None, guest_active=False
    ) is None
    assert ownership.claim_conversation_id(
        None, "agent", scope, "conversation", guest_active=False
    ) == "conversation"
    assert ownership.claim_conversation_id(
        hass, None, scope, "conversation", guest_active=False
    ) == "conversation"
    assert ownership.claim_conversation_id(
        SimpleNamespace(data=None),
        "agent",
        scope,
        "conversation",
        guest_active=False,
    ) == "conversation"
    assert hass.data == {}


def test_claim_reuses_same_owner_but_isolates_scope_agent_and_guest(
    monkeypatch,
) -> None:
    sequence = iter(["generated-one", "generated-two", "generated-three"])
    monkeypatch.setattr(
        ownership,
        "uuid4",
        lambda: SimpleNamespace(hex=next(sequence)),
    )
    hass = SimpleNamespace(data={})
    scope_a = _scope("a")
    scope_b = _scope("b")

    first = ownership.claim_conversation_id(
        hass, "agent-a", scope_a, "shared-id", guest_active=False
    )
    same = ownership.claim_conversation_id(
        hass, "agent-a", scope_a, "shared-id", guest_active=False
    )
    other_scope = ownership.claim_conversation_id(
        hass, "agent-a", scope_b, "shared-id", guest_active=False
    )
    other_agent = ownership.claim_conversation_id(
        hass, "agent-b", scope_a, "shared-id", guest_active=False
    )
    guest = ownership.claim_conversation_id(
        hass, "agent-a", scope_a, "shared-id", guest_active=True
    )

    assert first == same == "shared-id"
    assert other_scope == "extended-openai-agent-a-generated-one"
    assert other_agent == "extended-openai-agent-b-generated-two"
    assert guest == "extended-openai-agent-a-generated-three"

    owners = hass.data[ownership._CONVERSATION_ID_OWNERS]
    assert owners["shared-id"] == ("agent-a", "user:a")
    assert owners[other_scope] == ("agent-a", "user:b")
    assert owners[other_agent] == ("agent-b", "user:a")
    assert owners[guest] == ("agent-a", "guest")


def test_release_only_removes_matching_owner_and_tolerates_missing_storage() -> None:
    key = ownership._CONVERSATION_ID_OWNERS
    hass = SimpleNamespace(
        data={
            key: {
                "one": ("agent", "user:a"),
                "two": ("agent", "user:b"),
            }
        }
    )

    ownership.release_conversation_id_claim(
        hass, "one", ("agent", "user:wrong")
    )
    assert "one" in hass.data[key]

    ownership.release_conversation_id_claim(
        hass, "one", ("agent", "user:a")
    )
    assert "one" not in hass.data[key]
    assert "two" in hass.data[key]

    ownership.release_conversation_id_claim(None, "two", ("agent", "user:b"))
    ownership.release_conversation_id_claim(
        SimpleNamespace(data=None), "two", ("agent", "user:b")
    )
    ownership.release_conversation_id_claim(
        SimpleNamespace(data={}), "two", ("agent", "user:b")
    )


def test_cleanup_registration_releases_claim_and_ignores_unsupported_session() -> None:
    key = ownership._CONVERSATION_ID_OWNERS
    owner = ("agent", "user:a")
    hass = SimpleNamespace(data={key: {"conversation": owner}})
    callback = Mock()
    session = SimpleNamespace(async_on_cleanup=callback)

    ownership.register_conversation_id_cleanup(
        session, hass, "conversation", owner
    )
    callback.assert_called_once()
    release = callback.call_args.args[0]
    release()
    assert hass.data[key] == {}

    unsupported = SimpleNamespace(async_on_cleanup=None)
    ownership.register_conversation_id_cleanup(
        unsupported, hass, "conversation", owner
    )
