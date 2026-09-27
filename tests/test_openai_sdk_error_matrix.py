"""Provider failure taxonomy exercised through the installed OpenAI SDK wire."""

from __future__ import annotations

from collections.abc import Callable

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError
import pytest

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    API_MODE_RESPONSES,
)
from custom_components.extended_openai_conversation_responses.provider_errors import (
    classify_config_provider_error,
    provider_user_message,
)
from tests.test_openai_sdk_wire import (
    _chat_log,
    _chat_text_stream,
    _client,
    _close_client,
    _entity,
    _responses_text_stream,
    _stream_response,
    _Wire,
)

API_MODES = (API_MODE_RESPONSES, API_MODE_CHAT_COMPLETIONS)
STATUS_CASES = (
    (400, "provider_error"),
    (401, "invalid_auth"),
    (403, "provider_forbidden"),
    (404, "provider_error"),
    (408, "provider_error"),
    (409, "provider_error"),
    (429, "provider_rate_limited"),
    (500, "provider_unavailable"),
    (502, "provider_unavailable"),
    (503, "provider_unavailable"),
)


def _text_reply(api_mode: str) -> Callable[[httpx.Request], httpx.Response]:
    return _stream_response(
        _responses_text_stream("Recovered")
        if api_mode == API_MODE_RESPONSES
        else _chat_text_stream("Recovered")
    )


@pytest.mark.parametrize("api_mode", API_MODES)
@pytest.mark.parametrize(("status", "category"), STATUS_CASES)
async def test_real_sdk_http_error_is_classified_once_then_next_turn_succeeds(
    hass, api_mode: str, status: int, category: str
) -> None:
    """HTTP errors cannot create tool work or poison the next provider request."""

    def failure(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            headers={"content-type": "application/json", "x-request-id": "req-wire"},
            json={
                "error": {
                    "message": "model not found" if status == 404 else "wire failure",
                    "type": "invalid_request_error" if status < 500 else "server_error",
                    "code": "model_not_found" if status == 404 else "wire_error",
                }
            },
            request=request,
        )

    wire = _Wire([failure, _text_reply(api_mode)])
    client = _client(wire)
    entity = _entity(hass, client, api_mode)
    chat_log = _chat_log(hass)
    try:
        with pytest.raises(APIStatusError) as caught:
            await entity._async_handle_chat_log(chat_log, [], [])
        assert caught.value.status_code == status
        assert classify_config_provider_error(caught.value) == category
        message = provider_user_message(caught.value)
        assert f"HTTP {status}" in message
        assert "sk-wire-test" not in message
        assert len(wire.requests) == 1
        await entity._async_handle_chat_log(chat_log, [], [])
    finally:
        await _close_client(client)
    assert len(wire.requests) == 2
    assert chat_log.content[-1].content == "Recovered"


@pytest.mark.parametrize("api_mode", API_MODES)
@pytest.mark.parametrize(
    ("transport_error", "sdk_error"),
    [
        (httpx.ConnectError("connection failed"), APIConnectionError),
        (httpx.ConnectTimeout("connect timeout"), APITimeoutError),
        (httpx.ReadTimeout("read timeout"), APITimeoutError),
        (httpx.RemoteProtocolError("closed before headers"), APIConnectionError),
        (httpx.ReadError("connection closed"), APIConnectionError),
    ],
)
async def test_real_sdk_transport_error_is_cleaned_up_and_next_turn_succeeds(
    hass,
    api_mode: str,
    transport_error: httpx.RequestError,
    sdk_error: type[Exception],
) -> None:
    """Connection failures and timeouts are delivered by the SDK without replay."""

    def failure(_request: httpx.Request) -> httpx.Response:
        raise transport_error

    wire = _Wire([failure, _text_reply(api_mode)])
    client = _client(wire)
    entity = _entity(hass, client, api_mode)
    chat_log = _chat_log(hass)
    try:
        with pytest.raises(sdk_error) as caught:
            await entity._async_handle_chat_log(chat_log, [], [])
        assert classify_config_provider_error(caught.value) == "cannot_connect"
        assert len(wire.requests) == 1
        await entity._async_handle_chat_log(chat_log, [], [])
    finally:
        await _close_client(client)
    assert len(wire.requests) == 2
    assert chat_log.content[-1].content == "Recovered"
