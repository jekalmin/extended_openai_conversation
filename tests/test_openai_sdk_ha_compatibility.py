"""SDK compatibility with Home Assistant's shared legacy HTTPX transport."""

import json
from pathlib import Path

import httpx
from packaging.requirements import Requirement
import pytest

from custom_components.extended_openai_conversation_responses import helpers


@pytest.mark.parametrize("sdk", ["2.21.0", "2.45.0", "3.10.0"])
def test_manifest_allows_supported_ha_pins_and_bounds_unvalidated_sdk(sdk):
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "custom_components/extended_openai_conversation_responses/manifest.json"
        ).read_text()
    )
    requirement = next(
        Requirement(value)
        for value in manifest["requirements"]
        if Requirement(value).name == "openai"
    )
    assert requirement.specifier.contains(sdk)
    assert not requirement.specifier.contains("2.20.0")
    assert not requirement.specifier.contains("3.10.1")
    assert not requirement.specifier.contains("4.0.0")


@pytest.mark.parametrize("provider", ["openai", "azure"])
async def test_real_sdk_authentication_uses_ha_transport_once(
    hass, monkeypatch, provider
):
    requests = []

    async def models(request):
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "gpt-test",
                        "object": "model",
                        "created": 0,
                        "owned_by": "test",
                    }
                ],
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(models)
    ) as shared_client:
        monkeypatch.setattr(helpers, "get_async_client", lambda _hass: shared_client)
        client = await helpers.get_authenticated_client(
            hass,
            "test-key",
            "https://mock.invalid/v1",
            "2024-10-21" if provider == "azure" else None,
            None,
            provider,
        )
        assert client._client is shared_client
        assert len(requests) == 1
        if provider == "azure":
            assert requests[0].url.path == "/v1/openai/models"
            assert requests[0].url.params["api-version"] == "2024-10-21"
            assert requests[0].headers["api-key"] == "test-key"
        else:
            assert requests[0].url.path == "/v1/models"
            assert requests[0].headers["authorization"] == "Bearer test-key"


@pytest.mark.parametrize("sdk", ["2.21.0", "2.45.0", "3.10.0"])
def test_ci_dependency_install_respects_selected_ha_constraints(
    tmp_path, monkeypatch, sdk
):
    from types import SimpleNamespace

    from ci import install_ha_dependencies

    ha_package = tmp_path / "homeassistant"
    (ha_package / "components" / "conversation").mkdir(parents=True)
    constraints = ha_package / "package_constraints.txt"
    constraints.write_text(
        f"openai=={sdk}\nvoluptuous-openapi==0.2.0\npydantic==2.13.5\n"
    )
    (ha_package / "components" / "conversation" / "manifest.json").write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "requirements": [
                    "openai>=2.21.0,<=3.10.0",
                    "voluptuous-openapi==0.4.1",
                ],
                "dependencies": ["conversation"],
            }
        )
    )
    monkeypatch.setattr(
        install_ha_dependencies.homeassistant, "__path__", [str(ha_package)]
    )
    monkeypatch.setattr(
        install_ha_dependencies,
        "_parse_args",
        lambda: SimpleNamespace(manifest=manifest),
    )
    commands = []
    captured_constraints = []
    monkeypatch.setattr(
        install_ha_dependencies,
        "requires",
        lambda _name: [
            "pydantic==2.13.4",
            "aiohttp[speedups]>=3.9",
            "voluptuous-openapi==0.2.0",
            "unavailable==1; extra == 'unused'",
        ],
    )

    def capture(command):
        commands.append(command)
        captured_constraints.append(
            Path(command[command.index("--constraint") + 1]).read_text()
        )

    monkeypatch.setattr(install_ha_dependencies.subprocess, "check_call", capture)
    install_ha_dependencies.main()
    assert len(commands) == 1
    command = commands[0]
    assert captured_constraints == [f"pydantic==2.13.4\naiohttp>=3.9\nopenai=={sdk}\n"]
    assert "voluptuous-openapi==0.4.1" in command
    assert "openai>=2.21.0,<=3.10.0" in command
