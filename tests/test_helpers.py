"""Tests for helper functions."""

from unittest.mock import MagicMock, patch

from custom_components.extended_openai_conversation import helpers


class TestGetAuthenticatedClient:
    """Test get_authenticated_client http client selection."""

    async def test_follow_redirects_uses_dedicated_client(self, hass):
        """When enabled, a redirect-following httpx client is passed to the SDK."""
        redirect_client = MagicMock(name="redirect_client")
        shared_client = MagicMock(name="shared_client")

        with (
            patch.object(
                helpers, "create_async_httpx_client", return_value=redirect_client
            ) as mock_create,
            patch.object(
                helpers, "get_async_client", return_value=shared_client
            ) as mock_get,
            patch.object(helpers, "AsyncOpenAI") as mock_openai,
        ):
            await helpers.get_authenticated_client(
                hass=hass,
                api_key="test-key",
                base_url="https://example.com/v1",
                api_version=None,
                organization=None,
                api_provider="openai",
                skip_authentication=True,
                follow_redirects=True,
            )

        mock_create.assert_called_once_with(hass, follow_redirects=True)
        mock_get.assert_not_called()
        assert mock_openai.call_args.kwargs["http_client"] is redirect_client

    async def test_default_reuses_shared_client(self, hass):
        """By default, the HA shared httpx client (no redirects) is reused."""
        redirect_client = MagicMock(name="redirect_client")
        shared_client = MagicMock(name="shared_client")

        with (
            patch.object(
                helpers, "create_async_httpx_client", return_value=redirect_client
            ) as mock_create,
            patch.object(
                helpers, "get_async_client", return_value=shared_client
            ) as mock_get,
            patch.object(helpers, "AsyncOpenAI") as mock_openai,
        ):
            await helpers.get_authenticated_client(
                hass=hass,
                api_key="test-key",
                base_url="https://example.com/v1",
                api_version=None,
                organization=None,
                api_provider="openai",
                skip_authentication=True,
            )

        mock_get.assert_called_once_with(hass)
        mock_create.assert_not_called()
        assert mock_openai.call_args.kwargs["http_client"] is shared_client

    async def test_follow_redirects_applies_to_azure_client(self, hass):
        """The flag also applies to the Azure client branch."""
        redirect_client = MagicMock(name="redirect_client")

        with (
            patch.object(
                helpers, "create_async_httpx_client", return_value=redirect_client
            ) as mock_create,
            patch.object(helpers, "AsyncAzureOpenAI") as mock_azure,
        ):
            await helpers.get_authenticated_client(
                hass=hass,
                api_key="test-key",
                base_url="https://example.openai.azure.com",
                api_version="2024-02-01",
                organization=None,
                api_provider="azure",
                skip_authentication=True,
                follow_redirects=True,
            )

        mock_create.assert_called_once_with(hass, follow_redirects=True)
        assert mock_azure.call_args.kwargs["http_client"] is redirect_client
