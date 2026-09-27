"""Make critical HA Store write failures visible to EOAI transactions.

HA Store logs and swallows WriteError from its file writer. EOAI's rollback
managers must instead receive that failure before publishing the mutation.
"""

from __future__ import annotations

from typing import Any

from homeassistant.helpers.storage import Store
from homeassistant.util.file import WriteError


class PropagatingWriteStore(Store[dict[str, Any]]):
    """Preserve HA's atomic writer, but surface its OS failure to the caller."""

    async def _async_write_data(self, data: dict[str, Any]) -> None:
        try:
            await super()._async_write_data(data)
        except WriteError as error:
            cause = error.__cause__
            number = cause.errno if isinstance(cause, OSError) else None
            # Do not expose the HA storage path or a private payload upstream.
            raise OSError(number, "Private storage write failed") from None
