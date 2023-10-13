"""The exceptions used by Extended OpenAI Conversation."""
from homeassistant.exceptions import HomeAssistantError


class EntityNotFound(HomeAssistantError):
    """When referenced entity not found."""

    def __init__(self, entity_id: str) -> None:
        """Initialize error."""
        super().__init__(self, f"entity {entity_id} not found")
        self.entity_id = entity_id

    def __str__(self) -> str:
        """Return string representation."""
        return f"Unable to find entity {self.entity_id}"


class EntityNotExposed(HomeAssistantError):
    """When referenced entity not exposed."""

    def __init__(self, entity_id: str) -> None:
        """Initialize error."""
        super().__init__(self, f"entity {entity_id} not exposed")
        self.entity_id = entity_id

    def __str__(self) -> str:
        """Return string representation."""
        return f"entity {self.entity_id} is not exposed"


class CallServiceError(HomeAssistantError):
    """Error during service calling"""

    def __init__(self, domain: str, service: str, data: object) -> None:
        """Initialize error."""
        super().__init__(
            self,
            f"unable to call service {domain}.{service} with data {data}. 'entity_id' is required",
        )
        self.domain = domain
        self.service = service
        self.data = data

    def __str__(self) -> str:
        """Return string representation."""
        return f"unable to call service {self.domain}.{self.service} with data {self.data}. 'entity_id' is required"


class FunctionNotFound(HomeAssistantError):
    """When referenced function not found."""

    def __init__(self, function: str) -> None:
        """Initialize error."""
        super().__init__(self, f"function '{function}' does not exist")
        self.function = function

    def __str__(self) -> str:
        """Return string representation."""
        return f"function '{self.function}' does not exist"


class NativeNotFound(HomeAssistantError):
    """When native function not found."""

    def __init__(self, name: str) -> None:
        """Initialize error."""
        super().__init__(self, f"native function '{name}' does not exist")
        self.name = name

    def __str__(self) -> str:
        """Return string representation."""
        return f"native function '{self.name}' does not exist"
