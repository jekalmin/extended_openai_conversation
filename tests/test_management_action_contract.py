"""Run the reviewed Management action contract in ordinary CI."""

from tests_stress.test_management_action_inventory import (
    test_management_actions_have_reviewed_evidence as check_management_actions,
)


def test_every_management_action_has_semantics_authorization_and_evidence() -> None:
    check_management_actions()
