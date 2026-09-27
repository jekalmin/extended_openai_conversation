"""Run the route contract in ordinary PR CI, beyond nightly stress coverage."""

import unittest

from tests_stress.test_frontend_route_inventory import (
    test_every_frontend_route_has_reviewed_acceptance_level,
    test_new_route_fails_inventory_contract,
)


class RouteAcceptanceTests(unittest.TestCase):
    def test_every_route_has_named_behavioral_journey(self):
        test_every_frontend_route_has_reviewed_acceptance_level()

    def test_new_route_requires_acceptance(self):
        test_new_route_fails_inventory_contract()
