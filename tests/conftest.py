"""Shared fixtures: small meshes reused across tests."""

import pytest

from mras.domains import darcy


@pytest.fixture(scope="session")
def small_darcy():
    """Coarse Darcy mesh/spaces for numerical helper tests."""
    return darcy(maxh=0.2, order_u=1, order_q=0)
