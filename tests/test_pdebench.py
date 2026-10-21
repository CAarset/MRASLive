"""PDEBench Darcy loading: shapes, admissible values, centring roll."""

import numpy as np
import pytest

from mras.data import DEFAULT_HDF5
from mras.data.pdebench import load_nu_grid

pytest.importorskip("h5py")
pytestmark = pytest.mark.skipif(not DEFAULT_HDF5.exists(),
                                reason="PDEBench hdf5 not present")


def test_default_grid_layout():
    grid = load_nu_grid()
    assert grid.shape == (128, 128)
    # Stored in float32, so widen and round before comparing the values.
    values = np.round(np.unique(grid).astype(float), 5)
    assert set(values.tolist()) <= {0.1, 1.0}


def test_roll_rows_centres_inclusion():
    # The high-contrast zone spans rows 43..123 as stored; the Darcy
    # problem's roll of 12 shifts it to rows 31..111.
    plain = load_nu_grid(roll_rows=0)
    rolled = load_nu_grid(roll_rows=12)
    for grid, lo, hi in ((plain, 43, 123), (rolled, 31, 111)):
        rows = np.flatnonzero((grid > 0.5).any(axis=1))
        assert (int(rows.min()), int(rows.max())) == (lo, hi)


def test_other_samples_load():
    grid = load_nu_grid(sample=1)
    assert grid.shape == (128, 128)
    assert 1.0 in np.unique(grid)
