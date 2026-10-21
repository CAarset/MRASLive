"""PDEBench Darcy loading: shapes and admissible values."""

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


def test_other_samples_load():
    grid = load_nu_grid(sample=1)
    assert grid.shape == (128, 128)
    assert 1.0 in np.unique(grid)
