"""Plotting helpers, rendered headless to PNG."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from mras.fields import to_gridfunc
from mras.plotting import error_figure, evolution_figure, sample_field
from mras.problems.darcy import DarcyProblem


@pytest.fixture(scope="module")
def micro_result():
    prob = DarcyProblem(h=0.3, dt=1e-3, tend=0.005, seed=0)
    return prob.run(levels=[0.0], progress=False)[-1]


def test_sample_field_constant(small_darcy):
    gf = to_gridfunc(np.full((6, 6), 2.0), small_darcy.Ufes)
    _, _, Z = sample_field(gf, ((0, 0), (1, 1)), n=16)
    assert np.allclose(Z, 2.0)


def test_error_figure(tmp_path, micro_result):
    path = tmp_path / "errors.png"
    fig = error_figure([micro_result], 1e-3, fname=path)
    assert path.exists() and path.stat().st_size > 1000
    assert len(fig.axes) == 2


def test_evolution_figure(tmp_path, micro_result):
    path = tmp_path / "panels.png"
    rows = [[micro_result.state.gfu], [micro_result.state.gfq]]
    evolution_figure(rows, ((0, 0), (1, 1)), col_labels=["t=T"],
                     row_labels=["u", "q"], fname=path)
    assert path.exists() and path.stat().st_size > 1000
