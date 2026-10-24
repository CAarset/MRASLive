"""Plotting helpers, rendered headless to PNG through ngsplot."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mras.fields import to_gridfunc
from mras.live import build
from mras.plotting import draw_field, error_figure, evolution_figure


@pytest.fixture(scope="module")
def micro_result():
    prob = build("darcy", h=0.3, dt=1e-3, tend=0.005, seed=0)
    return prob.run(noise_levels=[0.0], progress=False)[-1]


def test_draw_field_paints_the_fem_raster(small_darcy):
    gf = to_gridfunc(np.full((6, 6), 2.0), small_darcy.Ufes)
    fig, ax = plt.subplots()
    draw_field(gf, ax=ax, title="flat")
    # ngsplot renders the FEM picture itself: one already-coloured raster,
    # not a scalar grid handed to imshow for interpolation.
    assert len(ax.images) == 1
    assert ax.images[0].get_array().dtype == np.uint8
    assert ax.get_title() == "flat"
    fig.clf()
    plt.close(fig)


def test_redraw_replaces_the_picture(small_darcy):
    gf = to_gridfunc(np.full((6, 6), 2.0), small_darcy.Ufes)
    fig, ax = plt.subplots()
    draw_field(gf, ax=ax)
    draw_field(gf, ax=ax)
    assert len(ax.images) == 1
    fig.clf()
    plt.close(fig)


def test_error_figure(tmp_path, micro_result):
    path = tmp_path / "errors.png"
    fig = error_figure([micro_result], 1e-3, fname=path)
    assert path.exists() and path.stat().st_size > 1000
    assert len(fig.axes) == 2
    # Noise terminology: the panel says which noise level, not "level 0".
    assert "noise_level=" in fig.axes[0].get_title()
    plt.close(fig)


def test_evolution_figure(tmp_path, micro_result):
    path = tmp_path / "panels.png"
    rows = [[micro_result.state.gfu], [micro_result.state.gfq]]
    fig = evolution_figure(rows, col_labels=["t=T"], row_labels=["u", "q"],
                           fname=path)
    assert path.exists() and path.stat().st_size > 1000
    panels = [ax for ax in fig.axes if ax.images]
    assert len(panels) == 2 and len(fig.axes) == 4  # one colorbar each
    assert {ax.get_title() for ax in panels} == {"u  t=T", "q  t=T"}
    plt.close(fig)
