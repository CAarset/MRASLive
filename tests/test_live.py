"""Live helper: the scheme hook fires, panels throttle and track the run."""

import numpy as np
import pytest

from mras.live import EXAMPLES, LivePanels, build, run_live

MICRO = dict(h=0.6, dt=0.1, tend=0.5)  # three assimilation steps


def test_on_step_fires_for_initial_and_every_step():
    problem = build("burgers", **MICRO)
    counters = []
    problem.on_step = lambda state: counters.append(state.counter)
    result = problem.run_level(0.0, progress=False)
    # One initial frame at counter_start, then one frame per solved step.
    assert counters == list(range(1, result.meta["steps"] + 2))


def test_panels_throttle_and_track_the_final_state():
    problem = build("burgers", **MICRO)
    panels = LivePanels(problem, every=2)
    redraws = []
    for scene in panels.top + panels.bottom:
        scene.Redraw = lambda s=scene, n=redraws: n.append(s.caption)
    problem.on_step = panels.update
    result = problem.run_level(0.0, progress=False)
    # Initial frame (n=1) skipped, steps n=2 and n=4 update: 2 rounds of 6.
    assert len(redraws) == 12
    panels.update(result.state, force=True)
    assert len(redraws) == 18

    state = result.state
    assert np.allclose(panels.z.vec.FV().NumPy(), state.z.vec.FV().NumPy())
    assert np.allclose(panels.qhat.vec.FV().NumPy(),
                       state.gfq.vec.FV().NumPy())
    assert np.allclose(
        panels.err_u.vec.FV().NumPy(),
        state.gfu.vec.FV().NumPy() - state.z.vec.FV().NumPy())
    # Forward-model data exposes the final-time truth before the run starts.
    assert not panels._final_tracked


def test_run_live_restores_the_hook_and_returns_results():
    result = run_live("burgers", 0.0, every=2, **MICRO)
    assert result.noise_level == 0.0
    assert result.meta["steps"] == 3


def test_burgers_is_the_first_example_option():
    assert next(iter(EXAMPLES)) == "burgers"


def test_unknown_example_lists_the_options():
    with pytest.raises(ValueError, match="burgers"):
        build("nope")
