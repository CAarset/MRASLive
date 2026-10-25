"""The example runner and live view: hook contract, throttling, HTML movie."""

import re

import numpy as np
import pytest

from mras.live import (EXAMPLES_DIR, LivePanels, build, example_registry,
                       list_examples, run_example, run_live)

MICRO = dict(h=0.6, dt=0.1, tend=0.5)  # three assimilation steps


def test_on_step_fires_for_initial_and_every_step():
    problem = build("burgers_pipe", **MICRO)
    counters = []
    problem.on_step = lambda state: counters.append(state.counter)
    result = problem.run_level(0.0, progress=False)
    # One initial frame at counter_start, then one frame per solved step.
    assert counters == list(range(1, result.meta["steps"] + 2))


def test_panels_throttle_and_track_the_final_state():
    problem = build("burgers_pipe", **MICRO)
    panels = LivePanels(problem, every=2, live=True)
    redraws = []
    real_render = panels.render
    panels.render = lambda: (redraws.append(1), real_render())
    problem.on_step = panels.update
    result = problem.run_level(0.0, progress=False)
    # n=1 (t = 0) is never skipped by the throttle; then n=2 and n=4.
    assert len(redraws) == 3
    assert len(panels.frames) == 3
    panels.update(result.state, force=True)
    assert len(redraws) == 4
    assert len(panels.frames) == 4

    state = result.state
    assert np.allclose(panels.z.vec.FV().NumPy(), state.z.vec.FV().NumPy())
    assert np.allclose(panels.qhat.vec.FV().NumPy(),
                       state.gfq.vec.FV().NumPy())
    assert np.allclose(
        panels.err_u.vec.FV().NumPy(),
        state.gfu.vec.FV().NumPy() - state.z.vec.FV().NumPy())
    # Forward-model data exposes the final-time truth before the run starts.
    assert not panels._final_tracked


def test_repainting_keeps_the_panels_their_own_size():
    # A colourbar carves its room out of the panel's grid cell; ngsplot hands
    # the cell back on rebuilds and reuses the standing bar across repaints,
    # so repainting must leave the panel exactly as wide as the first bar
    # left it.
    problem = build("burgers_pipe", **MICRO)
    panels = LivePanels(problem, every=1, live=False)
    widths = []
    for _ in range(3):
        panels.render()
        widths.append([ax.get_position(original=True).width
                       for _, ax, _ in panels._animated])
    assert np.allclose(widths, widths[0])


def test_updates_thin_to_a_fixed_count_at_any_length():
    # The default stride is read off the run length: a long horizon stays
    # a movie of about max_frames frames, evenly spaced, instead of a
    # slow slideshow of every thousandth step.
    problem = build("burgers_pipe", h=0.6, dt=0.1, tend=2.0)
    panels = LivePanels(problem, live=False, max_frames=4)
    assert 4 <= panels.every <= 6  # proportional to the step count
    counters = []
    real_update = panels.update

    def spy(state, force=False):
        counters.append(state.counter)
        return real_update(state, force)

    problem.on_step = spy
    problem.run_level(0.0, progress=False)
    rendered = [c for c in counters if c == 1 or c % panels.every == 0]
    assert rendered[0] == 1  # the t = 0 frame is always there
    assert len(panels.frames) == len(rendered) == 1 + problem.tsteps // panels.every
    assert len(panels.frames) <= 5  # max_frames plus the t = 0 frame


def test_the_view_opens_before_the_truth_is_generated():
    # The panels must exist and paint without the forward-truth run: the
    # view opens at t = 0, and only begin() pays for the data generation.
    problem = build("burgers_pipe", **MICRO)
    panels = LivePanels(problem, every=1, live=False)
    z_kw = panels._animated[0][2]
    assert z_kw.get("vmin") is None  # no truth consulted at construction
    panels.render()
    panels.begin()
    assert panels._animated[0][2].get("vmin") is not None
    assert panels._final_tracked in (True, False)


def test_run_live_takes_explicit_step_and_horizon(tmp_path):
    # dt and tend are named parameters of the live call, not hidden kwargs:
    # the notebook shows them in the signature, and DEFAULTS do not silently
    # override them.
    import inspect
    assert {"dt", "tend"} <= set(inspect.signature(run_live).parameters)
    result = run_live("burgers_pipe", 0.0, dt=0.1, tend=0.5,
                      path=tmp_path / "dt.html")
    assert result.meta["steps"] == 3  # DEFAULTS would give hundreds of steps


def test_run_live_restores_the_hook_and_returns_results(tmp_path):
    problem = build("burgers_pipe", **MICRO)
    result = run_live(problem, 0.0, every=2, live=True,
                      path=tmp_path / "hook.html")
    assert result.noise_level == 0.0
    assert result.meta["steps"] == 3
    assert problem.on_step is None


def test_run_movie_stitches_an_animated_html(tmp_path):
    path = tmp_path / "micro.html"
    result = run_live("burgers_pipe", 0.0, every=1, path=path, **MICRO)
    assert result.noise_level == 0.0
    html = path.read_text()
    assert "slider_id" in html  # matplotlib's animation-player script
    # The movie must actually animate: distinct PNG frames, not a still.
    # The embedded base64 is wrapped in the JS string with "\"+newline
    # continuations, so the frame blob is read up to the closing quote and
    # stripped down to base64 characters before comparing.
    frames = {re.sub(r"[^A-Za-z0-9+/=]", "", m.group(1))
              for m in re.finditer(r"data:image/png;base64,([^\"']+)", html)}
    assert len(frames) >= 2


def test_catalog_registers_every_example():
    assert set(list_examples()) >= {
        "darcy", "fisher", "burgers_pipe", "arrhenius", "arrhenius_linear",
        "potential", "allen_cahn", "helmholtz", "blank_reaction"}


def test_template_is_runnable_but_not_audited():
    # blank_reaction is the dogfood path, not a checklist entry.
    assert "blank_reaction" in list_examples()
    assert "blank_reaction" not in example_registry()


def test_unknown_example_lists_the_options():
    with pytest.raises(ValueError, match="burgers_pipe"):
        build("nope")


def test_examples_import_only_mras():
    # The runner contract: example modules never import mras.live at runtime
    # (the loader would re-enter the module being built); docstrings naming
    # the runner command are fine.
    imports = re.compile(r"^\s*(from|import)\s+\S*\.?live\b", re.M)
    for path in EXAMPLES_DIR.glob("*.py"):
        assert not imports.search(path.read_text()), path.name


def test_run_example_layering_sweeps_and_saves(tmp_path):
    config = tmp_path / "sweep.yaml"
    config.write_text(
        "defaults: {h: 0.6, dt: 0.1, tend: 0.5}\n"
        "runs:\n"
        "  - {seed: 1, noise_levels: [0.0]}\n")
    results = run_example("burgers_pipe", config=config, out=str(tmp_path))
    assert [r.noise_level for r in results] == [0.0]
    assert (tmp_path / "errors.png").exists()
    assert (tmp_path / "final_fields.png").exists()
    assert any(p.name.startswith("burgers_pipe_0")
               for p in tmp_path.glob("*.npz"))


def test_helmholtz_quick_sizes_match_the_audit():
    # The one example with no DEFAULTS: its QUICK sizes are the audit's.
    from mras.live import _locate
    module, _ = _locate("helmholtz")
    assert module.QUICK["helmholtz"] == {"h": 0.2, "tend": 0.4, "tsteps": 10}
