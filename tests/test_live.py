"""The example runner, the figure, the live view and the movie."""

import re
import subprocess

import numpy as np
import pytest

from mras.checkpoint import load
from mras.live import (EXAMPLES_DIR, LivePanels, Panels, _MovingRange,
                       _PinnedRange, build, example_registry, list_examples,
                       run_example, run_live)
from mras.movie import pace, render_movie

MICRO = dict(h=0.6, dt=0.1, tend=0.5)  # five assimilation steps


def test_on_step_fires_for_initial_and_every_step():
    problem = build("burgers_1d", **MICRO)
    steps = []
    problem.on_step = lambda state: steps.append(state.step)
    result = problem.run_level(0.0, progress=False)
    # One initial frame, then one frame per solved step, to tend exactly.
    assert steps == list(range(problem.tsteps + 1))
    assert result.meta["steps"] == problem.tsteps


def _followed(problem):
    """A run with the live view following it; the view and the result."""
    panels = LivePanels(problem)
    problem.on_step = panels.update
    result = problem.run_level(0.0, progress=False)
    panels.close(result.state)
    return panels, result


def test_panels_follow_the_paper_layout_and_the_final_state():
    problem = build("burgers_1d", **MICRO)
    panels, result = _followed(problem)
    titles = [ax.get_title() for _, ax, _ in panels._animated + panels._static]
    assert titles == [r"forcing $g$",
                      r"state $u$", r"data error $u - z$", r"datum $z$",
                      r"estimate $\hat q$", r"reconstruction error $\hat q - q$",
                      r"true parameter $q$"]
    assert panels.painted >= 1  # at least the final state
    state = result.state
    assert np.allclose(panels.u.vec.FV().NumPy(), state.gfu.vec.FV().NumPy())
    assert np.allclose(panels.z.vec.FV().NumPy(), state.z.vec.FV().NumPy())
    assert np.allclose(panels.qhat.vec.FV().NumPy(),
                       state.gfq.vec.FV().NumPy())
    assert np.allclose(
        panels.err_u.vec.FV().NumPy(),
        state.gfu.vec.FV().NumPy() - state.z.vec.FV().NumPy())
    # The forcing panel shows the g the last step was driven by, g(t^{n+1}),
    # even though the solver's own source term has moved on a step -- the
    # coefficient itself, not a projection of it.
    g = problem.source(state.t.Get())
    for point in ((0.7, 0.3), (2.2, 0.9), (5.1, 0.6)):
        mip = problem.mesh(*point)
        assert panels.g.field()(mip) == pytest.approx(g(mip))


def test_estimate_and_truth_share_the_padded_range_of_q():
    # q and qhat are painted over q's extremes padded by 5 % of its span at
    # each end, always: q clear of the ends, qhat comparable colour for
    # colour, an over- or undershoot saturated.
    panels, result = _followed(build("burgers_1d", **MICRO))
    q_lo, q_hi = (float(panels._payload(panels.q_true)[key])
                  for key in ("funcmin", "funcmax"))
    assert 0.33 < q_lo < 0.34 and 0.54 < q_hi < 0.55  # the wake field's ends
    lo, hi = panels._q_kw["vmin"], panels._q_kw["vmax"]
    pad = 0.05 * (q_hi - q_lo)
    assert lo == pytest.approx(q_lo - pad) and hi == pytest.approx(q_hi + pad)
    job = panels._state_job(result.state)
    (qhat,) = [kw for (ax, payload, kw), (gf, _, _)
               in zip(job["panels"], panels._animated) if gf is panels.qhat]
    assert (qhat["vmin"], qhat["vmax"]) == (lo, hi)
    assert panels._payload(panels.qhat)["funcmin"] < lo  # saturates


def test_fields_of_either_sign_are_painted_symmetrically_about_zero():
    # Beams that heat and cool: forcing and state of either sign.
    panels, _ = _followed(build("burgers_1d", signed=True, **MICRO))
    for kind in ("forcing", "state", "state_err", "param_err"):
        kw = panels._ranges[kind].kwargs()
        assert kw["vmin"] == pytest.approx(-kw["vmax"]), kind
    # A guaranteed positive state and source keep their own extremes: the
    # template's, and the light beams' (the default).
    for positive in (build("blank_reaction", tend=0.2),
                     build("burgers_1d", **MICRO)):
        panels = LivePanels(positive)
        assert not panels._ranges["state"].symmetric
        assert not panels._ranges["forcing"].symmetric


def test_sustained_fields_keep_their_largest_values_as_range():
    track = _PinnedRange(patience=5.0, tau=5.0, step=0.1, symmetric=True)
    track.add(0.0, 0.0, 0.0)  # a field at rest says nothing
    assert track.kwargs() == {"autoscale": True}
    track.add(1.0, -2.0, 1.0)  # pinned to the largest magnitude, both ways
    assert (track.kwargs()["vmin"], track.kwargs()["vmax"]) == (-2.0, 2.0)
    for t in np.arange(1.1, 6.0, 0.1):  # quiet, but not for the patience
        track.add(t, -0.5, 0.5)
    assert track.kwargs()["vmax"] == 2.0
    for t in np.arange(6.0, 40.0, 0.1):  # quiet for longer: it follows down
        track.add(t, -0.5, 0.5)
    # ... gently, and only until the field is within 90 % of it again
    assert 0.5 / 0.9 * 0.99 < track.kwargs()["vmax"] < 0.5 / 0.9 * 1.05
    track.add(40.0, -3.0, 0.0)  # a new maximum pins it again
    assert track.kwargs()["vmax"] == 3.0
    fresh = track.fresh()
    assert fresh.kwargs() == {"autoscale": True}


def test_moving_ranges_follow_slowly_and_start_at_the_first_extremes():
    track = _MovingRange(tau=10.0, step=0.1)
    track.add(0.0, 0.0, 0.0)  # a field at rest says nothing
    assert track.kwargs() == {"autoscale": True}
    track.add(0.1, -1.0, 2.0)
    assert track.kwargs()["vmin"] == pytest.approx(-1.0)
    assert track.kwargs()["vmax"] == pytest.approx(2.0)
    for t in np.arange(1.0, 100.0):
        track.add(t, -1.0, 2.0)
    track.add(100.0, -3.0, 4.0)  # a jump: a tenth of tau goes a tenth of the way
    moved = 1.0 - np.exp(-0.1)
    assert track.kwargs()["vmin"] == pytest.approx(-1.0 - 2.0 * moved, rel=1e-3)
    assert track.kwargs()["vmax"] == pytest.approx(2.0 + 2.0 * moved, rel=1e-3)
    for t in np.arange(101.0, 200.0):  # many tau later it has arrived
        track.add(t, -3.0, 4.0)
    assert track.kwargs()["vmin"] == pytest.approx(-3.0, rel=1e-3)
    assert track.kwargs()["vmax"] == pytest.approx(4.0, rel=1e-3)
    errors = _MovingRange(tau=10.0, step=0.1, symmetric=True)
    errors.add(0.0, -0.5, 2.0)
    assert errors.kwargs()["vmin"] == pytest.approx(-2.0)
    assert errors.kwargs()["vmax"] == pytest.approx(2.0)


def test_repainting_keeps_the_layout_fixed():
    # Each repaint keys the panel's bar, built once on its own slot, to the
    # new picture: the axes count and every panel position stay put, and a
    # picture lands one to one on its axes.
    panels, result = _followed(build("burgers_1d", **MICRO))
    job = panels._state_job(result.state)
    layout = []
    for _ in range(3):
        panels.paint(job)
        layout.append((len(panels.figure.axes),
                       [tuple(ax.get_position().bounds)
                        for _, ax, _ in panels._animated]))
    assert all(entry == layout[0] for entry in layout)
    ax = panels._animated[0][1]
    box = ax.get_window_extent()
    assert (round(box.width), round(box.height)) == (panels._w, panels._h)


def test_live_view_paints_beside_the_solve_and_shows_the_newest():
    # A picture is queued whenever the painter is idle, and the newest one
    # reaches the display; close() paints the final state and stops it.
    problem = build("burgers_1d", **MICRO)
    panels = LivePanels(problem)
    shown = []

    class Display:
        @property
        def value(self):
            return shown[-1]

        @value.setter
        def value(self, png):
            shown.append(png)

    panels._image = Display()
    problem.on_step = panels.update
    result = problem.run_level(0.0, progress=False)
    panels.close(result.state)
    assert shown and shown[-1].startswith(b"\x89PNG")
    assert not panels._worker.is_alive()


def test_a_failing_painter_stops_the_run():
    problem = build("burgers_1d", **MICRO)
    panels = LivePanels(problem)

    def broken(job):
        raise ValueError("boom")

    panels.paint = broken
    problem.on_step = panels.update
    # The failure surfaces at the next step, or at close() at the latest.
    with pytest.raises(RuntimeError, match="painter failed"):
        result = problem.run_level(0.0, progress=False)
        panels.close(result.state)


def _probe(path):
    """Codec, frame rate and frame count of a movie, as ffprobe reads them,
    and the MD5 of its decoded frames."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,r_frame_rate,nb_read_frames",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True,
        check=True)
    codec, rate, frames = out.stdout.strip().split(",")
    md5 = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "md5",
                          "-"], capture_output=True, text=True, check=True)
    return codec, rate, int(frames), md5.stdout.strip()


@pytest.fixture(scope="module")
def stored_run(tmp_path_factory):
    stem = tmp_path_factory.mktemp("movie") / "run"
    run_live(build("burgers_1d", h=0.6, dt=0.1, tend=1.2), 0.0, path=stem,
             movie=False, budget=None)
    return stem.with_suffix(".h5")


def test_the_movie_has_a_frame_per_stored_step_at_any_rate(stored_run,
                                                          tmp_path):
    steps = len(load(stored_run).steps)
    codec, rate, frames, _ = _probe(render_movie(
        stored_run, tmp_path / "all.mp4", fps=120, workers=1))
    assert (codec, rate, frames) == ("h264", "120/1", steps)
    # every-th stored step, and always the last
    frames = _probe(render_movie(stored_run, tmp_path / "half.mp4", every=2,
                                 workers=1))[2]
    assert frames == len(range(0, steps, 2)) + (steps - 1) % 2


def test_the_movie_paces_itself_between_four_and_ten_seconds(stored_run,
                                                            tmp_path):
    # Few frames: four seconds; many: ten, at up to 120 fps; more than
    # 1200 stored steps are thinned evenly.
    assert pace(40) == (1, 10.0)
    assert pace(300) == (1, 30.0)
    assert pace(1200) == (1, 120.0)
    every, fps = pace(3001)
    assert every == 3 and fps == 100.1  # 1001 frames, the last kept
    steps = len(load(stored_run).steps)  # 13: four seconds
    codec, rate, frames, _ = _probe(render_movie(
        stored_run, tmp_path / "paced.mp4", workers=1))
    assert (codec, frames) == ("h264", steps)
    num, den = map(int, rate.split("/"))
    assert num / den == pytest.approx(steps / 4.0, rel=1e-3)


def test_parallel_rendering_paints_the_same_frames(stored_run, tmp_path):
    serial = _probe(render_movie(stored_run, tmp_path / "1.mp4", workers=1))
    parallel = _probe(render_movie(stored_run, tmp_path / "2.mp4", workers=2))
    assert serial == parallel  # the decoded frames, bit for bit


def test_run_live_takes_explicit_step_and_horizon(tmp_path):
    # dt and tend are named parameters of the live call, not hidden kwargs:
    # the notebook shows them in the signature, and DEFAULTS do not silently
    # override them.
    import inspect
    assert {"dt", "tend"} <= set(inspect.signature(run_live).parameters)
    result = run_live("burgers_1d", 0.0, h=0.6, dt=0.1, tend=0.5,
                      path=tmp_path / "dt", workers=1)
    assert result.meta["steps"] == 5  # DEFAULTS would give thousands of steps


def test_run_live_restores_the_hook_and_returns_results(tmp_path):
    problem = build("burgers_1d", **MICRO)
    result = run_live(problem, 0.0, live=True, path=tmp_path / "hook",
                      workers=1)
    assert result.noise_level == 0.0
    assert result.meta["steps"] == 5
    assert problem.on_step is None


def test_run_live_writes_a_movie_and_a_checkpoint(tmp_path):
    stem = tmp_path / "micro"
    result = run_live("burgers_1d", 0.0, path=stem, workers=1, **MICRO)
    assert result.noise_level == 0.0
    codec, _, frames, _ = _probe(stem.with_suffix(".mp4"))
    assert codec == "h264" and frames == result.meta["steps"] + 1
    assert stem.with_suffix(".h5").exists()


def test_catalog_registers_every_example():
    assert set(list_examples()) >= {
        "darcy", "fisher", "burgers_1d", "arrhenius", "arrhenius_linear",
        "potential", "allen_cahn", "blank_reaction"}


def test_template_is_runnable_but_not_audited():
    # blank_reaction is the dogfood path, not a checklist entry.
    assert "blank_reaction" in list_examples()
    assert "blank_reaction" not in example_registry()


def test_unknown_example_lists_the_options():
    with pytest.raises(ValueError, match="burgers_1d"):
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
    results = run_example("burgers_1d", config=config, out=str(tmp_path))
    assert [r.noise_level for r in results] == [0.0]
    assert (tmp_path / "errors.png").exists()
    assert (tmp_path / "final_fields.png").exists()
    assert any(p.name.startswith("burgers_1d_0")
               for p in tmp_path.glob("*.npz"))


def test_the_error_strip_spans_the_curve_and_shows_the_floor_when_near():
    # The axis spans the curve, lowest to largest error; the L2 floor,
    # dotted, shows once the curve is within a factor two of it, and the
    # axis then reaches down to it.
    panels = Panels(build("burgers_1d", **MICRO))
    times = np.arange(4) * panels.prob.dt
    floor = panels.floor
    for history, shown in (([1.0, 0.6, 0.3, 4 * floor], set()),
                           ([1.0, 0.5, 0.3, 1.5 * floor], {"l2"})):
        panels.paint({"t": 0.0, "title": "", "panels": [], "times": times,
                      "history": np.array(history)})
        low, high = panels._strip.get_ylim()
        assert high == pytest.approx(max(history))
        lowest = min([min(history)] + [floor / 1.03 for _ in shown])
        assert low == pytest.approx(lowest)
        assert {k for k, line in panels._floors.items()
                if line.get_visible()} == shown

def test_the_time_axis_grows_until_the_end_is_known():
    # Live, the run's end is not known: the axis ends at the time shown.
    # A movie knows it, and the axis ends there.
    live = Panels(build("burgers_1d", **MICRO))
    for t in (0.3, 2.5, 7.0):
        assert live._axis_end(t) == pytest.approx(t)
    assert live._axis_end(0.0) == pytest.approx(0.1)  # one step, at the start
    movie = Panels(build("burgers_1d", **MICRO), horizon=0.4)
    assert movie._axis_end(0.1) == pytest.approx(0.4)
