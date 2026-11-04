"""Run checkpoints: what is written, read back, and resumed."""

import h5py
import numpy as np

from mras.checkpoint import load, resume
from mras.live import build, run_live

QUICK = dict(h=0.35, dt=0.05, k_q=1)


def test_checkpoint_holds_the_run_and_its_fields(tmp_path):
    stem = tmp_path / "burgers"
    result = run_live(build("burgers_1d", tend=0.6, **QUICK), 0.0,
                      path=stem, movie=False)
    stored = load(stem.with_suffix(".h5"))
    assert stored.attrs["completed"] and stored.attrs["problem"] == "burgers_1d"
    assert stored.attrs["scheme"] == "implicit"
    np.testing.assert_allclose(stored.history["error_q"], result.error_q)
    np.testing.assert_allclose(stored.history["misfit_u"], result.misfit_u)
    # The trajectory runs from the initial state to the last step.
    assert stored.steps[0] == 0 and stored.steps[-1] == result.meta["steps"]
    q = stored.field("q", -1)  # rebuilt from the recorded arguments
    np.testing.assert_allclose(q.vec.FV().NumPy(),
                               result.state.gfq.vec.FV().NumPy(), rtol=1e-6)
    restart = stored.restart()
    np.testing.assert_array_equal(restart["gfq"],
                                  result.state.gfq.vec.FV().NumPy())


def test_a_resumed_run_continues_as_if_never_stopped(tmp_path):
    whole = build("burgers_1d", tend=1.0, **QUICK).run_level(
        0.0, progress=False)
    stem = tmp_path / "interrupted"
    run_live(build("burgers_1d", tend=0.5, **QUICK), 0.0, path=stem,
             movie=False)
    resumed = resume(stem.with_suffix(".h5"), tend=1.0)
    assert len(resumed.error_q) == len(whole.error_q)
    # Round-off apart: the resumed forward model refactorises its
    # preconditioner where it stood.
    np.testing.assert_allclose(resumed.error_q, whole.error_q, rtol=1e-8)
    np.testing.assert_allclose(resumed.error_u, whole.error_u, rtol=1e-8)
    stored = load(stem.with_suffix(".h5"))
    assert stored.attrs["completed"] and stored.attrs["tend"] == 1.0
    assert list(stored.steps) == sorted(set(stored.steps))  # no duplicates
    assert stored.steps[-1] == whole.meta["steps"]


def test_a_noisy_run_resumes_with_its_smoother_and_safeguards(tmp_path):
    """The time filter, the smoother's alpha, the running means of the
    discrepancy and the leakage's anchor are in the restart point: a noisy
    run resumed goes on as if never stopped."""
    whole = build("burgers_1d", tend=1.0, **QUICK).run_level(
        0.05, progress=False)
    stem = tmp_path / "noisy"
    run_live(build("burgers_1d", tend=0.5, **QUICK), 0.05, path=stem,
             movie=False)
    resumed = resume(stem.with_suffix(".h5"), tend=1.0)
    np.testing.assert_allclose(resumed.error_q, whole.error_q, rtol=1e-8)
    np.testing.assert_allclose(resumed.effective_noise,
                               whole.effective_noise, rtol=1e-8)


def test_the_trajectory_keeps_within_its_budget(tmp_path):
    stem = tmp_path / "thin"
    prob = build("burgers_1d", tend=1.5, **QUICK)
    per = 4 * (2 * prob.Ufes.ndof + prob.Qfes.ndof)
    budget = 6.5 * per / 2**20  # room for six snapshots
    result = run_live(prob, 0.0, path=stem, movie=False, budget=budget)
    with h5py.File(stem.with_suffix(".h5"), "r") as f:
        stride = int(f.attrs["stride"])
        steps = f["trajectory/step"][()]
    assert stride > 1
    assert steps[0] == 0 and steps[-1] == result.meta["steps"]
    assert len(steps) <= 7  # every stride-th step and the last
    # halved as the budget filled: evenly spaced, whatever the run's length
    assert set(np.diff(steps[:-1])) == {stride}
