"""Scheme bookkeeping: loop length, snapshots, determinism, fresh starts."""

import numpy as np
import pytest

from mras.live import build


def test_fisher_micro_run_bookkeeping():
    # tend=0.003 with dt=1e-3: exactly three steps, ending at tend.
    problem = build("fisher", h=0.4, dt=1e-3, tend=0.003)
    result = problem.run([0.0], progress=False)[-1]
    # Error histories carry one entry at t=0 plus one per step.
    assert len(result.error_u) == 4
    assert len(result.error_q) == 4
    assert result.noise_level == 0.0
    assert result.state.t.Get() == 0.003
    # The series keeps t=0 and the steps 1 and 3 (the first and the last);
    # the saved list only t=0, as no step is a multiple of saved_every.
    assert len(result.series_gfu) == 3
    assert len(result.saved_gfu) == 1
    assert result.final_error_u > 0.0


def test_initial_errors_vanish_at_t0():
    # Levels start from the exact initial state, so the t=0 errors are zero.
    problem = build("darcy", h=0.2, dt=1e-3, tend=0.005, seed=3)
    result = problem.run([0.0], progress=False)[-1]
    assert result.error_u[0] == 0.0
    assert result.error_q[0] > 0.0  # the initial guess is the disk indicator


def test_seeded_darcy_is_reproducible():
    def run():
        problem = build("darcy", h=0.2, dt=1e-3, tend=0.005, seed=3)
        return problem.run([0.1], progress=False)[-1]

    r1, r2 = run(), run()
    # Identical to round-off: threaded assembly sums in a varying order.
    np.testing.assert_allclose(r1.error_u, r2.error_u, rtol=1e-10, atol=0)
    np.testing.assert_allclose(r1.error_q, r2.error_q, rtol=1e-10, atol=0)


def test_levels_are_fresh_starts():
    problem = build("darcy", h=0.2, dt=1e-3, tend=0.005, seed=3)
    chained = problem.run([0.0, 0.05], progress=False)
    assert [r.noise_level for r in chained] == [0.0, 0.05]
    assert chained[0].state is not None
    solo = problem.run_level(0.05, progress=False)
    np.testing.assert_allclose(chained[-1].error_u, solo.error_u, rtol=1e-10,
                               atol=0)
    np.testing.assert_allclose(chained[-1].error_q, solo.error_q, rtol=1e-10,
                               atol=0)


def test_a_schur_solve_that_does_not_converge_falls_back_to_the_joint_solve(
        monkeypatch):
    # Starved of iterations, CG cannot reach round-off on the Schur system;
    # the step must notice and take the factorised joint solve -- the same
    # step, to round-off -- instead of the unconverged iterate.
    import mras.solvers
    reference = build("fisher", h=0.4, dt=0.05, tend=0.25).run_level(
        0.0, progress=False)
    assert not reference.state.joint_steps
    monkeypatch.setattr(mras.solvers, "MAXIT", 1)
    starved = build("fisher", h=0.4, dt=0.05, tend=0.25).run_level(
        0.0, progress=False)
    assert starved.state.joint_steps
    np.testing.assert_allclose(starved.error_q, reference.error_q, rtol=1e-8)


def test_the_user_sets_the_initial_estimate():
    # q0 overrides the example's own start (Fisher's box indicator).
    from mras.fields import l2norm
    problem = build("fisher", h=0.4, dt=0.05, tend=0.05)
    problem.q0 = 0.4
    result = problem.run([0.0], progress=False)[-1]
    q = problem.q_true
    want = l2norm(problem.project_q(0.4) - q, problem.mesh) / l2norm(q, problem.mesh)
    # the run integrates under the task manager, in another order
    assert result.error_q[0] == pytest.approx(want, rel=1e-10)


def test_a_run_stops_at_its_target_error_or_its_time_limit():
    # The potential's estimate converges fast: with a target it stops at
    # the first step that reaches it; with target 0 it runs to tend.
    problem = build("potential", h=0.25, dt=0.05, tend=0.5)
    problem.target = 0.5
    result = problem.run_level(0.0, progress=False)
    assert result.meta["stopped"] == "target"
    assert result.error_q[-1] <= 0.5 < result.error_q[-2]
    assert result.meta["t_end"] < 0.5
    problem.target = 0.0
    result = problem.run_level(0.0, progress=False)
    assert result.meta["stopped"] == "tend"
    assert result.meta["t_end"] == pytest.approx(0.5)


def test_the_default_target_lies_just_above_the_floor():
    from mras.scheme import TARGET_FLOOR
    problem = build("fisher", h=0.4, dt=0.05, tend=0.05)
    floor = problem.q_floor()
    assert problem.stop_target() == TARGET_FLOOR * floor


def test_the_model_fit_lies_above_the_l2_floor_and_notices_a_stall():
    # The model fit is an estimate in Q_h, so it is never closer to q than
    # the L2 best fit; an error that fell by less than STALL has stalled.
    from mras.floor import STALL
    problem = build("fisher", h=0.2, dt=0.05, tend=3.0, q_h=0.15)
    problem.target = 0.0
    result = problem.run_level(0.0, progress=False)
    fit = result.state.model_fit
    assert [round(t, 6) for t, _ in fit.values] == [2.5, 3.0]  # step 50, end
    assert result.meta["model_fit"] >= result.meta["q_floor"] * (1 - 1e-9)
    fit.update(60, 3.0, 0.5)
    fit.update(60, 3.0, 0.5 * (1 - STALL / 2))
    assert fit.stalled
    fit.update(60, 3.0, 0.4)
    assert not fit.stalled
