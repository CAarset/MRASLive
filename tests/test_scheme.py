"""Scheme bookkeeping: loop length, snapshots, determinism, fresh starts."""

import numpy as np

from mras.problems import DarcyProblem, FisherProblem


def test_fisher_micro_run_bookkeeping():
    # tend=0.003 with dt=1e-3 and stop_margin=2 keeps exactly one step.
    problem = FisherProblem(h=0.4, dt=1e-3, tend=0.003)
    result = problem.run([0.0], progress=False)[-1]
    # Error histories carry one entry at t=0 plus one per step.
    assert len(result.error_u) == 2
    assert len(result.error_q) == 2
    assert result.noise_level == 0.0
    # Saved and series lists hold only the t=0 state: the step leaves the
    # counter at 2, which is neither a save multiple nor a series counter.
    assert len(result.series_gfu) == 1
    assert len(result.saved_gfu) == 1
    assert result.final_error_u > 0.0


def test_initial_errors_vanish_at_t0():
    # Levels start from the exact initial state, so the t=0 errors are zero.
    problem = DarcyProblem(h=0.2, dt=1e-3, tend=0.005, seed=3)
    result = problem.run([0.0], progress=False)[-1]
    assert result.error_u[0] == 0.0
    assert result.error_q[0] > 0.0  # the initial guess is the disk indicator


def test_seeded_darcy_is_reproducible():
    def run():
        problem = DarcyProblem(h=0.2, dt=1e-3, tend=0.005, seed=3)
        return problem.run([0.1], progress=False)[-1]

    r1, r2 = run(), run()
    assert np.array_equal(r1.error_u, r2.error_u)
    assert np.array_equal(r1.error_q, r2.error_q)


def test_levels_are_fresh_starts():
    problem = DarcyProblem(h=0.2, dt=1e-3, tend=0.005, seed=3)
    chained = problem.run([0.0, 0.05], progress=False)
    assert [r.noise_level for r in chained] == [0.0, 0.05]
    assert chained[0].state is not None
    solo = problem.run_level(0.05, progress=False)
    assert np.array_equal(chained[-1].error_u, solo.error_u)
    assert np.array_equal(chained[-1].error_q, solo.error_q)
