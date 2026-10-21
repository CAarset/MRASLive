"""Scheme bookkeeping: loop length, snapshots, determinism."""

import numpy as np

from mras.problems import DarcyProblem, FisherProblem


def test_fisher_micro_run_bookkeeping():
    # tend=0.003 with dt=1e-3 and stop_margin=2 keeps exactly one step.
    problem = FisherProblem(h=0.4, dt=1e-3, tend=0.003)
    result = problem.run([0.0], progress=False)[-1]
    assert len(result.error_u) == 1
    assert len(result.error_q) == 1
    assert result.noise_level == 0.0
    # Series snapshot at counter 1; no 100-step saves, and Fisher starts
    # without seeding the saved lists.
    assert len(result.series_gfu) == 1
    assert len(result.saved_gfu) == 0
    assert result.final_error_u > 0.0


def test_seeded_darcy_is_reproducible():
    def run():
        problem = DarcyProblem(h=0.2, dt=1e-3, tend=0.005, seed=3)
        return problem.run([0.1], progress=False)[-1]

    r1, r2 = run(), run()
    assert np.array_equal(r1.error_u, r2.error_u)
    assert np.array_equal(r1.error_q, r2.error_q)


def test_levels_chain_state():
    problem = FisherProblem(h=0.4, dt=1e-3, tend=0.003)
    results = problem.run([0.0, 0.05], progress=False)
    assert [r.noise_level for r in results] == [0.0, 0.05]
    assert results[0].state is not None
