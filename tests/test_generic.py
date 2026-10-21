"""Tests for the generic data layer: observations, truth, forward model."""

import numpy as np
import pytest
from ngsolve import (BilinearForm, GridFunction, LinearForm, dx, exp, grad,
                     x, y)

from mras.domains import darcy
from mras.scheme import Problem

BUMP = exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.05)


class ForwardProblem(Problem):
    """u_t - div(q_true grad u) = g with everything on the generic hooks.

    No data and no truth are supplied: the driver must generate observations
    by time-stepping ``forward_forms`` (backward Euler with q = q_true).
    """

    name = "forward_gen"

    def __init__(self, h=0.2, dt=2e-2, tsteps=5, seed=0):
        super().__init__(darcy(maxh=h, order_u=2, order_q=0),
                         dt=dt, tend=dt * tsteps, k_u=2, seed=seed)
        self.u0 = BUMP
        self.q_true = 0.5 + 0.25 * (x + y)

    def source(self, t):
        return BUMP * (1.0 + 0.2 * t)

    def forward_forms(self, step):
        a = BilinearForm(step.fes)
        res = LinearForm(step.fes)
        a += (1 / self.dt) * step.u * step.v * dx
        a += self.q_true * grad(step.u) * grad(step.v) * dx
        res += (1 / self.dt) * step.u_prev * step.v * dx
        res += step.g * step.v * dx
        return a, res

    def forms(self, state):
        dt = self.dt
        a = BilinearForm(self.X)
        res = LinearForm(self.X)
        a += self.u * self.v * dx
        a += dt * grad(self.u) * grad(self.v) * dx
        a += dt * self.q * grad(state.z) * grad(self.v) * dx
        a += self.q * self.s * dx
        res += state.gfu * self.v * dx
        res += dt * state.g * self.v * dx
        res += dt * grad(state.z) * grad(self.v) * dx
        res += state.gfq * self.s * dx
        res += dt * grad(state.z_old) * (grad(state.gfu) - grad(state.z_old)) * self.s * dx
        return a, res


class ObservationProblem(Problem):
    """Known data supplied directly, ground truth unknown: misfit only."""

    name = "obs_only"

    def __init__(self, h=0.2, dt=2e-2, tsteps=4, seed=0):
        super().__init__(darcy(maxh=h, order_u=2, order_q=0),
                         dt=dt, tend=dt * tsteps, k_u=2, seed=seed)
        self.u0 = BUMP

    def observations(self, state):
        return BUMP * (1.0 + 0.1 * state.t)

    def forms(self, state):
        a = BilinearForm(self.X)
        res = LinearForm(self.X)
        a += self.u * self.v * dx + self.q * self.s * dx
        res += state.gfu * self.v * dx + state.gfq * self.s * dx
        return a, res


class TruthProblem(Problem):
    """Ground truth supplied analytically: synthetic noisy data + errors."""

    name = "truth_gen"

    def __init__(self, h=0.2, dt=2e-2, tsteps=4, seed=0):
        super().__init__(darcy(maxh=h, order_u=2, order_q=0),
                         dt=dt, tend=dt * tsteps, k_u=2, seed=seed)
        self.u0 = BUMP
        self.q_true = 0.5 + 0.25 * (x + y)

    def truth(self, state):
        return BUMP * (1.0 + 0.1 * state.t)

    def forms(self, state):
        a = BilinearForm(self.X)
        res = LinearForm(self.X)
        a += self.u * self.v * dx + self.q * self.s * dx
        res += state.gfu * self.v * dx + state.gfq * self.s * dx
        return a, res


class StarvedProblem(Problem):
    name = "starved"

    def __init__(self):
        super().__init__(darcy(maxh=0.4, order_u=2, order_q=0),
                         dt=1e-1, tend=1e-1, k_u=2)

    def forms(self, state):
        a = BilinearForm(self.X)
        res = LinearForm(self.X)
        a += self.u * self.v * dx
        res += state.gfu * self.v * dx
        return a, res


def test_forward_generation_produces_errors():
    prob = ForwardProblem()
    res = prob.run([0.0], progress=False)[0]
    assert res.error_u is not None and res.error_q is not None
    assert res.error_u[0] == pytest.approx(0.0)
    assert len(res.error_u) == len(res.misfit_u) + 1  # t=0 seeding
    assert np.all(np.isfinite(res.error_u))
    # Exact data: the estimator stays close to the truth and to the data.
    assert res.error_u[-1] < 0.5
    assert res.misfit_u[-1] < 0.2
    # The generated data is time-dependent, not the initial state.
    u0 = GridFunction(prob.Ufes)
    u0.Set(prob.u0)
    ref = u0.vec.FV().NumPy().copy()
    now = np.asarray(res.state.u_true.vec.FV().NumPy())
    assert np.linalg.norm(now - ref) > 1e-3 * np.linalg.norm(ref)


def test_forward_generation_is_seeded_and_independent():
    prob = ForwardProblem()
    r1 = prob.run([0.05, 0.0], progress=False)
    r2 = ForwardProblem().run([0.05, 0.0], progress=False)
    for a, b in zip(r1, r2):
        np.testing.assert_allclose(a.error_u, b.error_u)
    # A fresh, exact-data level starts from zero error.
    assert r1[1].error_u[0] == pytest.approx(0.0)


def test_observations_only_reports_misfit_without_errors():
    prob = ObservationProblem()
    res = prob.run([0.05], progress=False)[0]
    assert res.error_u is None and res.error_q is None
    assert res.misfit_u is not None
    assert len(res.misfit_u) == prob.tsteps - 2  # stop_margin = 2 steps
    assert np.all(np.isfinite(res.misfit_u))


def test_truth_hook_adds_seeded_noise_and_errors():
    res = TruthProblem().run([0.0, 0.1], progress=False)
    for r in res:
        assert r.error_u[0] == pytest.approx(0.0)
        assert np.all(np.isfinite(r.error_u))
    assert res[1].error_u[-1] > 0.0


def test_starved_problem_raises():
    prob = StarvedProblem()
    with pytest.raises(NotImplementedError):
        prob.run([0.0], progress=False)
