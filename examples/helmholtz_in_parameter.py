#!/usr/bin/env python
"""Nonlinear-in-parameter Helmholtz demo on the mras.Problem interface.

Shows the headline workflow: bring your own mesh and weak forms. The model
here is u_t = nu Laplacian u + g(t) with the datum-driven parameter step
(sigma = 0, diffusion nu = 5, zero ground-truth parameter).

    .venv/bin/python -m mras.live helmholtz --quick


CLEANUP-FLAG (pre-push): placeholder for the scalar-q Helmholtz
rework (docs/ASSUMPTIONS.md section 6); deleting it also drops the
runner registration below and the tests/test_verify.py row.
"""

from ngsolve import BilinearForm, GridFunction, LinearForm, Parameter
from ngsolve import dx, exp, grad, x, y

from mras.domains import darcy
from mras.scheme import Problem
from mras.stepper import ImplicitStepper

SOURCE = 2 * exp(-((x - 0.5) ** 2 / 0.05 + (y - 0.45) ** 2 / 0.1))


class HelmholtzProblem(Problem):
    """Identify q in u_t = nu Laplacian u + g, with the quadratic q-step."""

    name = "helmholtz"
    sigma = 0.0
    diffusion = 5.0  # the model's Laplacian coefficient
    C_scale = 5.0  # tracking operator C = 5 * (-Laplace)
    # No q in the truth model, so the coupling ladder is vacuous and the
    # q-update is the paper's quadratic self-step (mras/verify.py reports
    # UNVERIFIABLE for every rung).
    theory_exempt = "parameter absent from the truth model"
    adjoint_deviation = ("parameter absent from the truth model: no adjoint "
                         "pair exists by construction (quadratic self-step)")

    def __init__(self, h=0.08, tend=1.5, tsteps=1000, k_u=3, seed=0):
        super().__init__(darcy(maxh=h, order_u=k_u, order_q=0),
                         dt=tend / tsteps, tend=tend, k_u=k_u, seed=seed)
        self.q_true = 0.0  # the ground-truth parameter is identically zero
        self.tpar = Parameter(0.0)
        self.coef_0, self.u_exact = self._make_data(h)

    def _make_data(self, h):
        """Ground truth: implicit solves of Laplacian u = source on a finer mesh."""
        V = darcy(maxh=0.07, order_u=2, order_q=0).Ufes
        u, v = V.TrialFunction(), V.TestFunction()
        a = BilinearForm(V)
        a += grad(u) * grad(v) * dx
        a.Assemble()
        stepper = ImplicitStepper(self.dt, a)
        gf = GridFunction(V)
        gf.Set(SOURCE)
        coef_0 = GridFunction(self.Ufes)
        coef_0.Set(gf)
        u_exact = []
        for counter in range(self.tsteps):
            self.tpar.Set(counter * self.dt)
            stepper.step(gf, SOURCE * exp(-self.tpar))
            snap = GridFunction(self.Ufes)
            snap.Set(gf)
            u_exact.append(snap)
        return coef_0, u_exact

    def init_run(self, state):
        state.gfu.Set(self.coef_0)
        state.u_true.Set(self.coef_0)  # exact state at t = 0
        state.gfq.Set(0.0)

    def truth(self, state):
        return self.u_exact[state.counter - 1]

    def forms(self, state):
        z, z_old, gfu, gfq = state.z, state.z_old, state.gfu, state.gfq
        g = SOURCE * exp(-state.t)
        dt, nu = self.dt, self.diffusion
        a = BilinearForm(self.X)
        a += self.u * self.v * dx
        a += dt * nu * grad(self.u) * grad(self.v) * dx
        a += -dt * gfq * self.q * z * self.v * dx
        a += self.q * self.s * dx
        res = LinearForm(self.X)
        res += gfu * self.v * dx
        res += dt * g * self.v * dx
        res += -dt * grad(z_old) * grad(self.v) * dx
        res += dt * nu * grad(z_old) * grad(self.v) * dx
        res += gfq * self.s * dx
        res += -2 * dt * z_old * (gfu - z_old) * self.s * dx
        return a, res


# --- runner registration (mras.live reads these markers) ----------------------

PROBLEM = HelmholtzProblem

QUICK = CHECK = {"helmholtz": dict(h=0.2, tend=0.4, tsteps=10)}
