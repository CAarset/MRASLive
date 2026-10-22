"""Blank example: copy this file, rename it, and edit the four TODO blocks.

The worked-out model is recovery of a reaction coefficient in

    Dt u - Laplace u + q(x) u = g(x, t),   u = 0 on the boundary, u(0) = 0.

Run it with ``python examples/blank_problem.py``: it identifies q from the
noise-free datum and from a noise level 5% datum (``noise_level=0.05``),
prints the theory checklist verdicts, and writes figures to
results/blank_reaction/.

What the framework requires of a problem -- everything else is yours:

1. ``forms(state)``   the joint weak form of one step of scheme (4);
2. one data hook      ``observations(state)`` for measured data,
                      ``truth(state)`` for an analytic reference state, or
                      ``forward_forms(step)`` to generate the datum by
                      solving the model at ``q_true`` (used here);
3. ``coupling(q, z, s)`` and ``coupling_adjoint(q, z, s)`` the q-dependent
   part of the coupling f as weak functionals (return the integrand
   without ``* dx``; the checklist appends it).  Here f(q, z) = -Laplace z
   + q z: the z-only Laplace term cancels in every difference the
   checklist forms, so only ``q z s`` belongs in coupling().

Comment out ``coupling`` below and the constructor raises ``TheorySpecError``
naming exactly what is missing; drop ``coupling_adjoint`` and the ladder
still runs but reports its pairing check as unverifiable.
"""

from math import pi
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netgen.geom2d import SplineGeometry
from ngsolve import (BilinearForm, GridFunction, H1, L2, LinearForm, Mesh, dx,
                     exp, grad, sin, x as xc, y as yc)

from mras.domains import Domains
from mras.plotting import draw_field
from mras.scheme import Problem
from mras.verify import run_checks

BOX = ((0.0, 0.0), (10.0, 10.0))  # domain; also used for the figures


# --- TODO 1: mesh and trial pair ---------------------------------------------
def plate(maxh, order=3):
    """Mesh on a square plate, homogeneous Dirichlet on U_h, P0 in L2 for Q_h.

    Any netgen/NGSolve mesh works; Q_h = L2 of order 0 is the paper's choice.
    A wide plate keeps the first Dirichlet eigenvalue small, which is what
    makes the gradient flow of scheme (4) converge within a short demo.
    """
    geo = SplineGeometry()
    geo.AddRectangle(*BOX, bcs=("wall",) * 4)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    Ufes = H1(mesh, order=order, dirichlet="wall")
    Qfes = L2(mesh, order=0)
    return Domains(mesh, Ufes, Qfes)


class ReactionProblem(Problem):
    """Identify the reaction coefficient q from state data."""

    name = "blank_reaction"
    sigma = 0.0    # f is affine in q here; the checklist proves it by probe
                   # and demands sigma = 0 (set sigma = 1.0 for a nonlinear f)
    C_scale = 1.0  # tracking operator C = -Laplace (cancels the -Laplace z
                   # carried by f in the state row, as in the paper's examples)

    def __init__(self, h=1.25, dt=0.05, tend=10.0, k_u=3, seed=0, data_h=0.7):
        super().__init__(plate(h, k_u), dt=dt, tend=tend, k_u=k_u, seed=seed)
        self.data_domains = plate(data_h)  # datum ring on a finer mesh

        # --- TODO 2: the true parameter q_true(x) ----------------------------
        self.q_true = 0.3 + 2.0 * exp(-((xc - 3.5) ** 2 + (yc - 6.0) ** 2) / 5.0)
        self.u0 = 0.0  # initial state; the estimate starts at initial_guess()

    # -- data ------------------------------------------------------------------
    # --- TODO 3: the source g(x, t) -------------------------------------------
    def source(self, t):
        # Nonnegative and never switched off: persistent excitation of the
        # parameter is what makes q identifiable from the datum.
        bump = exp(-((xc - 5.0) ** 2 + (yc - 5.0) ** 2) / 8.0)
        return 2.0 * (1.0 + 0.5 * sin(2 * pi * t)) * bump
    # ---------------------------------------------------------------------------

    def forward_forms(self, step):
        """Backward Euler at q = q_true: one step of the data-generating model."""
        u, v, u_prev = step.u, step.v, step.u_prev
        a = BilinearForm(step.fes)
        res = LinearForm(step.fes)
        a += u * v * dx
        a += step.dt * grad(u) * grad(v) * dx
        a += step.dt * self.q_true * u * v * dx
        res += u_prev * v * dx
        res += step.dt * step.g * v * dx
        return a, res

    # -- theory hooks (checked before anything is solved) ------------------------
    def coupling(self, q, z, s):
        """<f(q, z) - f(0, z), s> = int q z s dx.  f is affine in q, and the
        (A2) pairing int (qa - qb)^2 z is >= 0 because the datum z >= 0."""
        return q * z * s

    def coupling_adjoint(self, q, z, s):
        """<f_q(q, z) p, s> = int p z s dx, transcribed independently."""
        return z * s

    # --- TODO 4: the joint weak form of scheme (4) ------------------------------
    def forms(self, state):
        dt = self.dt
        z, z_old, gfu = state.z, state.z_old, state.gfu
        a = BilinearForm(self.X)
        res = LinearForm(self.X)
        # State row: Dt u + f(q, z) + C(u - z) = g with f(q, z) = -Laplace z
        # + q z and C = -Laplace; the two Laplacians cancel in the weak form.
        a += self.u * self.v * dx
        a += dt * grad(self.u) * grad(self.v) * dx
        a += dt * self.q * z * self.v * dx
        res += gfu * self.v * dx
        res += dt * state.g * self.v * dx
        # Parameter row: f is affine in q (sigma = 0), so Dt q =
        # f_q(., z)^* (u - z) = z (u - z), lagged to the previous step.
        a += self.q * self.s * dx
        res += state.gfq * self.s * dx
        res += dt * z_old * (gfu - z_old) * self.s * dx
        return a, res


def main():
    problem = ReactionProblem()
    noisy = None
    for noise_level in (0.0, 0.05):
        result = problem.run_level(noise_level, progress=False)
        kind = "noise-free" if noise_level == 0.0 else \
            f"noise level {int(100 * noise_level)}%"
        print(f"noise_level={noise_level:g} ({kind}): "
              f"err_q = {result.error_q[-1]:.3f}")
        if noise_level == 0.0:
            for c in run_checks(problem, result.state, result.error_q):
                print(f"  {c.name:26s} {c.status:12s} {c.detail}")
        else:
            noisy = result

    out = Path(__file__).resolve().parent.parent / "results" / "blank_reaction"
    out.mkdir(parents=True, exist_ok=True)
    q_true = GridFunction(problem.Qfes)
    q_true.Set(problem.q_true)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    draw_field(axes[0], noisy.state.gfq, BOX, mesh=problem.mesh,
               title="estimated q")
    draw_field(axes[1], q_true, BOX, mesh=problem.mesh, title="true q")
    fig.savefig(out / "fields.png", dpi=150, bbox_inches="tight")
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
