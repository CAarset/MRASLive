#!/usr/bin/env python
"""Nonlinear-in-parameter Helmholtz demo on the mras.Problem interface.

Shows the headline workflow: bring your own mesh and weak forms. The model
here is u_t = L Laplacian u + g(t) with the datum-driven parameter step
(sigma = 0, L = 5, zero ground-truth parameter).

    .venv/bin/python examples/helmholtz_in_parameter.py --quick
"""

import argparse
from pathlib import Path

from ngsolve import BilinearForm, GridFunction, LinearForm, Parameter
from ngsolve import dx, exp, grad, x, y

from mras.domains import darcy
from mras.plotting import error_figure, evolution_figure
from mras.scheme import Problem
from mras.stepper import ImplicitStepper

SOURCE = 2 * exp(-((x - 0.5) ** 2 / 0.05 + (y - 0.45) ** 2 / 0.1))


class HelmholtzProblem(Problem):
    """Identify q in u_t = L Laplacian u + g, with the quadratic q-step."""

    name = "helmholtz"
    sigma = 0.0
    L = 5.0

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

    def init_level(self, state):
        state.gfu.Set(self.coef_0)
        state.u_true.Set(self.coef_0)  # exact state at t = 0
        state.gfq.Set(0.0)

    def truth(self, state):
        return self.u_exact[state.counter - 1]

    def forms(self, state):
        z, z_old, gfu, gfq = state.z, state.z_old, state.gfu, state.gfq
        g = SOURCE * exp(-state.t)
        dt, L = self.dt, self.L
        a = BilinearForm(self.X)
        a += self.u * self.v * dx
        a += dt * L * grad(self.u) * grad(self.v) * dx
        a += -dt * gfq * self.q * z * self.v * dx
        a += self.q * self.s * dx
        res = LinearForm(self.X)
        res += gfu * self.v * dx
        res += dt * g * self.v * dx
        res += -dt * grad(z_old) * grad(self.v) * dx
        res += dt * L * grad(z_old) * grad(self.v) * dx
        res += gfq * self.s * dx
        res += -2 * dt * z_old * (gfu - z_old) * self.s * dx
        return a, res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h", type=float, default=0.08)
    ap.add_argument("--tend", type=float, default=1.5)
    ap.add_argument("--tsteps", type=int, default=1000)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.05])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="small sizes, seconds")
    ap.add_argument("--out", default="results/helmholtz")
    args = ap.parse_args()

    if args.quick:
        args.h, args.tend, args.tsteps = 0.2, 0.03, 20
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prob = HelmholtzProblem(h=args.h, tend=args.tend, tsteps=args.tsteps,
                            seed=args.seed)
    results = prob.run(args.levels)
    for res in results:
        res.save(out)
        print(f"level {res.noise_level:g}: err_u={res.final_error_u:.4f} "
              f"err_q={res.final_error_q:.4f}")

    error_figure(results, prob.dt, fname=out / "errors.png")
    evolution_figure([[r.state.gfu for r in results],
                      [r.state.gfq for r in results]],
                     ((0, 0), (1, 1)), col_labels=[f"{r.noise_level:g}"
                                                   for r in results],
                     row_labels=["u", "q"], fname=out / "final_fields.png")
    print(f"figures and npz files in {out}")


if __name__ == "__main__":
    main()
