#!/usr/bin/env python
"""Time-dependent Burgers flow in a perforated pipe — fully generic use case.

Bringing your own problem takes three things: a mesh, one forward (data)
weak form, and one MRAS estimator weak form.  No data files and no ground
truth functions are needed: the driver generates noisy observations by
stepping ``forward_forms`` on a finer mesh.

Model (scalar Burgers with unknown diffusion field q, advection in x):

    u_t - div(q grad u) + u u_x = g        in the pipe,
    u = 0                                 on walls, holes, inlet, outlet.

The pipe is a rectangle with circular holes (CSG2d); the true parameter is
low diffusion around the holes, high elsewhere, so recovery is visible as
three wakes.  The datum z enters the estimator as in the paper's continuous
scheme (sigma = 0: the model is linear in q); the nonsymmetric joint
form runs on GMRES (``symmetric = False``).  The parameter flow gains only
where |grad z| > 0 (persistent excitation), so a grid of oscillating
heaters keeps the whole pipe in the sensing region (``source(t)``).

    .venv/bin/python examples/burgers_pipe.py --quick
    .venv/bin/python examples/burgers_pipe.py --config configs/burgers.yaml
"""

import argparse
from pathlib import Path

from ngsolve import (BilinearForm, IfPos, LinearForm, dx, exp, grad, pi, sin,
                     x, y)

from mras.config import RunConfig, build, load_yaml
from mras.domains import burgers_pipe
from mras.plotting import error_figure, evolution_figure
from mras.scheme import Problem

LENGTH, HEIGHT = 6.0, 1.2
HOLE_XS, HOLE_R = (1.5, 3.0, 4.5), 0.18
Q_WAKE, Q_FREE, Q_RADIUS = 0.15, 1.0, 0.33

# Recovery of q is driven by the persistent-excitation condition: cells
# update only where |grad z| > 0.  The excitation is therefore a grid of
# phase-shifted oscillating heaters filling the pipe, on a multi-bump
# initial state, so every wake sits inside the sensing region.
U0 = sum(exp(-((x - cx) ** 2 + (y - HEIGHT / 2) ** 2) / 0.25)
         for cx in (LENGTH * (i + 0.5) / 4 for i in range(4))) \
    * y * (HEIGHT - y) * x * (LENGTH - x)


def SOURCE(t):
    """Six heaters on a 3x2 grid, each oscillating with its own phase.

    The q-update gain scales with the heater amplitude squared; 0.9 keeps
    the explicit parameter flow contracting for every dt in this example.
    """
    return sum(
        0.9 * sin(2 * pi * (0.4 * i + 0.7 * j + 0.25) + 2 * pi * t / 2.5)
        * exp(-((x - (i + 0.5) * LENGTH / 3) ** 2
                + (y - (j + 0.5) * HEIGHT / 2) ** 2) / 0.15)
        for i in range(3) for j in range(2))


def _wake_field():
    """q_true: low diffusion in disks around the holes, high elsewhere."""
    cf = Q_FREE
    for cx in HOLE_XS:
        cf = IfPos((x - cx) ** 2 + (y - HEIGHT / 2) ** 2 - Q_RADIUS ** 2,
                   cf, Q_WAKE)
    return cf


class BurgersPipe(Problem):
    """Identify q in Burgers flow from generated noisy state observations."""

    name = "burgers_pipe"
    box = ((0.0, 0.0), (LENGTH, HEIGHT))
    sigma = 0.0
    symmetric = False  # the u u_x advection makes the joint form nonsymmetric

    def __init__(self, h=0.15, dt=1e-2, tend=5.0, k_u=3, seed=0,
                 series_counters=()):
        super().__init__(burgers_pipe(maxh=h, order_u=k_u), dt=dt, tend=tend,
                         k_u=k_u, seed=seed)
        self.series_counters = tuple(series_counters)
        self.data_domains = burgers_pipe(maxh=h / 2, order_u=k_u)
        self.u0 = U0
        self.q_true = _wake_field()

    def source(self, t):
        return SOURCE(t)

    # -- forward (data-generating) model: backward Euler, advection lagged
    def forward_forms(self, step):
        a = BilinearForm(step.fes)
        res = LinearForm(step.fes)
        a += (1 / self.dt) * step.u * step.v * dx
        a += self.q_true * grad(step.u) * grad(step.v) * dx
        a += step.u_prev * grad(step.u)[0] * step.v * dx
        res += (1 / self.dt) * step.u_prev * step.v * dx
        res += step.g * step.v * dx
        return a, res

    # -- MRAS estimator: Darcy-type datum split plus the model's own
    # advection u u_x carried on the trial with datum coefficient (the
    # advection is known model, not an unknown-coefficient term, so it has
    # no right-hand-side mirror)
    def forms(self, state):
        dt = self.dt
        z, z_old, gfu, gfq = state.z, state.z_old, state.gfu, state.gfq
        a = BilinearForm(self.X)
        a += self.u * self.v * dx
        a += dt * grad(self.u) * grad(self.v) * dx
        a += dt * self.q * grad(z) * grad(self.v) * dx
        a += dt * z * grad(self.u)[0] * self.v * dx
        a += self.q * self.s * dx
        res = LinearForm(self.X)
        res += gfu * self.v * dx
        res += dt * state.g * self.v * dx
        res += dt * grad(z) * grad(self.v) * dx
        res += gfq * self.s * dx
        res += dt * grad(z_old) * (grad(gfu) - grad(z_old)) * self.s * dx
        return a, res


def run(cfg):
    prob = build(BurgersPipe, cfg, series_counters=cfg.counters())
    results = prob.run(cfg.levels)
    out = Path(cfg.out or "results/burgers_pipe")
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'level':>6} {'err_u':>9} {'err_q':>9} {'misfit':>9}")
    for res in results:
        res.save(out)
        print(f"{res.noise_level:>6g} {res.final_error_u:>9.4f} "
              f"{res.final_error_q:>9.4f} {res.misfit_u[-1]:>9.4f}")
    error_figure(results, prob.dt, fname=out / "errors.png")
    evolution_figure([[r.state.gfu for r in results],
                      [r.state.gfq for r in results]], prob.box,
                     col_labels=[f"{r.noise_level:g}" for r in results],
                     row_labels=["u", "q"], fname=out / "final_fields.png")
    print(f"results and figures in {out}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="YAML run configuration (see mras.config)")
    ap.add_argument("--h", type=float, default=0.15)
    ap.add_argument("--dt", type=float, default=1e-2)
    ap.add_argument("--tend", type=float, default=5.0)
    ap.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.05])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="small sizes, seconds")
    args = ap.parse_args()

    if args.config:
        for cfg in load_yaml(args.config):
            run(cfg)
        return
    cfg = RunConfig(h=0.35, dt=0.02, tend=0.6, levels=[0.0, 0.1], seed=0,
                    out="results/burgers_pipe_quick") if args.quick else \
        RunConfig(h=args.h, dt=args.dt, tend=args.tend,
                  levels=args.levels, seed=args.seed)
    run(cfg)


if __name__ == "__main__":
    main()
