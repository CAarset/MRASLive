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
three wakes.  The estimator is the paper's scheme (4) with the model split
f(q, u) = -div(q grad u) + u u_x: linear in q, so sigma = 0 and the
parameter equation is the Darcy adjoint rule (Prop. 4.1) verbatim,
q_t = grad z . grad (u - z) — the advection carries no q and drops out of
the q-derivative.  Per Remark 2.1 the remaining nonlinearity is evaluated
on the datum, z z_x, so it cancels from the continuous error system; the
state row below is Darcy's weak form (18)-(19) plus the datum-side term
-(z z_x, v).  (A1) holds exactly (L = 0) and (A3) with C = -Laplace, M = 1
as for Darcy; (A2) is left unverified exactly as the paper leaves it for
Darcy (Remark 4.2), and the heater grid is its discrete persistent-
excitation surrogate.  See docs/DEVIATIONS.md for the full audit.

    .venv/bin/python examples/burgers_pipe.py --quick
    .venv/bin/python examples/burgers_pipe.py --config configs/burgers.yaml
"""

import argparse
from pathlib import Path

from ngsolve import BilinearForm, LinearForm, dx, exp, grad, pi, sin, sqrt, x, y, IfPos

from mras.config import RunConfig, build, load_yaml
from mras.domains import burgers_pipe
from mras.plotting import error_figure, evolution_figure
from mras.scheme import Problem

LENGTH, HEIGHT = 6.0, 1.2
HOLE_XS, HOLE_RADIUS = (1.5, 3.0, 4.5), 0.18
Q_WAKE, Q_FREE, Q_RADIUS = 1.0, 1.0e-1, 0.33
Q_SMOOTH = 0.0875  # wake decay scale, a quarter of the demo cell size

# Common forcing period. Every heater frequency is an integer multiple of 1/T,
# so SOURCE(t + T) == SOURCE(t) exactly.
T_FORCING = 8.0

# Source amplitude. This is deliberately much smaller than 0.9: the state is
# dissipative, so the source amplitude is not the state amplitude.
SOURCE_AMPLITUDE = 5


def SOURCE(t):
    """Six heaters on a 3x2 grid with an exact common period T_FORCING.

    Frequencies are integer multiples of 1/T_FORCING, so the source is
    exactly periodic. The phases differ by cell, preserving spatial variety.
    """
    # Harmonic numbers: 5, 4, 4, 4, 3, 3 cycles per forcing period.
    harmonics = (
        (0, 0, 5),
        (0, 1, 4),
        (1, 0, 4),
        (1, 1, 4),
        (2, 0, 3),
        (2, 1, 3),
    )

    return sum(
        SOURCE_AMPLITUDE
        * sin(2 * pi * (k * t / T_FORCING + phase))
        * exp(
            -(
                (x - (i + 0.5) * LENGTH / 3) ** 2
                + (y - (j + 0.5) * HEIGHT / 2) ** 2
            )
            / 0.15
        )
        for i, j, k, phase in (
            (i, j, k, (0.4 * i + 0.7 * j + 0.25) % 1.0)
            for i, j, k in harmonics
        )
    )


def _wake_field():
    """Structured diffusion field: smooth, positive, and visually distinct.

    Low diffusion in two horizontal wake bands; higher diffusion elsewhere.
    The field is bounded in [Q_MIN, Q_MAX] and has no sharp disk boundaries.
    """
    Q_MIN = 0.12
    Q_MAX = 0.55

    # Two smooth low-diffusion bands around y = 0.4 and y = 0.8.
    wake = exp(-((y - 0.4) ** 2) / 0.03) + exp(-((y - 0.8) ** 2) / 0.03)

    # A mild longitudinal modulation avoids an artificially uniform wake.
    wake *= 0.75 + 0.25 * sin(2 * pi * x / LENGTH)

    return Q_MIN + (Q_MAX - Q_MIN) * (1.0 - wake / (1.0 + wake))


def _periodic_state():
    """Small, smooth initial state compatible with zero Dirichlet boundaries.

    It is not an exact periodic solution, but it is small and has no large
    transient, so the state settles quickly into the source-driven cycle.
    """
    return (
        SOURCE_AMPLITUDE
        * sin(2 * pi * x / LENGTH)
        * sin(pi * y / HEIGHT)
        * sin(2 * pi * 0.0 / T_FORCING)
    )

class BurgersPipe(Problem):
    """Identify q in Burgers flow from generated noisy state observations."""

    name = "burgers_pipe"
    box = ((0.0, 0.0), (LENGTH, HEIGHT))
    # f(q, z) = -div(q grad z) + (z grad z) is affine in q: sigma = 0 and
    # L = 0 exactly; (A2) shares the Darcy situation -- mras/verify.py.
    sigma = 0.0
    L = 0.0
    C_scale = 1.0
    symmetric = False  # the q -> u coupling block makes the joint form
    # nonsymmetric (block-triangular, as for Darcy; GMRES is the safe choice)

    def coupling(self, q, z, s):
        # The convective term z z_x carries no q and cancels in every
        # difference the checklist forms; only the diffusion coupling is q-
        # dependent.
        return q * grad(z) * grad(s)

    def coupling_adjoint(self, q, z, s):
        return grad(z) * grad(s)

    def __init__(self, h=0.15, dt=1e-2, tend=10.0, k_u=3, seed=0,
                 series_counters=()):
        super().__init__(burgers_pipe(maxh=h, order_u=k_u), dt=dt, tend=tend,
                         k_u=k_u, seed=seed)
        self.series_counters = tuple(series_counters)
        self.data_domains = burgers_pipe(maxh=h / 2, order_u=k_u)
        self.u0 = _periodic_state()
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

    # -- MRAS estimator: Darcy's weak form (18)-(19) plus the model's own
    # advection evaluated on the datum (Remark 2.1: f acts on z, never on
    # the trial state; the z z_x term is known model, so it is explicit on
    # the right-hand side and drops out of the q row entirely)
    def forms(self, state):
        dt = self.dt
        z, z_old, gfu, gfq = state.z, state.z_old, state.gfu, state.gfq
        a = BilinearForm(self.X)
        a += self.u * self.v * dx
        a += dt * grad(self.u) * grad(self.v) * dx
        a += dt * self.q * grad(z) * grad(self.v) * dx
        a += self.q * self.s * dx
        res = LinearForm(self.X)
        res += gfu * self.v * dx
        res += dt * state.g * self.v * dx
        res += dt * grad(z) * grad(self.v) * dx
        res += -dt * z * grad(z)[0] * self.v * dx
        res += gfq * self.s * dx
        res += dt * grad(z_old) * (grad(gfu) - grad(z_old)) * self.s * dx
        return a, res


def run(cfg):
    prob = build(BurgersPipe, cfg, series_counters=cfg.counters())
    results = prob.run(cfg.noise_levels)
    out = Path(cfg.out or "results/burgers_pipe")
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'noise_level':>7} {'err_u':>9} {'err_q':>9} {'misfit':>9}")
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
    ap.add_argument("--tend", type=float, default=100.0)
    ap.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.05])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="small sizes, seconds")
    args = ap.parse_args()

    if args.config:
        for cfg in load_yaml(args.config):
            run(cfg)
        return
    cfg = RunConfig(h=0.35, dt=0.02, tend=0.6, noise_levels=[0.0, 0.1], seed=0,
                    out="results/burgers_pipe_quick") if args.quick else \
        RunConfig(h=args.h, dt=args.dt, tend=args.tend,
                  noise_levels=args.noise_levels, seed=args.seed)
    run(cfg)


if __name__ == "__main__":
    main()
