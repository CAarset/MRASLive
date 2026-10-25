#!/usr/bin/env python
"""Time-dependent Burgers flow in a perforated pipe — fully generic use case.

Bringing your own problem takes three things: a mesh, one forward (data)
weak form, and one MRAS estimator weak form.  No data files and no ground
truth functions are needed: the driver generates noisy observations by
stepping ``forward_forms`` on a finer mesh.

Model (scalar Burgers with unknown diffusion field q, advection in x):

    u_t - div(q grad u) + u u_x = g        in the pipe,
    u = 0                                 on walls, holes, inlet, outlet.

The pipe is a rectangle pierced by circular holes (CSG2d).  The true
parameter is a pair of low-diffusion wake bands straddling the hole row
(y = 0.4 and y = 0.8), higher diffusion elsewhere, lightly modulated
along x.  The state starts at rest and the heater grid keeps it moving.

The estimator is the paper's scheme (4) with the model split
f(q, u) = -div(q grad u) + u u_x: linear in q, so sigma = 0 and the
parameter equation is the Darcy adjoint rule (Prop. 4.1) verbatim,
q_t = grad z . grad (u - z) — the advection carries no q and drops out of
the q-derivative.  Per Remark 2.1 the remaining nonlinearity is evaluated
on the datum, z z_x, so it cancels from the continuous error system; the
state row below is Darcy's weak form (18)-(19) plus the datum-side term
-(z z_x, v).  (A1) holds exactly (L = 0) and (A3) with C = -Laplace,
M = 1 as for Darcy; (A2) is left unverified exactly as the paper leaves
it for Darcy (Remark 4.2).  The persistent-excitation surrogate here is
the forcing itself: heater frequencies are pairwise incommensurate, so
the source — and the state it sustains — never repeats (see
docs/DEVIATIONS.md for the full audit).

    .venv/bin/python -m mras.live burgers_pipe --quick
    .venv/bin/python -m mras.live burgers_pipe --config configs/burgers.yaml
"""

from ngsolve import BilinearForm, LinearForm, dx, exp, grad, pi, sin, x, y

from mras.domains import burgers_pipe
from mras.scheme import Problem

LENGTH, HEIGHT = 6.0, 1.2

# The golden ratio is the usual way to get pairwise irrational multiples:
# heater (i, j) runs at (3 + i + j/PHI) cycles per T_FORCING, and the
# ratio of any two of these six frequencies is irrational, so no common
# period exists.  The source is deterministic — every frequency and
# phase is fixed — but aperiodic; it is the discrete persistent-
# excitation surrogate for (A2).
PHI = (1 + 5**0.5) / 2
T_FORCING = 8.0          # reference period; heater periods are 1.4 - 2.7
SOURCE_AMPLITUDE = 5.0   # per-heater Gaussian peak; the resulting state
# amplitude is far smaller — the pipe is dissipative and absorbing on
# walls, holes, and both ends.


def SOURCE(t):
    """Six Gaussian heaters on a 3x2 cell grid, incommensurate and staggered."""
    return sum(
        SOURCE_AMPLITUDE
        * sin(2 * pi * ((3 + i + j / PHI) * t / T_FORCING
                        + (0.4 * i + 0.7 * j + 0.25) % 1.0))
        * exp(-((x - (i + 0.5) * LENGTH / 3) ** 2
                + (y - (j + 0.5) * HEIGHT / 2) ** 2) / 0.15)
        for i in range(3)
        for j in range(2)
    )


def _wake_field():
    """True diffusion: two smooth low-diffusion wake bands.

    The field is bounded in [Q_MIN, Q_MAX] and has no sharp boundaries;
    abrupt disks leave most of the L2 error in the kink layer rather
    than in the wake.
    """
    Q_MIN, Q_MAX = 0.12, 0.55

    wake = exp(-((y - 0.4) ** 2) / 0.03) + exp(-((y - 0.8) ** 2) / 0.03)
    wake *= 0.75 + 0.25 * sin(2 * pi * x / LENGTH)  # mild streamwise modulation

    return Q_MIN + (Q_MAX - Q_MIN) * (1.0 - wake / (1.0 + wake))


class BurgersPipe(Problem):
    """Identify q in Burgers flow from generated noisy state observations."""

    name = "burgers_pipe"
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

    def __init__(self, h=0.15, dt=1e-2, tend=10.0, k_u=3, seed=0):
        super().__init__(burgers_pipe(maxh=h, order_u=k_u), dt=dt, tend=tend,
                         k_u=k_u, seed=seed)
        self.data_domains = burgers_pipe(maxh=h / 2, order_u=k_u)
        self.u0 = 0.0  # at rest; the aperiodic heaters drive the state
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


# --- runner registration (mras.live reads these markers) ----------------------

PROBLEM = BurgersPipe

DEFAULTS = {"burgers_pipe": dict(h=0.35, dt=0.02, tend=100.0)}
CHECK = {"burgers_pipe": dict(h=0.35, dt=0.05, tend=0.5)}
QUICK = CHECK
