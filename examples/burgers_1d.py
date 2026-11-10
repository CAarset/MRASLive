#!/usr/bin/env python
"""Time-dependent Burgers flow in a perforated pipe — fully generic use case.

Bringing your own problem takes a mesh and the physics, declared once: the
model operator f(q, u) split into its q-free part (``model``) and its
q-dependent part (``coupling``, with the adjoint density
``coupling_adjoint``), plus the source.  No data files and no ground truth
functions are needed: the driver generates noisy observations by stepping
the same physics, at q_true, in the estimator's own discretisation.

Model (scalar Burgers with unknown diffusion field q, advection in x):

    u_t - div(q grad u) + u u_x = g        in the pipe,
    u = 0                                 on walls, holes, inlet, outlet.

The pipe is a rectangle pierced by circular holes (CSG2d).  The true
parameter is a pair of low-diffusion wake bands straddling the hole row
(y = 0.4 and y = 0.8), higher diffusion elsewhere, lightly modulated
along x.  The state starts at rest and three beams, one turning about
each hole, keep it moving.

The estimator is the paper's scheme (4) with the model split
f(q, u) = -div(q grad u) + u u_x: linear in q, with sigma = 0 the
parameter equation is the Darcy adjoint rule (Prop. 4.1) verbatim,
q_t = grad z . grad (u - z) — the advection carries no q and drops out of
the q-derivative.  Per Remark 2.1 the remaining nonlinearity is evaluated
on the datum, z z_x, so it cancels from the continuous error system.
(A1) holds exactly (L = 0) and (A3) with C = -Laplace,
M = 1 as for Darcy; (A2) is left unverified exactly as the paper leaves
it for Darcy (Remark 4.2).  The persistent-excitation surrogate here is
the forcing itself: three pulsing beams turning about the holes with no
common period, so the source -- and the state it sustains -- never repeats
and excites every part of the pipe in turn (see docs/DEVIATIONS.md for the
full audit).

    .venv/bin/python -m mras.live burgers_1d --quick
    .venv/bin/python -m mras.live burgers_1d --config configs/burgers.yaml
"""

from ngsolve import IfPos, atan2, cos, exp, grad, pi, sin, x, y

from mras.domains import burgers_pipe, pipe_wake_field
from mras.scheme import MRASProblem

LENGTH, HEIGHT = 6.0, 1.2

HOLES = (1.5, 3.0, 4.5)  # hole centres along the centre line y = HEIGHT / 2
HOLE_R = 0.27

# Three beams of light, one emitted from each hole: a wedge of 10 - 20
# degrees, of constant strength along its direction out to the walls or the
# pipe ends, flat across it and falling off smoothly only near its edges.
# Light does not pass the holes: each beam leaves the shadows of the other
# two dark.  Each beam pulses (|sin|, never below ``PULSE_FLOOR`` of its peak:
# positive light; ``signed=True`` makes it heat and cool in turn, through
# zero), breathes in width, and turns slowly around its
# hole, alternate beams in opposite senses; beams cross freely.  Its pulse,
# its width and its turn run at periods of their own, drawn from
# P_n = 24 * 2**frac(n PHI) (25 - 39 time units, at least 1.4 apart,
# pairwise incommensurate; the turns at twice that), except the middle
# beam's turn, ``TURNS`` (two full turns by t = 100), all stretched by
# ``period_scale``, so the source never repeats and, in a movie at ~12 time
# units a second, the beams sweep smoothly.  What identifies q is the
# variety of spatial response patterns the forcing excites, everywhere:
# turning beams lay strong gradients over the whole pipe in turn, the ends
# included -- heaters circling the holes left the ends unexcited and q there
# unidentified (docs/DEVIATIONS.md, Burgers PE evidence).
PHI = (1 + 5**0.5) / 3
WIDTH, WIDTH_WOBBLE = 15.0, 5.0   # full beam width, degrees
EDGE = 4                          # flatness of the beam's angular profile
AMPLITUDE = 30.0                  # peak heating
TURNS = {1: 50.0}                 # turn periods set by hand: the middle beam's
PULSE_FLOOR = 0.25                # the light's minimum, as a fraction of its peak


def _period(n):
    """The n-th forcing period, 24 * 2**frac(n PHI): in [24, 48)."""
    return 24.0 * 2 ** ((n * PHI) % 1.0)


def _lit(k, dx, dy):
    """1 where the light from hole k reaches, 0 in the other holes' shadows.

    Seen from the centre of hole k, a point at offset (dx, dy) lies behind
    hole j -- its centre at (D, 0) from there, on the same centre line --
    when the ray to it passes within HOLE_R of that centre before reaching
    it: (D dy)^2 < (HOLE_R L)^2, D dx > 0 and D dx < L^2, with
    L^2 = dx^2 + dy^2.  Polynomial tests, no root.
    """
    reach = dx * dx + dy * dy
    lit = 1
    for j, other in enumerate(HOLES):
        if j != k:
            d = other - HOLES[k]
            behind = (IfPos((HOLE_R ** 2) * reach - (d * dy) ** 2, 1, 0)
                      * IfPos(d * dx, 1, 0) * IfPos(reach - d * dx, 1, 0))
            lit = lit * (1 - behind)
    return lit


def SOURCE(t, beams=(0, 1, 2), signed=False, period_scale=1.0):
    """The beams from the holes ``beams``, turning and pulsing."""
    def period(n):
        return period_scale * _period(n)

    g = 0
    for k in beams:
        dx, dy = x - HOLES[k], y - HEIGHT / 2
        turn = (period_scale * TURNS[k] if k in TURNS
                else 2 * period(3 * k + 3))
        heading = (-1) ** k * 2 * pi * t / turn + 2 * pi * k / 3
        along = dx * cos(heading) + dy * sin(heading)
        across = -dx * sin(heading) + dy * cos(heading)
        half = pi / 360 * (WIDTH + WIDTH_WOBBLE
                           * sin(2 * pi * t / period(3 * k + 2) + k))
        # angle from the beam's axis: flat inside, smooth fall-off at the edge
        profile = exp(-(atan2(across, along) / half) ** EDGE)
        pulse = sin(2 * pi * t / period(3 * k + 1) + 1.3 * k + 0.4)
        if not signed:
            pulse = PULSE_FLOOR + (1 - PULSE_FLOOR) * (pulse ** 2) ** (1 / 2)
        g = g + AMPLITUDE * pulse * profile * _lit(k, dx, dy)
    return g


class Burgers1D(MRASProblem):
    """Identify q in Burgers flow from generated noisy state observations."""

    name = "burgers_1d"
    # f(q, u) = -div(q grad u) + u u_x is affine in q: sigma = 0 and L = 0
    # exactly; (A2) shares the Darcy situation -- mras/verify.py.
    sigma = 0.0
    L = 0.0
    beams = (0, 1, 2)  # the holes that emit

    def __init__(self, h=0.15, dt=1e-2, tend=10.0, k_u=3, k_q=0, seed=0,
                 signed=False, period_scale=1.0):
        pipe = dict(hole_xs=HOLES, hole_r=HOLE_R)
        super().__init__(burgers_pipe(maxh=h, order_u=k_u, order_q=k_q, **pipe),
                         dt=dt, tend=tend, k_u=k_u, seed=seed)
        self.u0 = 0.0  # at rest; the aperiodic beams drive the state
        self.q_true = pipe_wake_field(LENGTH)
        self.signed = bool(signed)
        self.period_scale = float(period_scale)
        # light (|sin|) and u0 = 0 keep the state positive
        self.positive_source = self.positive_state = not self.signed

    def source(self, t):
        return SOURCE(t, self.beams, self.signed, self.period_scale)

    def model(self, u, w):
        # The advection carries no q.  On the datum it is a known load
        # (Remark 2.1); in the data model, the nonlinearity Newton resolves.
        return u * grad(u)[0] * w

    def coupling(self, q, u, w):
        return q * grad(u) * grad(w)

    def coupling_adjoint(self, q, u, w):
        return grad(u) * grad(w)


class Burgers1DDark(Burgers1D):
    """The same pipe with the left beam off: the left end is never lit, and
    q there is not identified -- excitation acts where the light falls."""

    name = "burgers_1d_dark"
    beams = (1, 2)


# --- runner registration (mras.live reads these markers) ----------------------

PROBLEMS = {"burgers_1d": Burgers1D,
            "burgers_1d_dark": Burgers1DDark}

# Q_h is discontinuous P1: the best fit of q_true in it is 0.071% off at
# h = 0.05 and 0.29% at h = 0.1 (P0: 1.2% and 2.4%), so a reconstruction
# within 1% is possible in principle.
DEFAULTS = {name: dict(h=0.05, dt=0.025, tend=100.0, k_q=1)
            for name in PROBLEMS}
# The audit runs to t = 1: the beams take q from 1 to 0.88 by then (0.94
# at t = 0.5, too close to the checklist's 10 % descent to be a verdict).
CHECK = {"burgers_1d": dict(h=0.35, dt=0.05, tend=1.0)}
QUICK = {name: dict(h=0.35, dt=0.05, tend=0.5) for name in PROBLEMS}
