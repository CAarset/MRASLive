#!/usr/bin/env python
"""Two-dimensional (vector) Burgers flow up a channel with barriers.

The velocity u = (u_1, u_2) obeys

    u_t - div(q grad u) + S (u . grad) u = 0      in the channel,
    u = u_in(t) on the inlet,  u = 0 on the barriers (no-slip),
    q grad u . n + beta (u . n) n = 0 on the walls (slip walls),
    q grad u . n = 0 on the outlet (do-nothing outflow),

in a vertical channel (``mras.domains.burgers_channel``): the inlet at the
bottom, the outlet at the top, walls bulging inward at staggered heights
and three irregular barriers in the lanes between the bulges.  The unknown
viscosity q is higher in a star left of the lowest barrier, a heart right
of the middle one and a diamond left of the top one, in equal bounding
squares, the star and the diamond on one x and the
heart mirrored about the channel's middle (``mras.domains.
CHANNEL_INCLUSIONS``), exact on
a mesh of its own that follows them -- the one quantity with a mesh of its
own: the data are the estimator's own discrete model at q, on its mesh.
That mesh does not follow the shapes (``fitted=True`` puts the estimator on
a fitted mesh), so the best fit in Q_h = P0 is off along their outlines.  Nothing drives the flow
but its inflow: there is no source, and the data enter through
time-dependent Dirichlet values -- the state and the datum carry the same
inflow, so the MRAS error lives in the homogeneous space, as the theory
requires.

Burgers flow has no pressure to push it along: no-slip walls would drain
its momentum, sideways into the walls at a rate of about q (pi / w)^2 in a
lane of width w, and the flow would reach about S U_MEAN w^2 / (pi^2 q)
before dying out.  The walls are slip walls instead: no stress along them,
and a stiff resistance ``WALL_RESISTANCE`` = beta against flow through them
(a Robin condition), which lets through about q du_n/dn / beta.  Only the
barriers drain the flow.  The convection is moderate, S U_MEAN = 20: Burgers
flow steepens into layers about q / (S U) wide, here 0.025, half an element
of the estimator's mesh, and the flow spreads across the channel, about half
of it reaching the outlet; at S U = 50 the inflow runs on as a jet
(docs/DEVIATIONS.md).  The parameter law scales with the datum squared, so
the inflow's speed sets how fast the estimate moves as well.

f(q, u) = -div(q grad u) + S (u . grad) u, with the walls' resistance (a
q-free, linear boundary term), is linear in q (sigma = 0, L = 0); the
parameter law is q_t = grad z : grad (u - z), the Frobenius product of the
velocity gradients, and the convection is evaluated on the datum (Remark
2.1).  C = -Laplace, componentwise, with walls and outlet natural, so
Poincare holds through the Dirichlet part (inlet and barriers) and (A3) is
as for Darcy.
(A2) is the gradient-coupling situation of Remark 4.2: identifiability
rests on persistent excitation, here from the inflow alone.  It rises
smoothly from rest, tapers toward the walls and sways slowly from side to
side, one sine of period ``SWAY_PERIOD`` = 50 against a settling time under
10, so that the flow passes through near-steady states.  A steady
flow z excites h only where div(h grad z) is not small -- where it shears
-- and a held inflow leaves the estimate stalled where its one steady
flow does not; the sway moves the shear about the channel.

    .venv/bin/python -m mras.live burgers_2d --quick
"""

from math import pi, sin

from ngsolve import CoefficientFunction, InnerProduct, grad, x

from mras import domains
from mras.domains import burgers_channel
from mras.fields import subdomain_field
from mras.scheme import MRASProblem

U_MEAN = 20         # centreline inflow speed, once ramped up
STRENGTH = 1        # S, the strength of the convection
RAMP = 5.0          # the inflow rises smoothly from rest over this time, then holds
WALL_TAPER = 0.5    # the inflow at the walls, as a fraction of its centre
SWAY = 0.25         # the transverse inflow's peak, as a fraction of U_MEAN
SWAY_PERIOD = 50.0  # its period, slow beside the flow's settling (t < 10)
WALL_RESISTANCE = 100.0  # beta of the slip walls, against flow through them
NO_SLIP = "inlet|barrier"  # the Dirichlet part: the inflow, and the barriers
Q_BACKGROUND, Q_SHAPES = 0.5, 1.0   # the true viscosity: background, shapes
Q_START = 0.75      # the initial estimate qhat_0, between the two


def ramp(t):
    """The inflow's amplitude at time ``t`` (a number or a Parameter): a
    smoothstep from 0 at rest to 1 at ``RAMP``, then 1."""
    t = t.Get() if hasattr(t, "Get") else float(t)
    r = min(max(t / RAMP, 0.0), 1.0) if RAMP > 0 else 1.0
    return r * r * (3 - 2 * r)


def sway(t):
    """The transverse inflow at time ``t`` (a number or a Parameter), as a
    fraction of ``U_MEAN`` mid-channel: one slow sine of peak ``SWAY`` and
    period ``SWAY_PERIOD``, rising from zero."""
    t = t.Get() if hasattr(t, "Get") else float(t)
    return SWAY * sin(2 * pi * t / SWAY_PERIOD)


def inflow(t):
    """The inflow velocity at time ``t``: up the channel, tapering to
    ``WALL_TAPER`` of its centre at the (slip) walls, and swaying slowly
    from side to side (:func:`sway`), its transverse part zero at the walls,
    which no flow crosses; the whole ramped up from rest (:func:`ramp`)."""
    width = domains.CHANNEL_WIDTH  # read when called: the channel may change
    xi = 2 * x / width - 1                  # -1 and 1 on the walls
    profile = 1 - (1 - WALL_TAPER) * xi ** 2  # 1 mid-channel
    across = sway(t) * (1 - xi ** 2)          # 0 on the walls
    return ramp(t) * U_MEAN * CoefficientFunction((across, profile))


class Burgers2D(MRASProblem):
    """Identify the viscosity q of vector Burgers flow from velocity data."""

    name = "burgers_2d"
    sigma = 0.0
    L = 0.0

    def __init__(self, h=0.12, dt=0.05, tend=10.0, k_u=3, k_q=0, seed=0,
                 fitted=False):
        super().__init__(burgers_channel(maxh=h, order_u=k_u, order_q=k_q,
                                         dirichlet=NO_SLIP, fitted=fitted),
                         dt=dt, tend=tend, k_u=k_u, seed=seed)
        self.u0 = CoefficientFunction((0, 0))  # at rest; the inflow ramps up
        # The shapes on a mesh of their own, exact there.
        self.q_domains = burgers_channel(maxh=h / 2, order_u=k_u, fitted=True)
        self.q_true = subdomain_field(
            self.q_domains.mesh,
            {"star": Q_SHAPES, "heart": Q_SHAPES, "diamond": Q_SHAPES},
            default=Q_BACKGROUND)

    def initial_guess(self):
        return Q_START

    def dirichlet(self, t):
        # The inflow on the inlet; the barriers, the rest of the Dirichlet
        # part, no-slip.
        return {"inlet": inflow(t)}

    def model_boundary(self, u, w):
        # Slip walls: a stiff resistance to flow through them, none along.
        return {"wall": WALL_RESISTANCE * InnerProduct(u, self.normal)
                * InnerProduct(w, self.normal)}

    def model(self, u, w):
        return STRENGTH * InnerProduct(grad(u) * u, w)

    def coupling(self, q, u, w):
        return q * InnerProduct(grad(u), grad(w))

    def coupling_adjoint(self, q, u, w):
        return InnerProduct(grad(u), grad(w))


# --- runner registration (mras.live reads these markers) ----------------------

PROBLEM = Burgers2D

# Q_h is P0, the class of the shapes.
DEFAULTS = {"burgers_2d": dict(h=0.05, dt=0.05, tend=100.0)}
# The audit runs to t = 3, still under the inflow's ramp, where q has come
# from 0.44 to 0.25; the flow has not reached the outlet by then, so the
# excitation fails there.
CHECK = {"burgers_2d": dict(h=0.35, dt=0.05, tend=3.0)}
QUICK = CHECK
