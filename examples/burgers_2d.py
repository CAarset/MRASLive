#!/usr/bin/env python
"""Two-dimensional (vector) Burgers flow up a channel with barriers.

The velocity u = (u_1, u_2) obeys

    u_t - div(q grad u) + S (u . grad) u = 0      in the channel,
    u = u_in(t) on the inlet,
    q grad u . n + beta (u . n) n = 0 on the walls and barriers (slip),
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

Burgers flow has no pressure to push it along: no-slip surfaces would drain
its momentum, sideways into them at a rate of about q (pi / w)^2 in a lane
of width w, and the flow would reach about S U_MEAN w^2 / (pi^2 q) before
dying out.  Walls and barriers are slip surfaces instead: no stress along
them, and a stiff resistance ``WALL_RESISTANCE`` = beta against flow through
them (a Robin condition), which lets through about q du_n/dn / beta.  What
meets a barrier head-on still stops, and leaves a shadow that only viscosity
fills.  The parameter law scales with the datum squared, so the inflow's
speed sets how fast the estimate moves as well.

f(q, u) = -div(q grad u) + S (u . grad) u, with the walls' resistance (a
q-free, linear boundary term), is linear in q (sigma = 0, L = 0); the
parameter law is q_t = grad z : grad (u - z), the Frobenius product of the
velocity gradients, and the convection is evaluated on the datum (Remark
2.1).  C = -Laplace, componentwise, with slip surfaces and outlet natural,
plus their resistance B: on f alone B would act on the datum, and the stiff
beta would take its noise's normal trace for a load several times the
parameter's own term; in C as well, B acts on the state instead, and the
error equation is that of C + B, coercive with Poincare through the Dirichlet
part (the inlet), so (A3) is as for Darcy.
(A2) is the gradient-coupling situation of Remark 4.2: identifiability
rests on persistent excitation, here from the inflow alone.  It rises
smoothly from rest, tapers toward the walls and sways slowly from side to
side, one sine of period ``SWAY_PERIOD`` = 62.5 against a settling time under
10, so that the flow passes through near-steady states.  A steady
flow z excites h only where div(h grad z) is not small -- where it shears
-- and a held inflow leaves the estimate stalled where its one steady
flow does not; the sway moves the shear about the channel.

Noisy data want a long time filter here: most of the channel is near plug
flow, so nothing holds q where noise pulls it, and the noise is relative to
a datum whose gradient is small beside its mean.  The flow is slow enough to
allow one (``smooth_scale``), and the ramp is slow so that the filter's
commutator with the convection stays small while the flow is being set up
(docs/DEVIATIONS.md).

    .venv/bin/python -m mras.live burgers_2d --quick
"""

from math import pi, sin

from ngsolve import CoefficientFunction, InnerProduct, grad, x

from mras import domains
from mras.domains import burgers_channel
from mras.fields import subdomain_field
from mras.scheme import MRASProblem

U_MEAN = 40         # centreline inflow speed, once ramped up
STRENGTH = 1        # S, the strength of the convection
RAMP = 40.0         # the inflow rises smoothly from rest over this time, then holds
WALL_TAPER = 0.5    # the inflow at the walls, as a fraction of its centre
SWAY = 0.25         # the transverse inflow's peak, as a fraction of U_MEAN
SWAY_PERIOD = 62.5  # its period, slow beside the flow's settling (t < 10)
WALL_RESISTANCE = 100.0  # beta of the slip surfaces, against flow through them
SLIP = "wall|barrier"    # the slip surfaces
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
    smooth_scale = 100.0  # tau = 1.7 at 1 % noise, 5 at 5 %

    def __init__(self, h=0.12, dt=0.05, tend=10.0, k_u=3, k_q=0, seed=0,
                 fitted=False):
        super().__init__(burgers_channel(maxh=h, order_u=k_u, order_q=k_q,
                                         dirichlet="inlet", fitted=fitted),
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
        return {"inlet": inflow(t)}

    def model_boundary(self, u, w):
        # A stiff resistance to flow through them, none along.
        return {SLIP: WALL_RESISTANCE * InnerProduct(u, self.normal)
                * InnerProduct(w, self.normal)}

    def tracking_boundary(self, u, w):
        # The same term in C: on the state, not on the noisy datum, whose
        # normal trace the stiff beta would otherwise take for a load.
        return self.model_boundary(u, w)

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
# The audit stops under the inflow's ramp, where q has come from 0.44 to
# 0.36.
CHECK = {"burgers_2d": dict(h=0.35, dt=0.05, tend=8.0)}
QUICK = CHECK
