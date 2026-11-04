"""Semilinear reaction-diffusion benchmark (Fisher-type; paper Figures 5-6).

Identification of q in  u_t - div(q grad u) + u - u^2 = g:  f(q, u) =
-div(q grad u) + u - u^2 is linear in q (sigma = 0, L = 0), and the
logistic reaction evaluated on the datum cancels from the error system
(Remark 2.1).  The true q is a ring, piecewise constant on three pieces,
defined exactly on a mesh of its own that follows the two circles, of
width ``q_h``.  The estimator's mesh does not (``fitted=True`` puts it on a
fitted mesh): q is estimated in P0 across elements its jumps cut through,
so the best fit is off along the circles, and only there.  The data are
generated in lockstep, by the estimator's own discretisation at q, from
rest.

The forcing is a train of waves running out from the centre to the walls
(not the paper's: its Gaussian pulse at the centre never reached the
corners, where q then stayed unidentified -- docs/DEVIATIONS.md).  Each
crest is a circle about the middle.
"""

from ngsolve import IfPos, grad, pi, sin, sqrt, x as xc, y as yc

from mras.domains import fisher
from mras.fields import subdomain_field
from mras.scheme import MRASProblem

R_IN, R_OUT = 0.3, 1.05     # ring radii of the true permeability
Q_IN, Q_OUT = 1.0, 0.25     # ring / exterior coefficient values
Q_MESH_ORDER = 4    # the true ring's mesh, curved to this order
# Outgoing waves A sin(2 pi r / WAVELENGTH - OMEGA t): crests run out at
# 0.25 per unit time, from centre to corner in about the default horizon.
# The amplitude keeps the datum below 1 (peak 0.88), beyond which the
# logistic term u^2 - u would blow the state up.
AMPLITUDE = 6.0
WAVELENGTH = 0.8
OMEGA = 2.0
BOX = 1.15                  # half-width of the box the initial estimate fills


class FisherProblem(MRASProblem):
    """Identification of q in u_t - div(q grad u) = g + u^2 - u."""

    name = "fisher"
    # (A2) shares the Darcy situation (P0 parameter space, sign-indefinite
    # smoothed pairing) -- mras/verify.py.
    sigma = 0.0
    L = 0.0

    def __init__(self, h=0.1, dt=1e-3, tend=10.0, k_u=3, k_q=0, seed=0,
                 q_h=0.09, fitted=False):
        rings = (R_IN, R_OUT)
        super().__init__(fisher(maxh=h, order_u=k_u, order_q=k_q, rings=rings,
                                fitted=fitted),
                         dt=dt, tend=tend, k_u=k_u, seed=seed)
        # The ring on a mesh of its own that follows it, exact there.
        self.q_domains = fisher(maxh=q_h, order_u=Q_MESH_ORDER, rings=rings,
                                fitted=True)
        self.q_true = subdomain_field(self.q_domains.mesh, {"ring": Q_IN},
                                      default=Q_OUT)
        self.u0 = 0.0

    def initial_guess(self):
        return (IfPos(xc + BOX, 1, 0) * IfPos(BOX - xc, 1, 0)
                * IfPos(yc + BOX, 1, 0) * IfPos(BOX - yc, 1, 0))

    def source(self, t):
        """Waves running out from the middle."""
        r = sqrt(xc * xc + yc * yc)
        return AMPLITUDE * sin(2 * pi * r / WAVELENGTH - OMEGA * t)

    def model(self, u, w):
        # Linear and quadratic parts apart: the linear one is assembled once.
        return [u * w, -u * u * w]

    def coupling(self, q, u, w):
        return q * grad(u) * grad(w)

    def coupling_adjoint(self, q, u, w):
        return grad(u) * grad(w)


# --- runner registration (mras.live reads these markers) ----------------------

PROBLEM = FisherProblem

# h = 0.04 keeps the band of elements the circles cut through thin against
# the ring (0.75 wide).
DEFAULTS = {"fisher": dict(h=0.04, dt=0.02, tend=100.0, q_h=0.03)}
CHECK = {"fisher": dict(h=0.2, dt=0.05, tend=0.5, q_h=0.15)}
QUICK = CHECK
