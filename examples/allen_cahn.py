"""Modified Allen-Cahn benchmark (paper Section 5.3, Figures 9-11).

The reference state is a separable sine profile decaying linearly in time
toward a positive offset; the true parameter is piecewise constant on a
three-piece domain partition.  The measurement noise is the seeded Gaussian
FEM noise of :mod:`mras.noise`.
"""

from ngsolve import Integrate, grad, pi, sin, sqrt, x as xc, y as yc

from mras.domains import allen_cahn
from mras.fields import l2norm, subdomain_field

from mras.scheme import MRASProblem

DECAY_T = 10.0              # time scale of the linear decay (DECAY_T - t) / DECAY_T
OFFSET = 1.0                # positive offset required by the analysis
Z_MIN, Z_MAX = 0.2, 1.0     # state bounds entering the stabilisation bound C(.)
V_BG, V_INC1, V_INC2 = 1.0, 4.0, 2.0   # inclusion values of q^dagger


def piecewise_constant_field(mesh):
    """q^dagger: background ``V_BG`` with ``V_INC1`` / ``V_INC2`` inclusion
    pieces, exact on ``mesh``, a mesh that follows the interfaces
    (``allen_cahn(..., fitted=True)``)."""
    return subdomain_field(mesh, {"background": V_BG, "upper": V_INC1,
                                  "lower": V_INC2}, default=V_BG)


class AllenCahnProblem(MRASProblem):
    """Identify q in the modified Allen-Cahn model of Example 5.3.

    f(q, u) = -Laplace u + q u^3 + q |q|^{2/3} u (Prop. 7.1), nonlinear in
    q, so sigma = 1; the analytic reference state fixes the source,
    g = u^dagger_t + f(q^dagger, u^dagger).  The true parameter is
    piecewise constant on three pieces, exact on a mesh of its own that
    follows their interfaces; the estimator's mesh does not (unless
    ``fitted``), so the best fit in Q_h = P0 is off along the interfaces.
    """

    name = "allen_cahn"
    sigma = 1.0
    C_scale = None  # C(||q||) = c(||q||) (-Laplace), see tracking_scale
    manufactured_source = True
    decaying = True  # the reference relaxes linearly toward its offset
    positive_state = True  # the paper's positivity assumption (27)

    def __init__(self, h=0.1, dt=1e-3, tend=5.0, k_u=3, k_q=0, seed=0,
                 fitted=False):
        super().__init__(allen_cahn(maxh=h, order_u=k_u, order_q=k_q,
                                    fitted=fitted),
                         dt=dt, tend=tend, k_u=k_u, seed=seed)
        # q^dagger on a mesh of its own that follows its pieces, exact there
        self.q_domains = allen_cahn(maxh=h / 2, order_u=1, fitted=True)
        q_mesh = self.q_domains.mesh
        self.q_true = piecewise_constant_field(q_mesh)
        self.u_wot = sin(0.25 * pi * (xc - 2)) * sin(0.5 * pi * (yc - 1))
        # ||q^dagger||_{L2}^{2/3} + ||q^dagger||_{L3}^{2/3}, with q^dagger > 0
        self._c_base = (l2norm(self.q_true, q_mesh) ** (2 / 3)
                        + float(Integrate(self.q_true ** 3, q_mesh))
                        ** (2 / 9))

    def truth(self, t):
        """u^dagger = u_wot (DECAY_T - t) / DECAY_T + OFFSET."""
        return self.u_wot * (DECAY_T - t) / DECAY_T + OFFSET

    def truth_rate(self, t):
        return -self.u_wot / DECAY_T

    def c_coe(self):
        """(A2): q -> q |q|^{2/3} is monotone, so the pairing is >= int
        (u^dagger)^3 (q - p)^2, and C_coe = (ess inf u^dagger)^3.  u_wot, a
        product of two sines of one sign on the rectangle, lies in [0, 1],
        so u^dagger >= OFFSET until DECAY_T and OFFSET - (t - DECAY_T) /
        DECAY_T after."""
        return (OFFSET + min(0.0, (DECAY_T - self.tend) / DECAY_T)) ** 3

    def tracking_scale(self, q):
        """c(||q||) of the paper's stabilisation bound."""
        core = l2norm(q, self.mesh) ** (2 / 3) + self._c_base
        return 1 + 2 * (Z_MAX ** 2 / Z_MIN) * core ** 2

    def dirichlet(self, t):
        """The state is prescribed on the boundary (paper (38)): the
        reference state's values, so u - z vanishes there."""
        return {"wall": self.truth(t)}

    def model(self, u, w):
        return grad(u) * grad(w)

    def coupling(self, q, u, w):
        """q u^3 + q |q|^{2/3} u: monotone in q wherever u >= 0.  The datum
        u^dagger has a positive a.e. lower bound (its nodal P3 values ring
        below zero at sine extrema, but the cell means stay >= 1 - |u_wot|
        > 0), so (A2) holds on the actual space-time range."""
        return (q * u ** 3 + q * sqrt(q * q) ** (2 / 3) * u) * w

    def coupling_adjoint(self, q, u, w):
        """d/dq [q u^3 + q |q|^{2/3} u] = u^3 + (5/3) |q|^{2/3} u,
        transcribed independently."""
        return (u ** 3 + (5 / 3) * sqrt(q * q) ** (2 / 3) * u) * w


# --- runner registration (mras.live reads these markers) ----------------------

PROBLEM = AllenCahnProblem

# The time limit is DECAY_T, where the reference has decayed to its offset.
DEFAULTS = {"allen_cahn": dict(h=0.12, dt=0.01, tend=DECAY_T)}
CHECK = {"allen_cahn": dict(h=0.25, dt=0.05, tend=0.5)}
QUICK = CHECK
