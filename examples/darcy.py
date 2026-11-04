"""Time-dependent Darcy flow benchmark (paper Section 4, Figures 4 and 12).

Identification of the permeability q = nu in  u_t - div(nu grad u) = g:
f(q, u) = -div(q grad u) is linear in q (sigma = 0, L = 0), and the tracking
operator C = -Laplace makes the reference model plain diffusion.

The true permeability is a PDEBench 2D Darcy sample as PDEBench defines
it: a binary field (values 0.1 and 1.0), constant on each cell of a
128 x 128 grid, held exactly on that grid (``mras.domains.pixel_square``).
The estimator's mesh is unstructured and does not follow the cells
(``fitted=True`` puts it on the grid, refined or coarsened by a power of
two, where q lies in Q_h): q is estimated in P0 across elements its jumps
cut through, so the best fit is off along the jumps, and only there.
PDEBench's Darcy set is steady:
each sample is a permeability map and the one steady solution of
-div(nu grad u) = beta, with no time series (its "t-coordinate" is a
placeholder), so only the map is read here.  The data are generated in
lockstep, by the estimator's own discretisation at q, from the smooth
state ``U0``, driven by i.i.d.
N(0, 1) pixel fields -- the persistent excitation the paper uses for this
gradient coupling -- drawn at the keyframes t = k SOURCE_PERIOD from a
stream keyed by (seed, k) and interpolated linearly in time between them,
so the forcing is a function of time, the same for every step size (the
paper's fresh field per step is the case dt = SOURCE_PERIOD), and the
estimator and the data model read the same source.
"""

import numpy as np
from ngsolve import IfPos, grad, pi, sin, x as xc, y as yc

from mras.data.pdebench import load_nu_grid
from mras.domains import darcy, pixel_square
from mras.fields import pixel_field
from mras.noise import seeded_rng
from mras.scheme import MRASProblem

RADIUS = 0.42       # radius of the disk initial guess a_0
# Middle tag of the source stream (seed, SOURCE_STREAM, k): the datum noise
# streams are keyed (seed, noise level * 1e6, step), and no noise level is
# 700 %.
SOURCE_STREAM = 7_000_000
SOURCE_PERIOD = 0.02  # time between independent pixel fields of the source

U0 = sin(pi * xc) * sin(pi * yc)  # deterministic initial state


def _cells(h, pixels):
    """Cells per side of a fitted mesh: the pixel grid refined or coarsened
    by the power of two that brings its width nearest to ``h``."""
    return max(1, round(pixels * 2.0 ** round(np.log2(1 / (pixels * h)))))


class DarcyProblem(MRASProblem):
    """Identification of q = nu in  u_t - div(nu grad u) = g(x, t).

    Parameters
    ----------
    h, k_u, k_q : design mesh width and the polynomial orders of U_h and
        Q_h; with ``fitted``, the mesh is the pixel grid refined or
        coarsened by the power of two that brings its width nearest to h.
    sample, hdf5 : which PDEBench permeability field to use.
    seed : pixel source stream and datum noise (both seeded).
    """

    name = "darcy"
    sigma = 0.0
    L = 0.0
    # (A2) holds only formally: the pairing int q grad(z) grad(q) with q in
    # the P0 parameter space is 0 a.e., and its smoothed relaxation
    # -1/2 int q^2 Laplace z is sign-indefinite -- see mras/verify.py.

    def __init__(self, h=1 / 128, dt=1e-3, tend=5.0, k_u=3, k_q=0, sample=0,
                 hdf5=None, seed=0, fitted=False):
        nu = np.asarray(load_nu_grid(hdf5, sample=sample))
        self.grid = nu.shape
        domains = (pixel_square(_cells(h, nu.shape[0]), order_u=k_u,
                                order_q=k_q) if fitted
                   else darcy(maxh=h, order_u=k_u, order_q=k_q))
        super().__init__(domains, dt=dt, tend=tend, k_u=k_u, seed=seed)
        # Constant on each pixel, exact at every point of any mesh; painted on
        # the pixel grid itself, where it lies in P0.
        self.q_true = pixel_field(nu, linear=False)
        self.q_domains = pixel_square(nu.shape[0], order_u=1)
        self.u0 = U0

    def initial_guess(self):
        """Disk indicator a_0 for the parameter initialisation."""
        return IfPos(RADIUS ** 2 - ((xc - 0.5) ** 2 + (yc - 0.5) ** 2), 1, 0)

    def source(self, t):
        """The pixel source at time ``t``: the keyframes on either side,
        weighted linearly."""
        s = (t.Get() - self.tstart) / SOURCE_PERIOD
        k = round(s)
        if abs(s - k) < 1e-9:  # on a keyframe
            return self._keyframe(k)
        k = int(np.floor(s))
        theta = s - k
        return (1 - theta) * self._keyframe(k) + theta * self._keyframe(k + 1)

    def _keyframe(self, k):
        rng = seeded_rng(self.seed, SOURCE_STREAM, k)
        return pixel_field(rng.standard_normal(self.grid))

    def coupling(self, q, u, w):
        return q * grad(u) * grad(w)

    def coupling_adjoint(self, q, u, w):
        return grad(u) * grad(w)


# --- runner registration (mras.live reads these markers) ----------------------

PROBLEM = DarcyProblem

# P0, the class of the pixel field, on an unstructured mesh as fine as the
# pixels; the data on one twice as fine.
DEFAULTS = {"darcy": dict(h=1 / 128, dt=0.005, tend=100.0, k_u=2, k_q=0)}
CHECK = {"darcy": dict(h=0.15, dt=0.05, tend=0.5)}
QUICK = CHECK
