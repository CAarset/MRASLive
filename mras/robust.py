"""Noisy data and the safeguards of the parameter law.

Noisy data enter the MRAS smoothed (Kaltenbacher-Nguyen 2021, (28); the
AMMC paper's (10), docs/THEORY_KN21.md): ``R_sp(R_ti(z^delta))``, a causal
filter in time -- an online method has no future data -- and a smoothing
step in space, whose results replace z in every term, the sigma-term's
D_t z included.  The theory then bounds the error by the smoothing errors,
up to a term that grows with time for noise that never stops: a run has to
stop, by the discrepancy principle, at the residual the data allow.

The residual is the data's in the estimator's discrete model,
``||D_t z + f(q, z) - g||_U*`` (the model fit's integrand), and its value at
the best fit in Q_h, P_h q^dagger, is the *effective* noise level: the
data's noise, their smoothing error and the part of q^dagger outside Q_h
at once.  Noise-free data still have one, zero only where q^dagger lies in
Q_h (``fitted``): the data come from the estimator's own discrete model, at
the true parameter.  Without a true parameter the declared
noise level stands in for it.

The parameter law has two modifications from robust adaptive control
(Ioannou & Sun 1996), each a flag on the problem:

* ``projection``  (on) q is projected onto ``q_bounds`` after every step.
                  Onto a convex set holding q^dagger the projection is
                  non-expansive, ||P q - q^dagger|| <= ||q - q^dagger||, so
                  the theory's energy decrease survives it (exactly on P0;
                  on higher orders it is the L2 projection of the clipped
                  field).
* ``leakage``     (off) -lambda (q - q_0) joins the parameter law, lambda =
                  ``LEAKAGE`` times the noise level.  It bounds noise-driven
                  drift but moves the fixed point by lambda (q_0 - q^dagger):
                  the MRAS theory does not support it.
"""

import numpy as np
from ngsolve import GridFunction, InnerProduct

from .solvers import factorize

# Time constant of the data's time filter: tau = FILTER_TIME delta^(2/3)
# dt^(1/3), which balances the filter's lag, O(tau D_t z), against the noise
# it leaves of a datum drawn afresh every step, O(delta (dt / tau)^(1/2)) --
# the order-optimal choice for data of a fixed time scale, its constant
# fitted on the Burgers pipe (2026-11-03: Matern noise at 1 % and 5 %, best
# at about 2 and 5 steps of 0.05).
FILTER_TIME = 5.0
WINDOW_STEPS = 50    # time constant of the residuals' running means, in steps
DISCREPANCY = 1.1    # tau: stop at tau times the effective noise level
# (implemented, off by default: in the 2026-11-03 smokes it stops too early,
# the windowed residual reaching its level while slow parameter directions
# still converge -- clean Burgers 1D at 21 % error, noisy at t = 5)
LEAKAGE = 1.0        # leakage rate per unit noise level, per unit time
ALPHA_STEPS = 40     # bisection steps of the smoother's alpha (log scale)


def filter_time(dt, noise_level):
    """The time filter's default time constant (``FILTER_TIME``), at least a
    step."""
    return max(dt, FILTER_TIME * noise_level ** (2 / 3) * dt ** (1 / 3))


def _h_norm(mass, vec):
    work = vec.CreateVector()
    work.data = mass * vec
    return max(float(InnerProduct(work, vec)), 0.0) ** 0.5


class Smoother:
    """``R_sp(R_ti(z))`` for one run's noisy data.

    In time, the exponential filter ``zbar <- zbar + (dt / tau)(z - zbar)``,
    tau = ``filter_time(dt, delta)`` unless given:
    white noise of level delta leaves ``delta (w / (2 - w))^(1/2)`` in it,
    w = dt / tau, and its time derivative is bounded by about delta / tau
    instead of delta / dt; the price is a lag of about tau D_t z.  In space,
    one implicit heat step, ``(M + alpha A) R = M zbar`` with the
    constrained dofs (the boundary data) kept.  alpha follows the
    discrepancy rule of the smoother: ``||R zbar - zbar||_H = delta_f
    ||zbar||_H``, delta_f the noise level of the filtered data -- chosen
    once, on the first datum that is not zero (data starting from rest have
    nothing to smooth at first), and kept: choosing again later would make
    the data jump.  Until then the data pass unsmoothed.
    """

    def __init__(self, problem, noise_level, tau=None, space=True):
        self.problem = problem
        self.delta = float(noise_level)
        self.tau = float(tau) if tau else filter_time(problem.dt, noise_level)
        self.weight = min(1.0, problem.dt / self.tau)
        cache = problem._prepare()
        self.mass, self.stiff = cache["Mu"], cache["Au"]
        self.free = problem.Ufes.FreeDofs()
        self.zbar = GridFunction(problem.Ufes)
        self.out = GridFunction(problem.Ufes)
        self.count = 0
        self.alpha = None if space else 0.0  # 0: the time filter alone
        self._inv = None

    def filtered_level(self):
        w = self.weight
        return self.delta * (w / (2 - w)) ** 0.5

    def __call__(self, z):
        zbar = self.zbar.vec.FV().NumPy()
        if self.count == 0:
            zbar[:] = z.vec.FV().NumPy()
        else:
            # NumPy, not an NGSolve expression: ``zbar.data += w (z -
            # zbar)`` reads zbar while it writes it, which NGSolve's lazy
            # evaluation does not order.
            zbar += self.weight * (z.vec.FV().NumPy() - zbar)
        self.count += 1
        if self.alpha == 0.0:
            self.out.vec.data = self.zbar.vec
            return self.out
        if self.alpha is None:
            if _h_norm(self.mass, self.zbar.vec) == 0.0:
                self.out.vec.data = self.zbar.vec
                return self.out
            self._choose(self.filtered_level())
        self.out.vec.data = self._apply(self._inv, self.alpha)
        return self.out

    def _operator(self, alpha):
        matrix = self.mass.CreateMatrix()
        matrix.AsVector().data = (self.mass.AsVector()
                                  + alpha * self.stiff.AsVector())
        return matrix, factorize(matrix, self.free, symmetric=True)

    def _apply(self, inv, alpha):
        """zbar + K^-1 (M zbar - K zbar) on the free dofs: the heat step,
        the constrained dofs of zbar kept."""
        out = self.zbar.vec.CreateVector()
        rhs = self.zbar.vec.CreateVector()
        rhs.data = (-alpha) * (self.stiff * self.zbar.vec)
        out.data = self.zbar.vec
        out.data += inv * rhs
        return out

    def _gap(self, alpha):
        _, inv = self._operator(alpha)
        diff = self._apply(inv, alpha)
        diff.data -= self.zbar.vec
        scale = max(_h_norm(self.mass, self.zbar.vec), 1e-300)
        return _h_norm(self.mass, diff) / scale, inv

    def _choose(self, target):
        """The alpha whose step changes zbar by ``target`` relative to it,
        by bisection on log alpha (the change grows with alpha)."""
        lo, hi = 0.0, 1e-6
        gap, inv = self._gap(hi)
        while gap < target and hi < 1e6:
            lo, hi = hi, hi * 10
            gap, inv = self._gap(hi)
        if gap >= target:
            for _ in range(ALPHA_STEPS):
                mid = (lo * hi) ** 0.5 if lo > 0 else hi / 10
                gap_mid, inv_mid = self._gap(mid)
                if gap_mid < target:
                    lo = mid
                else:
                    hi, inv = mid, inv_mid
                if lo > 0 and hi / lo < 1.05:
                    break
        self.alpha, self._inv = hi, inv

    # -- restart ----------------------------------------------------------
    def export(self):
        return ({"z_bar": self.zbar.vec.FV().NumPy().copy()},
                {"smooth_count": self.count,
                 "smooth_alpha": -1.0 if self.alpha is None else self.alpha})

    def restore(self, arrays, attrs):
        self.zbar.vec.FV().NumPy()[:] = arrays["z_bar"]
        self.count = int(attrs["smooth_count"])
        alpha = float(attrs["smooth_alpha"])
        if alpha == 0.0:
            self.alpha = 0.0
        elif alpha > 0:
            self.alpha = alpha
            _, self._inv = self._operator(alpha)


class Watch:
    """The data's residual in the discrete model at the estimate, ||D_t z +
    f(q, z) - g||_U*, against its effective noise level, the same at
    P_h q^dagger, both relative to the parameter's term in it, ||f(q, z) -
    f(0, z)||_U*, as running means of the squares over ``WINDOW_STEPS``
    steps (the data's noise makes them fluctuate from step to step, and
    data starting from rest have no size at first): the discrepancy stop
    reads them, once the means have run for two time constants.  Unlike the state's tracking residual u - z, which starts at
    zero and stays small whatever q is, this residual starts large at a
    poor q and falls toward the effective level, as Morozov's rule
    presumes."""

    def __init__(self):
        self.weight = 1.0 / WINDOW_STEPS
        self.res2 = self.eff2 = self.data2 = 0.0
        self.count = 0

    def add(self, residual2, effective2, data2):
        """The squared norms of one step: the residual at the estimate, at
        P_h q^dagger, and the parameter's term."""
        w = self.weight if self.count else 1.0
        self.res2 += w * (residual2 - self.res2)
        self.eff2 += w * (effective2 - self.eff2)
        self.data2 += w * (data2 - self.data2)
        self.count += 1

    def _relative(self, value2):
        return (value2 / self.data2) ** 0.5 if self.data2 > 0 else 0.0

    @property
    def residual(self):
        return self._relative(self.res2)

    @property
    def effective(self):
        return self._relative(self.eff2)

    @property
    def ready(self):
        return self.count >= 2 * WINDOW_STEPS

    def below(self, factor):
        return self.ready and self.residual <= factor * self.effective

    def export(self):
        return {}, {"watch_res2": self.res2, "watch_eff2": self.eff2,
                    "watch_data2": self.data2, "watch_count": self.count}

    def restore(self, attrs):
        self.res2 = float(attrs["watch_res2"])
        self.eff2 = float(attrs["watch_eff2"])
        self.data2 = float(attrs["watch_data2"])
        self.count = int(attrs["watch_count"])


def dual2(inv, functional):
    """||r||_U*^2 = r^T K^-1 r for a functional r on U_h, ``inv`` = K^-1
    of the U-norm (zero on the constrained dofs)."""
    work = functional.CreateVector()
    work.data = inv * functional
    return max(float(InnerProduct(work, functional)), 0.0)


def clip_values(values, bounds):
    lo, hi = bounds
    if lo is not None:
        values = np.maximum(values, lo)
    if hi is not None:
        values = np.minimum(values, hi)
    return values
