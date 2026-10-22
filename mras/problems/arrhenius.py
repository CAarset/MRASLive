"""Arrhenius reaction-diffusion benchmarks: identification of the rate
field q(x) in

    Dt u - Laplace u + q(x) r(u) = g,      r(u) = exp(u)   (exponential)
                                         r(u) = 1 + u     (linearised)

Literature pedigree.  The exponential absorption q(x) exp(u) is the
Arrhenius reaction factor of Zeldovich / Frank-Kamenetskii combustion
theory; its steady states -Laplace u = g - q exp(u) are Gelfand's
Liouville problem, and the absorbing-sign flow is the globally
well-posed branch of the semilinear heat equation.  The linear member
q(x)(1 + u) is the exact first-order Taylor expansion of the exponential
factor at u = 0 and, up to the known reaction term q(x) * 1, the canonical
inverse-coefficient model for a zero-order reaction rate (recovery of
q(x) in Dt u - Laplace u + q(x) u = 0 from Cauchy data, Isakov, ch. 3),
and a standard benchmark for adaptive reaction-rate identifiers.

Theory (Assumption 2.2, paper eq. (4)).  f(q, z) = q r(z) is affine in q:
sigma = 0 exactly, and (A1) holds with L = 0 exactly, hence M~ = M.
(A2) is parameter-monotonicity of the absorption,

    <f(q,z) - f(q~), q - q~> = int r(z) (q - q~)^2 dx >= C_coe ||q - q~||^2,

with C_coe = inf r(z) on the datum: the exponential factor is strictly
positive with no positivity assumption on the data (the truth is in fact
nonnegative by the maximum principle, g >= 0 and u0 = 0, giving C_coe = 1
for the exponential member and C_coe = 1 + inf z > 0 for the linearised
one).  :meth:`c_coe` returns the constant witnessed on the generated
data.  (A3): C = -Laplace gives <Cv, v> = ||v||_V^2, N~ = 1, and with
L = 0 the bound M~ = M is satisfied.  This is the zero-order coupling
linear in q that the gradient-coupling examples (Darcy, Fisher, Burgers)
could not supply with a verified (A2).

Scale design.  The estimator flow for this zero-order coupling is the
composition delta q -> r(z) (minus Laplace)^-1 (r(z) delta q): each spatial
mode k decays at rate ~ r^2 / lambda_k, so the slowest mode is capped by
the domain's first Dirichlet eigenvalue lambda_1 = 2 (pi / L)^2.  On the
unit square lambda_1 ~ 20 forces assimilation horizons far beyond the
physical reaction time scale.  The box is therefore a large reaction
vessel, [0, 4 pi]^2, with lambda_1 = 1/8 (measured 0.12500 by the CG
inverse-iteration probe in mras.verify): the rate field is then seen at
the squared amplitude of the (pre-heated, reaction-dominated) state, and
the heater amplitudes are chosen reaction-dominated (three to four times
the background) so that the r^2 gain is large.  Measured on this design
(h = 0.3, dt = 0.02, noise level 0%): relative q-error 0.37 at t = 4, 0.13 at
t = 8, 0.081 at t = 28 with a tail decay of about 1 % per time unit - the
rate scales as the heater amplitude squared, as the flow-composition
argument predicts.  The asymptotic plateau is rate-limited by the
zero-order coupling, not by the representation (the P0 floor of q_true is
0.0085 on this grid) nor by dt (plateau invariant under dt refinement).
"""

from math import pi

import numpy as np
from ngsolve import (BilinearForm, GridFunction, LinearForm, cos, dx, exp,
                     grad, sqrt, x as xc, y as yc)

from ..domains import darcy
from ..fields import l2norm
from ..scheme import Problem

# Reaction-vessel size; see the scale-design note above (lambda_1 = 1/8).
SIZE = 4.0 * pi

# Moving multi-tone heaters on a uniform pre-heating background: the
# half-cosine envelopes and the background keep g >= 0, and the pre-heating
# holds the state away from u = 0, where P3 elements have no discrete
# maximum principle, and gives the datum the positive weak lower bound
# r(z) >= r(exp(0) = 1 on the exponential member's (A2) witness.
# Amplitudes are reaction-dominated (three times the background): the
# estimator flow gain scales with the squared reaction rate, so the
# heater strength is the recovery-rate knob (scale-design note above).
HEATERS = (  # centre_x, centre_y, amplitude, frequency, phase
    (4.0, 4.4, 3.6, 1.7, 0.4),
    (9.0, 8.4, 3.0, 2.3, 2.0),
    (6.4, 2.8, 2.4, 1.1, 3.7),
)
PREHEAT = 0.9  # uniform background heating, scaled by amplitude
HEATER_WIDTH = 10.4  # Gaussian variance of the heater spots

# True rate field: a low baseline with two smooth hot spots, placed and
# widened to match the 4 pi vessel.
Q_TRUE = (0.2 + 0.9 * exp(-((xc - 4.2) ** 2 + (yc - 4.6) ** 2) / 3.6)
          + 0.55 * exp(-((xc - 8.6) ** 2 + (yc - 8.0) ** 2) / 4.8))


class _ArrheniusBase(Problem):
    """Shared plumbing: heater source, backward-Euler truth on a finer
    mesh with Picard-lagged reaction, zero initial state and estimate."""

    counter_start = 1
    stop_margin = 2.0
    sigma = 0.0  # f is affine in q: no sigma Dt z residual in the flow
    L = 0.0      # (A1) exact: f'_q(q, z) = r(z) is independent of q
    box = ((0.0, 0.0), (SIZE, SIZE))  # domain bounding box, for figures

    # Reaction factor and its derivative, as CoefficientFunction maps.
    @staticmethod
    def r(w):
        raise NotImplementedError

    @staticmethod
    def rprime(w):
        raise NotImplementedError

    def __init__(self, h=0.3, dt=0.02, tend=10.0, k_u=3, seed=0,
                 data_h=0.15, data_order=2, amplitude=1.0):
        super().__init__(darcy(maxh=h, order_u=k_u, size=SIZE), dt=dt,
                         tend=tend, k_u=k_u, seed=seed)
        self.series_counters = frozenset(
            [1, 5, 10, 25, 50, 75, 100, 200, 500, 1000, self.tsteps - 2])
        self.data_domains = darcy(maxh=data_h, order_u=data_order,
                                  size=SIZE)
        self.amplitude = amplitude
        self.q_true = Q_TRUE
        self.u0 = 0.0

    def source(self, t):
        g = self.amplitude * PREHEAT
        for cx, cy, amp, omega, phase in HEATERS:
            bump = exp(-((xc - cx) ** 2 + (yc - cy) ** 2) / HEATER_WIDTH)
            # Power-modulated, never pulsed off: the envelope stays above
            # 15 % so the datum persistently excites the rate field.
            # Convergence of the estimator requires persistent excitation;
            # a fully dark phase stalls identification (oscillating err_q).
            envelope = 0.15 + 0.85 * (1 + cos(omega * t + phase)) / 2
            g = g + self.amplitude * amp * envelope * bump
        return g

    def data_range(self):
        """Range of the generated (noise-free) datum; the (A2) witness."""
        snaps = self._forward_data()
        return float(snaps.min()), float(snaps.max())

    def essinf_datum(self, rel_tol=1e-6, iters=25):
        """Weak essential infimum of the generated datum.

        The nodal minimum overstates how negative the datum is: a P3
        interpolant may dip below its weak lower bound between nodes, so
        only the integrated negative part (r0 - u)^+ legitimately probes
        ess inf - it vanishes exactly when u >= r0 a.e.  Bisection on the
        worst snapshot, monotone in r0.
        """
        snaps = self._forward_data()
        gf = GridFunction(self.Ufes)
        neg = np.zeros(len(snaps))
        scale = np.array([float(np.linalg.norm(s)) + 1.0 for s in snaps])

        def worst(r0):
            for k, s in enumerate(snaps):
                gf.vec.data = np.asarray(s, dtype=np.float64)
                part = (r0 - gf + sqrt((r0 - gf) ** 2)) / 2
                neg[k] = l2norm(part, self.mesh) / scale[k]
            return neg.max()

        hi = 0.0
        if worst(min(0.0, float(snaps.min()))) <= rel_tol:
            return float(snaps.min())  # nodal minimum is a.e. valid
        lo = float(snaps.min())
        if worst(lo) <= rel_tol:
            return lo
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if worst(mid) <= rel_tol:
                lo = mid
            else:
                hi = mid
        return lo

    def c_coe(self):
        """Witnessed (A2) constant, inf r(z) over the generated datum,
        with the ess inf taken in the weak (a.e.) sense."""
        return float(self.witness(self.essinf_datum()))

    @staticmethod
    def witness(lo):
        raise NotImplementedError

    C_scale = 1.0  # tracking operator C = -Laplace

    def coupling(self, q, z, s):
        """<f(q, z), s> = int q r(z) s dx: the (A2) pairing is int q^2 r(z)."""
        return q * self.r(z) * s

    def coupling_adjoint(self, q, z, s):
        return self.r(z) * s

    def forward_forms(self, step):
        """Implicit Euler with the Arrhenius factor Picard-lagged in u_prev:
        the Newton-linearised reaction r(u_n) + r'(u_n)(u - u_n)."""
        u, v, u_prev = step.u, step.v, step.u_prev
        a = BilinearForm(step.fes)
        res = LinearForm(step.fes)
        a += u * v * dx
        a += step.dt * grad(u) * grad(v) * dx
        a += step.dt * self.q_true * self.rprime(u_prev) * u * v * dx
        res += u_prev * v * dx
        res += step.dt * step.g * v * dx
        res += -step.dt * self.q_true \
            * (self.r(u_prev) - self.rprime(u_prev) * u_prev) * v * dx
        return a, res

    def forms(self, state):
        dt = self.dt
        z, z_old, gfu = state.z, state.z_old, state.gfu
        a = BilinearForm(self.X)
        res = LinearForm(self.X)
        # State row of (13): Dt u + f(q, z) + C(u - z) = g with
        # f(q, z) = -Laplace z + q r(z) and C = -Laplace.  The tracking
        # operator coincides with the datum-slot diffusion, so the two
        # cancel in the weak form and the estimator is a parabolic solve
        # with the lagged-coupling q r(z); sigma = 0, M~ = M.
        a += self.u * self.v * dx
        a += dt * grad(self.u) * grad(self.v) * dx
        a += dt * self.q * self.r(z) * self.v * dx
        res += gfu * self.v * dx
        res += dt * state.g * self.v * dx
        # Parameter row: scheme (4), Dt q = f'_q(., z)^* (u - z)
        #                     = r(z_old) (u^n - z_old),  lagged per (14).
        a += self.q * self.s * dx
        res += state.gfq * self.s * dx
        res += dt * self.r(z_old) * (gfu - z_old) * self.s * dx
        return a, res


class ArrheniusProblem(_ArrheniusBase):
    """Exponential Arrhenius absorption q(x) exp(u)."""

    name = "arrhenius"

    @staticmethod
    def r(w):
        return exp(w)

    @staticmethod
    def rprime(w):
        return exp(w)

    @staticmethod
    def witness(lo):
        return exp(lo)


class ArrheniusLinearProblem(_ArrheniusBase):
    """Linearised member q(x)(1 + u): first-order Taylor of exp at 0."""

    name = "arrhenius_linear"

    @staticmethod
    def r(w):
        return 1 + w

    @staticmethod
    def rprime(w):
        return 1.0

    @staticmethod
    def witness(lo):
        return 1 + lo
