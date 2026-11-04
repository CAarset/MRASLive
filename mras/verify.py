"""Automated theory checklist: every example through the same questions.

Theory sources: the AMMC paper (Kaltenbach-Aarset-Nguyen 2026, ledger in
docs/ASSUMPTIONS.md) and the convergence theory it adapts, Kaltenbacher-
Nguyen 2021 (Inverse Problems 37, 055006; digest in docs/THEORY_KN21.md),
whose (A2)-(A4) are the (A1)-(A3) here and whose coercivity is a hypothesis
for sigma = 1 only (its Remark 1.3).

The paper's convergence theory (Assumption 2.2: (A1) Lipschitz remainder,
(A2) monotonicity with constant C_coe, (A3) tracking coercivity above
M~ = L^2 / (2 C_coe) + M) plus the adjoint pairing behind the parameter
row and the persistent excitation of the data make up a fixed list of
claims, and the convergence row says which statement they add up to for
the declared sigma.  ``run_checks`` asks each claim of a problem instance
and reports one of

    ok             confirmed numerically (margin reported)
    fail           contradicted numerically (the honest negative result)
    deviation      expected deviation, documented by ``adjoint_deviation``
    estimated      a numerical bound stands in for an unproven constant;
                   warned, never silently accepted
    unverifiable   no probe exists or the pairing degenerates discretely;
                   flagged, never silently skipped

Everything the ladder needs from the model is the single hook
``coupling(q, z, s)`` (the q-dependent part of f as a weak functional) on
:class:`mras.scheme.MRASProblem`: the same hook drives the affinity probe (L =
0 exactly for f affine in q), the Lipschitz estimator when f is nonlinear
in q, and the (A2) monotonicity probe.  A second, independently
transcribed hook ``coupling_adjoint(q, z, s)`` lets a pre-verification
compare the declared pairing density against the numerical q-derivative of
``coupling`` before any solving, and gives the sensitivity f_q(q~, z) of
the persistent-excitation check; missing hooks yield ``unverifiable``
with the reason.  ``scripts/checklist.py`` runs the whole registry and
writes the markdown table that backs the report's theory column.

What a program can settle decides the verdicts.  Discrete identities (the
adjoint pairing, the transcription of f_q, affinity in q) are confirmed
to round-off.  Constants of the discrete problem that are eigenvalues are
computed exactly on Q_h and U_h at the sampled data: the (A2) constant
when f is affine in q, the coercivity of C in the U-norm, the
persistent-excitation constant.  Infima and suprema over a nonlinear
dependence on q are only sampled, and a sample bounds them from one side
only - a negative (A2) pairing refutes, a positive minimum or a remainder
maximum estimates; confirming them takes an analytic bound the author
declares (``c_coe()``, ``L``), which the checklist evaluates on the data
and tests against the pairing on Q_h.

The U-norm is ||grad v|| where U_h has a Dirichlet part (a norm by
Poincare-Friedrichs, as in the paper) and the full H1 norm where it has
none; (A1) measures the remainder in its dual, (A3) the coercivity of C
in it, and the persistent excitation the sensitivity in its dual.

The (A2) probe evaluates the pairing ``int [f(q,z) - f(p,z)] (q - p) dx``
on the first, middle and last noise-free datum: exactly, as the smallest
eigenvalue of the pairing matrix on Q_h, where f is affine in q, and on
random patch pairs where it is not.  For gradient couplings on a P0
parameter space (Darcy, Fisher) this pairing is exactly zero - grad q = 0
a.e. - which is a degeneration of the *discretisation*, so the check
additionally evaluates the smoothed relaxation (the pairing tested against
the H1-conforming view of the difference in U_h); a sign-indefinite
relaxation reports ``fail`` with that note, matching the paper's own
refusal to claim (A2) there (Remark 4.2).  On a higher-order Q_h (Burgers,
P1) the elementwise pairing is evaluated as it stands, and is
sign-indefinite.

sigma is the author's choice, not a consequence of f.  The energy identity
of the error equations decides what each choice gives
(``check_convergence_route``): sigma = 1, the decay estimate from (A1)-(A3);
sigma = 0 with f affine in q, convergence through the persistent
excitation only, (A2) playing no part; sigma = 0 with f nonlinear in q,
nothing.

Persistent excitation asks that, on windows of time, the data see every
parameter direction: the window mean of ||f_q(q~, z) p||_U*^2 bounded
below by gamma ||p||_H^2.  On Q_h, gamma is the smallest eigenvalue of
the windowed Gramian; its eigenvector is the direction the data see
least, and where it lives says where the parameter is identified last.
"""

from collections import namedtuple
import warnings

import numpy as np
from ngsolve import (BilinearForm, CoefficientFunction, GridFunction,
                     InnerProduct, Integrate, LinearForm, MultiVector,
                     Preconditioner, dx, grad, x as xc, y as yc, z as zc)
from ngsolve.solvers import BVP

from .adjoint import adjoint_mismatch
from .solvers import factorize

OK, FAIL, UNVERIFIABLE, DEVIATION, ESTIMATED = (
    "ok", "fail", "unverifiable", "deviation", "estimated")

# ``value``: the constant a confirmed check establishes, for the checks
# that build on it ((A3) reads the (A2) constant here).
Check = namedtuple("Check", "name status detail value", defaults=(None,))

_TOL = 1e-12
PE_SAMPLES = 8        # datum instants over the horizon: two windows of four
PE_SAMPLES_AFTER = 50  # ... over a finished run's horizon (check_pe_after)
DENSE_MAX_DOFS = 5000  # dense eigenproblems on Q_h up to this size
PE_TOL = 1e-10        # smallest PE eigenvalue relative to the largest: kernel
WITNESS_TOL = 1e-2    # a declared C_coe may exceed the pairing on Q_h by this
STRONG_TOL = 5e-2     # model_strong against model on U_h: the face jumps of
                      # the datum's derivatives, not a transcription error


def _int(problem, cf):
    return float(Integrate(cf, problem.mesh))


def _datum_samples(problem, n=3):
    """Noise-free datum as U_h GridFunctions at spread-out times.

    Works for every problem: the datum mechanism ``problem.data`` is
    evaluated on a fresh noise-free state (no noise draw), which covers
    both generated snapshots and analytic references.
    """
    from .scheme import State
    # One state for all samples: the times ascend, so a generated datum is
    # stepped forward once instead of from t = 0 for every sample.
    state = State(problem, 0.0)
    problem.init_run(state)
    out = []
    for frac in np.linspace(0.05, 0.95, n):
        datum = max(1, int(round(frac * problem.tsteps)))  # read by step datum - 1
        if out and datum == state.step + 1:
            out.append(out[-1])
            continue
        state.step = datum - 1
        state.t.Set(problem.tstart + state.step * problem.dt)
        problem.data(state)
        z = GridFunction(problem.Ufes)
        z.vec.data = state.z.vec
        out.append(z)
    return out


def _datum_array(problem, n=25):
    rows = []
    for z in _datum_samples(problem, n):
        rows.append(np.array(z.vec.FV().NumPy()))
    return np.array(rows)


def _random_q(problem, rng):
    gf = GridFunction(problem.Qfes)
    gf.vec.FV().NumPy()[:] = rng.standard_normal(problem.Qfes.ndof)
    return gf


def _random_u(problem, rng):
    gf = GridFunction(problem.Ufes)
    gf.vec.FV().NumPy()[:] = rng.standard_normal(problem.Ufes.ndof)
    return gf


def _l2sq(problem, cf):
    return _int(problem, cf * cf)


def _dirichlet(problem):
    """Whether U_h has a Dirichlet part (constrained dofs)."""
    free = problem.Ufes.FreeDofs()
    return free.NumSet() < len(free)


def _u_gram(problem):
    """The U-norm on U_h, as its Gram form: ||grad v|| where U_h has a
    Dirichlet part (a norm by Poincare-Friedrichs, the paper's choice), the
    full H1 norm where it has none.  (A1), (A3) and the persistent
    excitation all measure in it."""
    U = problem.Ufes
    u, v = U.TnT()
    form = BilinearForm(U)
    form += InnerProduct(grad(u), grad(v)) * problem.dx
    if not _dirichlet(problem):
        form += InnerProduct(u, v) * problem.dx
    form.Assemble()
    return form


def _u_norm_name(problem):
    return "||v||_U = ||grad v||" if _dirichlet(problem) else \
        "||v||_U = H1 norm"


def _dense(mat, n):
    rows, cols, vals = mat.COO()
    out = np.zeros((n, n))
    np.add.at(out, (np.array(rows), np.array(cols)), np.array(vals))
    return out


def _q_mass_whitening(problem):
    """L^-1 for the Cholesky factor L of the Q_h mass matrix: eigenvalues
    relative to ||.||_H become plain symmetric eigenvalues."""
    Q = problem.Qfes
    p, s = Q.TnT()
    form = BilinearForm(Q)
    form += p * s * problem.dx
    form.Assemble()
    return np.linalg.inv(np.linalg.cholesky(_dense(form.mat, Q.ndof)))


def _linearisation_point(problem):
    """q~ for f_q(q~, z) in the probes: the truth where it is known, else
    the run's start; f affine in q does not see it."""
    for start in (problem.q_true, problem.q0):
        if start is not None:
            return problem.project_q(start)
    return problem.project_q(problem.initial_guess())


def check_adjoint(problem, state):
    """<A x, y> = <x, A y> for the linearised joint step (aligned datum)."""
    rep = adjoint_mismatch(problem, state)
    aligned = rep["aligned"]
    if aligned.get("vacuous"):
        return Check("adjoint pairing", UNVERIFIABLE,
                     "cross q<->u blocks inactive: "
                     + str(problem.adjoint_deviation or "by construction"))
    ratio = aligned["max_ratio"]
    if ratio < 1e-10:
        return Check("adjoint pairing", OK,
                     f"machine precision (max ratio {ratio:.1e}; "
                     f"lagged-datum ratio {rep['lagged']['max_ratio']:.1e})")
    if problem.adjoint_deviation:
        return Check("adjoint pairing", DEVIATION,
                     f"max ratio {ratio:.1e}: {problem.adjoint_deviation}")
    return Check("adjoint pairing", FAIL,
                 f"cross blocks disagree: max ratio {ratio:.1e}")


# -- (A1): affinity probe and Lipschitz estimator --------------------------

def _probe_affine(problem, samples, seed=0, n=8):
    """q -> <f(q, z), s> is affine exactly when the chord gap
    |F(lam q1 + (1-lam) q2) - lam F(q1) - (1-lam) F(q2)| vanishes; the
    ratio is normalised by the magnitudes of the three terms."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for z in samples:
        for _ in range(n):
            q1, q2 = _random_q(problem, rng), _random_q(problem, rng)
            s = _random_u(problem, rng)
            f1 = _int(problem, problem.coupling(q1, z, s) * dx)
            f2 = _int(problem, problem.coupling(q2, z, s) * dx)
            for lam in (0.3, 0.7):
                mix = lam * q1 + (1.0 - lam) * q2
                gap = _int(problem, problem.coupling(mix, z, s) * dx)
                scale = abs(gap) + abs(lam) * abs(f1) + abs(1 - lam) * abs(f2)
                worst = max(worst, abs(gap - lam * f1 - (1 - lam) * f2)
                            / max(scale, 1e-300))
    return worst < 1e-9, worst


def _estimate_lipschitz(problem, samples, seed=0, n=8, radii=(1.0, 3.0)):
    """Sampled bound on the (A1) remainder
    R = f(q+d, z) - f(q, z) - f_q(q, z) d, measured as max ||R||_U* / ||d||:
    f_q(q, .) d is the central difference of the hook, and ||R||_U* the
    Riesz norm of the residual functional in the U-norm of (A3).  A sampled
    maximum bounds L from below only."""
    U = problem.Ufes
    v = U.TestFunction()
    inv = factorize(_u_gram(problem).mat, U.FreeDofs(), symmetric=True)
    rng = np.random.default_rng(seed)
    eta = 1e-4
    lhat = 0.0
    for radius in radii:
        for z in samples:
            for _ in range(n):
                q = _scaled_q(problem, rng, radius)
                d = _scaled_q(problem, rng, radius / 2.0)
                lf = LinearForm(U)
                lf += (problem.coupling(q + d, z, v)
                       - problem.coupling(q, z, v)
                       - (problem.coupling(q + eta * d, z, v)
                          - problem.coupling(q - eta * d, z, v))
                       / (2 * eta)) * dx
                lf.Assemble()
                sol = GridFunction(U)
                sol.vec.data = inv * lf.vec
                rho = float(np.sqrt(max(
                    np.asarray(lf.vec) @ np.asarray(sol.vec), 0.0)))
                lhat = max(lhat, rho / np.sqrt(_l2sq(problem, d)))
    return lhat


def _scaled_q(problem, rng, radius):
    """Random field of Q_h with ||q||_L2 at most ``radius`` (scale-free probes
    need the bounded sets that the local Lipschitz statement runs on)."""
    gf = _random_q(problem, rng)
    norm = np.sqrt(_l2sq(problem, gf))
    if norm > 0:
        gf.vec.FV().NumPy()[:] *= radius * rng.uniform(0.2, 1.0) / norm
    return gf


def check_a1(problem, samples, affine=None):
    """(A1) Lipschitz remainder: zero exactly where f is affine in q,
    declared (``L``) or estimated where it is not.  Which sigma the
    problem may take is the convergence check's question
    (``check_convergence_route``)."""
    name = "(A1) Lipschitz remainder"
    if not callable(getattr(problem, "coupling", None)):
        return Check(name, UNVERIFIABLE, "coupling(q, z, s) not declared")
    if affine is None:
        affine, gap = _probe_affine(problem, samples)
    else:
        affine, gap = affine
    if affine:
        note = "" if problem.L in (None, 0.0) else \
            f" (declared L = {problem.L} unused: affinity gives L = 0)"
        return Check(name, OK,
                     f"f affine in q on sampled pairs (probe gap {gap:.1e}): "
                     f"remainder L = 0 exactly" + note, 0.0)
    if problem.L is not None:
        return Check(name, OK,
                     f"f nonlinear in q: constant L = {problem.L} declared "
                     "by the author (local Lipschitz on bounded sets)",
                     problem.L)
    lhat = _estimate_lipschitz(problem, samples)
    warnings.warn(
        f"{problem.name}: (A1) constant estimated numerically "
        f"(L_hat = {lhat:.3g}) -- declare L on the class to confirm it",
        stacklevel=3)
    return Check(name, ESTIMATED,
                 f"f nonlinear in q: sampled remainder bound L_hat = "
                 f"{lhat:.3g} on radius-3 balls (U* norm, "
                 f"{_u_norm_name(problem)}); a sampled maximum bounds L "
                 "from below only, declare L to confirm")


# -- (A2): monotonicity -----------------------------------------------------

def _a2_probe(problem, samples, seed=0, n=12, nrelax=6):
    """Localise the test pairs on a few cells: (A2) must hold for every
    q, i.e. pointwise in the space-time cylinder, and full-domain pairs
    only probe the datum-weighted mean of the monotonicity density, which
    averages out sign-changing data.  A pointwise coupling then pairs the
    difference qa - qb against itself in the raw Q_h test slot.  Gradient
    couplings are discretely degenerate there (grad q = 0 a.e. on P0, or
    the gradient operator does not even exist on the parameter space), so
    the probe additionally evaluates the smoothed relaxation: global
    directions tested against their U_h view.  On a higher-order Q_h the
    elementwise gradient exists and the raw pairing is evaluated."""
    rng = np.random.default_rng(seed)
    deg = []
    degenerate = False
    for z in samples:
        for _ in range(n):
            qa, qb = _patch_q(problem, rng), _patch_q(problem, rng)
            # The difference as a field of Q_h, not a sum of two: ngsolve
            # differentiates grid functions (elementwise on L2), never sums.
            diff = GridFunction(problem.Qfes)
            diff.vec.data = qa.vec - qb.vec
            nrm = _l2sq(problem, diff)
            try:
                deg.append(
                    (_int(problem, problem.coupling(qa, z, diff) * dx)
                     - _int(problem, problem.coupling(qb, z, diff) * dx))
                    / nrm)
            except Exception:
                # grad does not exist on the parameter space: that absence
                # is itself the discrete degeneracy of a gradient coupling.
                degenerate = True
    d = 0.0 if degenerate else min(deg)
    l = None
    if abs(d) <= 1e-10:
        rng = np.random.default_rng(seed)
        relaxed = []
        for z in samples:
            for _ in range(nrelax):
                qa, qb = _random_q(problem, rng), _random_q(problem, rng)
                nrm = _l2sq(problem, qa - qb)
                for view in _views(problem, qa - qb):
                    relaxed.append(
                        (_int(problem, problem.coupling(qa, z, view) * dx)
                         - _int(problem, problem.coupling(qb, z, view) * dx))
                        / nrm)
        l = min(relaxed)
    return d, l, degenerate


def _views(problem, field):
    """A Q_h direction viewed in U_h: the field itself for a scalar state,
    the field along each component for a vector one."""
    dim = GridFunction(problem.Ufes).dim
    views = []
    for j in range(dim):
        view = GridFunction(problem.Ufes)
        view.Set(field if dim == 1 else CoefficientFunction(
            tuple(field if i == j else 0 for i in range(dim))))
        views.append(view)
    return views


def _datum_cells_mean(problem):
    """L2 projections onto Q_h (cell means on P0) of sampled data: nodal
    values of a P3 datum ring below its a.e. bounds at extrema, so only
    integral quantities probe the space-time cylinder honestly."""
    Q = problem.Qfes
    u, v = Q.TnT()
    mm = BilinearForm(Q)
    mm += u * v * dx
    mm.Assemble()
    means = []
    for z in _datum_samples(problem):
        l = LinearForm(Q)
        l += z * v * dx
        l.Assemble()
        g = GridFunction(Q)
        BVP(bf=mm, lf=l, gf=g, pre=Preconditioner(mm, "local"), print=False)
        means.append(np.asarray(g.vec.FV().NumPy()).copy())
    return np.vstack(means)


def _patch_q(problem, rng, k=3):
    """Random field of Q_h with k nonzero coefficients, so supported on
    at most k cells (the whole space if Q is coarser), with random signs
    and amplitudes: the pairing ratio then localises the monotonicity
    density on the support."""
    q = GridFunction(problem.Qfes)
    fv = q.vec.FV().NumPy()
    fv[:] = 0.0
    cells = rng.choice(len(fv), size=min(k, len(fv)), replace=False)
    fv[cells] = rng.choice([-1.0, 1.0], size=len(cells)) \
        * rng.uniform(0.1, 3.0, size=len(cells))
    return q


def _exact_coe(problem, samples):
    """The (A2) constant on Q_h for f affine in q: the smallest eigenvalue,
    relative to ||.||_H, of the symmetrised pairing int coupling(p, z, s)
    over p, s in Q_h, minimised over the sampled data.  None where Q_h
    cannot test the coupling (no gradient on it, or it vanishes there:
    P0 under a gradient coupling) or is too large for a dense solve."""
    Q = problem.Qfes
    if Q.ndof > DENSE_MAX_DOFS:
        return None
    p, s = Q.TnT()
    whiten = _q_mass_whitening(problem)
    low = None
    for z in samples:
        form = BilinearForm(Q)
        try:
            form += problem.coupling(p, z, s) * problem.dx
            form.Assemble()
        except Exception:
            return None
        pairing = _dense(form.mat, Q.ndof)
        if np.abs(pairing).max() <= _TOL:
            return None
        pairing = whiten @ (0.5 * (pairing + pairing.T)) @ whiten.T
        lam = float(np.linalg.eigvalsh(pairing)[0])
        low = lam if low is None else min(low, lam)
    return low


def check_a2(problem, samples, affine=None):
    """(A2) monotonicity: int [f(q,z) - f(p,z)] (q - p) >= C_coe |q - p|^2.

    A declared bound ``c_coe()`` is the claim; the pairing on Q_h tests
    it, exactly (the smallest eigenvalue of the pairing matrix) where f is
    affine in q and by sampled patch pairs where it is not.  A negative
    pairing is a counterexample.  A positive sampled minimum bounds C_coe
    from above only, so undeclared it stays an estimate; the exact value
    confirms (A2) for Q_h at the sampled data.  The verdict is the
    property's; with sigma = 0 it is not a hypothesis of convergence (the
    sigma-term carries it, Kaltenbacher-Nguyen 2021, Remark 1.3), which the
    detail says."""
    check = _a2_verdict(problem, samples, affine)
    if not problem.sigma and check.status != UNVERIFIABLE:
        check = check._replace(detail=(
            check.detail + "; with sigma = 0 not a hypothesis of "
            "convergence (KN21, Remark 1.3)"))
    return check


def _a2_verdict(problem, samples, affine):
    name = "(A2) monotonicity"
    if not callable(getattr(problem, "coupling", None)):
        return Check(name, UNVERIFIABLE,
                     "coupling(q, z, s) not declared; parameter may be "
                     "absent from the state model entirely")
    exact = _exact_coe(problem, samples) if affine else None
    if exact is not None:
        probe, how = exact, "exact on Q_h"
    else:
        probe, l, degenerate = _a2_probe(problem, samples)
        how = "min over sampled patch pairs"
        if degenerate or abs(probe) <= 1e-10:
            if l > _TOL:
                return Check(name, UNVERIFIABLE,
                             "pairing vanishes on the discrete Q_h (grad q "
                             "= 0 a.e. on P0, or grad is undefined there); "
                             f"smoothed relaxation positive (min {l:.3g}) - "
                             "continuous claim not decidable")
            return Check(name, FAIL,
                         "pairing degenerates on Q_h and the smoothed "
                         f"relaxation is sign-indefinite (min {l:.3g}): "
                         "gradient coupling, as the paper concedes (Remark "
                         "4.2: not claimed for this example)")
    if probe <= _TOL:
        return Check(name, FAIL,
                     f"min pairing ratio {probe:.3g} ({how}): monotonicity "
                     "violated on Q_h")
    if callable(getattr(problem, "c_coe", None)):
        bound = float(problem.c_coe())
        if bound <= _TOL:
            return Check(name, FAIL,
                         f"declared bound c_coe() = {bound:.3g} <= 0")
        if bound > probe * (1 + WITNESS_TOL):
            return Check(name, FAIL,
                         f"declared bound c_coe() = {bound:.3g} exceeds the "
                         f"pairing ratio {probe:.3g} on Q_h ({how}): the "
                         "bound is wrong")
        return Check(name, OK,
                     f"C_coe >= {bound:.3g} (analytic bound c_coe(), "
                     f"evaluated on the data; pairing on Q_h, {how}: "
                     f"{probe:.3g})", bound)
    if exact is not None:
        return Check(name, OK,
                     f"C_coe,h = {probe:.3g} (exact on Q_h at the sampled "
                     "data; declare c_coe() for the continuous claim)",
                     probe)
    return Check(name, ESTIMATED,
                 f"no violation found: min ratio {probe:.3g} over sampled "
                 "patch pairs bounds C_coe from above only; declare c_coe() "
                 "to confirm")


# -- coupling pre-verification ----------------------------------------------

def check_pairing(problem, samples, n=100, seed=0):
    """The parameter row uses <f_q(q, z) p, s> = int p * coupling_adjoint(
    q, z, s) dx; pre-verify the declared density against the numerical
    q-derivative of coupling() on random draws, before any solving."""
    if not callable(getattr(problem, "coupling", None)):
        return Check("coupling pre-verification", UNVERIFIABLE,
                     "coupling(q, z, s) not declared")
    if not callable(getattr(problem, "coupling_adjoint", None)):
        return Check("coupling pre-verification", UNVERIFIABLE,
                     "coupling_adjoint(q, z, s) not declared: the pairing "
                     "cannot be pre-verified (recommended hook)")
    rng = np.random.default_rng(seed)
    # f is affine exactly when the forward quotient is the exact
    # difference; nonlinear remainders (Holder corners in |q|^{2/3} for
    # the potential flow) tolerate the O(eta^{2/3}) forward-difference
    # error but not a transcription mistake.
    affine, _ = _probe_affine(problem, samples, seed=seed, n=4)
    eta = 1.0 if affine else 1e-5
    tol = 1e-9 if affine else 1e-2
    lhs, rhs = [], []
    for i in range(n):
        z = samples[i % len(samples)]
        # continuous (not patch) draws: a nonlinear forward difference has
        # an O(eta) remainder that a near-zero exact pairing -- disjoint
        # patch supports -- cannot be compared against; a datum and a full
        # random q, p keep |rhs| away from zero so the relative test bites.
        q, p, s = (_random_q(problem, rng), _random_q(problem, rng),
                   _random_u(problem, rng))
        lhs.append((_int(problem, problem.coupling(q + eta * p, z, s) * dx)
                    - _int(problem, problem.coupling(q, z, s) * dx)) / eta)
        rhs.append(_int(problem, p * problem.coupling_adjoint(q, z, s) * dx))
    scale = float(np.median(np.abs(rhs)))
    ratio = max(abs(a - b) for a, b in zip(lhs, rhs)) / max(scale, 1e-300)
    if ratio > tol:
        return Check("coupling pre-verification", FAIL,
                     f"coupling_adjoint disagrees with the q-derivative of "
                     f"coupling: max ratio {ratio:.1e} over {n} draws")
    how = ("exact difference, f affine" if affine
           else f"forward difference eta = {eta:g} (Holder remainder)")
    detail = (f"pairing matches the numerical q-derivative of coupling on "
              f"{n} draws (max ratio {ratio:.1e}; {how})")
    if callable(getattr(problem, "model_strong", None)):
        gap = _strong_gap(problem, samples)
        if gap > STRONG_TOL:
            return Check("coupling pre-verification", FAIL,
                         f"model_strong disagrees with model on U_h: "
                         f"relative gap {gap:.1e} on the sampled data "
                         f"(tolerance {STRONG_TOL:g})")
        detail += (f"; model_strong matches model on U_h (relative gap "
                   f"{gap:.1e})")
    return Check("coupling pre-verification", OK, detail)


def _strong_gap(problem, samples):
    """The declared strong form of the model against its weak form, on the
    test functions of U_h that vanish on the boundary: there the two agree
    but for the jumps of the datum's derivatives across faces.  The largest
    relative gap over the sampled data."""
    U = problem.Ufes
    v = U.TestFunction()
    free = np.array(U.FreeDofs(), dtype=bool)
    worst = 0.0
    for z in samples:
        weak, strong = LinearForm(U), LinearForm(U)
        for term in problem.model_terms(z, v):
            weak += term
        strong += InnerProduct(problem.model_strong(z), v) * problem.dx
        weak.Assemble()
        strong.Assemble()
        a = weak.vec.FV().NumPy()[free]
        b = strong.vec.FV().NumPy()[free]
        worst = max(worst, float(np.linalg.norm(a - b))
                    / max(float(np.linalg.norm(b)), 1e-300))
    return worst


def _lambda1(problem, iters=200, seed=0):
    """Smallest eigenvalue of -Laplace on U_h relative to the L2 mass (the
    Poincare constant of ||grad .||), by inverse iteration."""
    U = problem.Ufes
    u, v = U.TrialFunction(), U.TestFunction()
    a = BilinearForm(U)
    a += InnerProduct(grad(u), grad(v)) * dx
    m = BilinearForm(U)
    m += InnerProduct(u, v) * dx
    a.Assemble()
    m.Assemble()
    inv = factorize(a.mat, U.FreeDofs(), symmetric=True)
    vec = a.mat.CreateColVector()
    work = vec.CreateVector()
    vec.FV().NumPy()[:] = np.random.default_rng(seed).standard_normal(U.ndof)
    lam = None
    for _ in range(iters):
        work.data = m.mat * vec
        vec.data = inv * work
        work.data = m.mat * vec
        vec.data = (1.0 / np.sqrt(InnerProduct(work, vec))) * vec
        work.data = a.mat * vec
        lam = InnerProduct(work, vec)
    return float(lam)


def _constant_mode(problem):
    """||grad v||^2 / ||v||^2 for the constant field v (each component 1)
    on U_h: zero, up to round-off, where U_h has no Dirichlet part."""
    v = GridFunction(problem.Ufes)
    v.Set(1.0 if v.dim == 1 else CoefficientFunction((1.0,) * v.dim))
    return (_int(problem, InnerProduct(grad(v), grad(v)))
            / _int(problem, InnerProduct(v, v)))


def check_a3(problem, a2, affine=None):
    """(A3): <C v, v> >= (L^2 / (2 C_coe) + M) ||v||_U^2 with M > 0.

    C = c (-Laplace) gives <C v, v> = c ||grad v||^2: coercive in the
    U-norm exactly where U_h has a Dirichlet part, with M~ = c in the norm
    ||grad .||.  With L = 0 (f affine in q) the condition is M~ = M > 0,
    whatever C_coe; otherwise it needs a declared L and a confirmed C_coe."""
    name = "(A3) coercivity"
    if not _dirichlet(problem):
        return Check(name, FAIL,
                     "C = c (-Laplace) vanishes on the constants (||grad "
                     f"v||^2/||v||^2 = {_constant_mode(problem):.1e} for v = "
                     "1): U_h has no Dirichlet part, so C is coercive in no "
                     "norm of U = H1, for any c (the paper's C adds the "
                     "boundary trace, -Laplace_Tr)")
    lam1 = _lambda1(problem)
    norm = f"in ||v||_U = ||grad v||, a norm by Poincare (lambda_1 = {lam1:.3g})"
    if problem.C_scale is None:
        return Check(name, UNVERIFIABLE,
                     f"C(||q||) = c(||q||) (-Laplace) is coercive {norm} for "
                     "every c > 0; the bound c(r) >= L(r)^2/(2 C_coe) + M "
                     "needs L as a function of ||q||, not declared")
    c = problem.C_scale
    if affine:
        return Check(name, OK,
                     f"L = 0 (f affine in q), so M~ = M: <Cv, v> = {c:.3g} "
                     f"||v||_U^2 {norm}, M = N~ = {c:.3g}", c)
    if problem.L is None:
        return Check(name, UNVERIFIABLE,
                     "no (A1) constant available: declare L on the class "
                     "(an estimated L_hat is too weak a basis for M~)")
    if a2.status != OK:
        return Check(name, UNVERIFIABLE,
                     f"(A2) not confirmed (status: {a2.status}), so "
                     "M~ = L^2/(2 C_coe) + M is undefined")
    need = problem.L ** 2 / (2 * a2.value)
    if c > need:
        return Check(name, OK,
                     f"<Cv, v> = {c:.3g} ||v||_U^2 {norm} > L^2/(2 C_coe) "
                     f"= {need:.3g}: M = {c - need:.3g}", c - need)
    return Check(name, FAIL,
                 f"<Cv, v> = {c:.3g} ||v||_U^2 {norm} <= L^2/(2 C_coe) = "
                 f"{need:.3g}")


# -- persistent excitation --------------------------------------------------

def _coupling_matrix(problem, z, q_lin):
    """B = f_q(q~, z): Q_h -> U_h*, assembled."""
    U, Q = problem.Ufes, problem.Qfes
    p, v = Q.TrialFunction(), U.TestFunction()
    b = BilinearForm(trialspace=Q, testspace=U)
    b += p * problem.coupling_adjoint(q_lin, z, v) * problem.dx
    b.Assemble()
    return b.mat


def _gram(problem, mat, inv):
    """B^T K^-1 B on Q_h, K the U-norm Gram matrix: p^T (B^T K^-1 B) p =
    ||B p||_U*^2."""
    Q = problem.Qfes
    cols = MultiVector(mat.CreateColVector(), Q.ndof)
    unit = mat.CreateRowVector()
    for j in range(Q.ndof):
        unit[:] = 0.0
        unit[j] = 1.0
        cols[j].data = mat * unit
    solved = MultiVector(mat.CreateColVector(), Q.ndof)
    solved[:] = inv * cols
    return np.array(InnerProduct(solved, cols))


def _pe_gram(problem, z, q_lin, inv):
    """||f_q(q~, z) p||_U*^2 as a matrix on Q_h."""
    return _gram(problem, _coupling_matrix(problem, z, q_lin), inv)


def persistent_excitation(problem, samples=None, n=PE_SAMPLES):
    """The discrete persistent-excitation constant of the data.

    PE asks for windows [t, t+T] on which the data see every parameter
    direction: the window mean of ||f_q(q~, z(s)) p||_U*^2 is at least
    gamma ||p||_H^2.  On Q_h this is the smallest eigenvalue of the window
    mean of B^T K^-1 B relative to the mass of Q_h; the datum is sampled at
    ``n`` instants of the horizon, two windows of half its length.
    Returns ``{"gamma", "top", "direction", "window", "length",
    "horizon"}``: the smallest eigenvalue over the windows, the largest in
    that window, its eigenvector (the direction the data see least, a
    GridFunction on Q_h), the window it occurs in, the window length and
    the end of the sampled horizon; None when Q_h is too large for the
    dense eigenproblem.
    """
    Q = problem.Qfes
    if Q.ndof > DENSE_MAX_DOFS:
        return None
    samples = samples or _datum_samples(problem, n)
    inv = factorize(_u_gram(problem).mat, problem.Ufes.FreeDofs(),
                    symmetric=True)
    q_lin = _linearisation_point(problem)
    half = len(samples) // 2
    sums = [0.0, 0.0]  # each window's Gramians, summed as they come
    for k, z in enumerate(samples):
        sums[k >= half] = sums[k >= half] + _pe_gram(problem, z, q_lin, inv)
    whiten = _q_mass_whitening(problem)
    best = None
    for window, count in enumerate((half, len(samples) - half)):
        gram = whiten @ (sums[window] / count) @ whiten.T
        lam, vec = np.linalg.eigh(0.5 * (gram + gram.T))
        if best is None or lam[0] < best["gamma"]:
            best = {"gamma": float(lam[0]), "top": float(lam[-1]),
                    "coeffs": whiten.T @ vec[:, 0], "window": window}
    direction = GridFunction(Q)
    direction.vec.FV().NumPy()[:] = best.pop("coeffs")
    best["direction"] = direction
    best["length"] = 0.5 * (problem.tend - problem.tstart)
    best["horizon"] = problem.tend
    return best


def persistent_excitation_baumeister(problem, samples=None, n=PE_SAMPLES):
    """Baumeister et al.'s persistent excitation, as Kaltenbacher-Nguyen
    2021 recall it (Remark 1.6), in the norm of H on Q_h:

        mu = inf_h  sup over [t1, t2] in a window of
             ||int_{t1}^{t2} f_q(q~, z(s)) h ds||_U* / ||h||_H .

    The integrals are the sampled couplings summed times the sample
    spacing.  Each subinterval alone bounds the supremum from below, so the
    best of three per window -- the window and its two halves -- bounds mu
    from below; the smaller over the two windows is reported, as in
    ``persistent_excitation``.  By Cauchy-Schwarz (mu / l)^2 <= gamma for a
    window of length l: the condition is the stronger one, failing where
    the coupling cancels over time.  Returns ``{"mu", "top", "direction",
    "window", "length", "horizon"}`` (``top``: the largest singular value
    on the subinterval chosen); None beyond the dense eigenproblem."""
    Q = problem.Qfes
    if Q.ndof > DENSE_MAX_DOFS:
        return None
    samples = samples or _datum_samples(problem, n)
    inv = factorize(_u_gram(problem).mat, problem.Ufes.FreeDofs(),
                    symmetric=True)
    q_lin = _linearisation_point(problem)
    spacing = (problem.tend - problem.tstart) / len(samples)
    mats = [_coupling_matrix(problem, z, q_lin) for z in samples]
    half = len(mats) // 2
    whiten = _q_mass_whitening(problem)
    best = None
    for window, part in enumerate((mats[:half], mats[half:])):
        mid = len(part) // 2
        bound = None
        for piece in (part, part[:mid], part[mid:]):
            if not piece:
                continue
            total = piece[0].CreateMatrix()
            total.AsVector().data = spacing * piece[0].AsVector()
            for mat in piece[1:]:
                total.AsVector().data += spacing * mat.AsVector()
            gram = whiten @ _gram(problem, total, inv) @ whiten.T
            lam, vec = np.linalg.eigh(0.5 * (gram + gram.T))
            if bound is None or lam[0] > bound[0]:
                bound = (float(lam[0]), float(lam[-1]), vec[:, 0])
        if best is None or bound[0] < best["mu2"]:
            best = {"mu2": bound[0], "top": bound[1] ** 0.5 if bound[1] > 0
                    else 0.0, "coeffs": whiten.T @ bound[2],
                    "window": window}
    direction = GridFunction(Q)
    direction.vec.FV().NumPy()[:] = best.pop("coeffs")
    best["mu"] = max(best.pop("mu2"), 0.0) ** 0.5
    best["direction"] = direction
    best["length"] = half * spacing
    best["horizon"] = problem.tend
    return best


def _whereabouts(problem, field, share=0.9):
    """Where ``field`` lives: the fraction of the domain carrying ``share``
    of its squared mass (densest cells first), and its centre."""
    mesh = problem.mesh
    mass = np.array(Integrate(field * field, mesh, element_wise=True))
    area = np.array(Integrate(1.0, mesh, element_wise=True))
    order = np.argsort(mass / area)[::-1]
    reach = np.searchsorted(np.cumsum(mass[order]) / mass.sum(), share) + 1
    fraction = area[order[:reach]].sum() / area.sum()
    centre = ", ".join(
        f"{float(Integrate(c * field * field, mesh)) / mass.sum():.2f}"
        for c in (xc, yc, zc)[:mesh.dim])
    return (f"{share:.0%} of it on {100 * fraction:.2g}% of the domain, "
            f"centred at ({centre})")


def check_pe(problem, samples=None, n=PE_SAMPLES):
    """Persistent excitation: the smallest eigenvalue of the windowed PE
    Gramian on Q_h (``persistent_excitation``) -- the classical Gramian
    form; Kaltenbacher-Nguyen 2021 (Remark 1.6) recall Baumeister et al.'s
    form instead, an integral of f(h, u) over a subinterval against
    ||h||_Q ||v||_U (docs/THEORY_KN21.md).  Zero up to round-off is a
    direction of Q_h the data never see; otherwise the eigenvalue and its
    eigenvector say how weakly, and where, the data see the parameter.
    For sigma = 0 and C = c (-Laplace) = c K, eliminating the fast state
    error leaves D_t eps_q = -(B^T C^-1 B) eps_q, so the slowest direction
    e-folds in about c / gamma.  The persistent-excitation verdict proper is
    Baumeister et al.'s (``check_pe_baumeister``); this row gives the rate."""
    name = "persistent excitation (Gramian)"
    pe = persistent_excitation(problem, samples, n)
    if pe is None:
        return Check(name, UNVERIFIABLE,
                     f"Q_h has {problem.Qfes.ndof} dofs, beyond the dense PE "
                     f"eigenproblem ({DENSE_MAX_DOFS}): check on a coarser "
                     "mesh")
    gamma, top = pe["gamma"], pe["top"]
    where = _whereabouts(problem, pe["direction"])
    if top <= 0 or gamma <= PE_TOL * top:
        return Check(name, FAIL,
                     f"a direction of Q_h the data up to t = "
                     f"{pe['horizon']:.3g} do not see: smallest eigenvalue "
                     f"{gamma:.2e} of largest {top:.2e}, zero to the "
                     f"accuracy of the solve; {where}")
    pace = ""
    if not problem.sigma and problem.C_scale is not None \
            and _dirichlet(problem):
        pace = (f"; slowest direction e-folds in about "
                f"{problem.C_scale / gamma:.3g} time units")
    return Check(name, OK,
                 f"gamma_h = {gamma:.3g} (largest {top:.3g}) on windows of "
                 f"length {pe['length']:.3g} up to t = {pe['horizon']:.3g}, "
                 f"in U* "
                 f"({_u_norm_name(problem)}); weakest direction: "
                 f"{where}{pace}", gamma)


def check_reproducible(build, cfg, noise_level=0.05):
    """Same seed must reproduce the same error histories, to round-off:
    threaded assembly sums in an order that varies from run to run (and an
    ill-conditioned step, the potential's C ~ 1e6, amplifies that to 1e-10),
    while a different noise draw moves the histories by the noise level."""
    r1 = build(**cfg).run_level(noise_level, progress=False).error_u
    r2 = build(**cfg).run_level(noise_level, progress=False).error_u
    same = r1 is not None and np.allclose(r1, r2, rtol=1e-8, atol=0.0)
    return Check("seeded reproducibility", OK if same else FAIL,
                 "identical error histories across two runs (to round-off)"
                 if same else "noise draws differ between same-seed runs")


def check_pe_baumeister(problem, samples=None, n=PE_SAMPLES):
    """Persistent excitation as the theory states it -- Baumeister et al.'s,
    which Kaltenbacher-Nguyen 2021 recall (Remark 1.6) --
    (``persistent_excitation_baumeister``): a lower bound on mu, zero to the
    accuracy of the solve failing.  It sees what the Gramian cannot: data
    whose coupling cancels over a window, such as a forcing oscillating in
    time.  The checklist's persistent-excitation verdict."""
    name = "persistent excitation"
    pe = persistent_excitation_baumeister(problem, samples, n)
    if pe is None:
        return Check(name, UNVERIFIABLE,
                     f"Q_h has {problem.Qfes.ndof} dofs, beyond the dense "
                     f"eigenproblem ({DENSE_MAX_DOFS}): check on a coarser "
                     "mesh")
    mu, top = pe["mu"], pe["top"]
    where = _whereabouts(problem, pe["direction"])
    if top <= 0 or mu ** 2 <= PE_TOL * top ** 2:
        return Check(name, FAIL,
                     f"a direction of Q_h whose excitation the data up to t "
                     f"= {pe['horizon']:.3g} cancel or never give: mu "
                     f"{mu:.2e} of largest {top:.2e}, zero to the accuracy "
                     f"of the solve; {where}")
    return Check(name, OK,
                 f"mu >= {mu:.3g} (largest {top:.3g}) on windows of length "
                 f"{pe['length']:.3g} up to t = {pe['horizon']:.3g}, in U* "
                 f"and H: (mu / l)^2 = {(mu / pe['length']) ** 2:.3g}; "
                 f"weakest direction: {where}", mu)


def check_pe_after(name, t_end, n=PE_SAMPLES_AFTER):
    """Persistent excitation (Baumeister's) over the horizon a run of example ``name``
    actually took, ``[0, t_end]``: known only once the run has stopped (at
    its target or its time limit).  The example is rebuilt at its audit
    discretization (CHECK, else QUICK), where the dense eigenproblem fits,
    and its noise-free data are sampled at ``n`` instants up to ``t_end``;
    None for an example with neither."""
    from mras.live import _locate, build
    module, _ = _locate(name)
    sizes = (getattr(module, "CHECK", {}).get(name)
             or getattr(module, "QUICK", {}).get(name))
    if sizes is None:
        return None
    return check_pe_baumeister(build(name, **{**sizes, "tend": t_end}), n=n)


def check_convergence_route(problem, linear, a1, a2, a3, pe, gram=None):
    """Which convergence statement the theory gives for the declared sigma,
    and whether its hypotheses are confirmed.

    Testing the error equations with (eps_q, eps_u) leaves the energy
    identity
        1/2 d/dt (|eps_q|^2 + |eps_u|^2) + sigma <f(q,z) - f(q^dagger,z),
            eps_q> + <R, eps_u> + <C eps_u, eps_u> = 0,
    R the (A1) remainder.  With sigma = 1, (A2) dissipates eps_q, absorbs R
    and gives the decay estimate; an indefinite pairing makes the energy
    grow for eps_q(0) = p with <f_q p, p> < 0, eps_u(0) = 0.  With sigma = 0
    and f affine in q, R = 0 and only the state error is dissipated: the
    parameter error decays through the state error it excites, at rates
    set by the persistent excitation, and (A2) plays no part.  With sigma
    = 0 and f nonlinear in q nothing absorbs R: no estimate."""
    sigma = problem.sigma
    name = f"convergence (sigma = {sigma:g})"
    if not sigma:
        if not linear:
            return Check(name, FAIL,
                         "f nonlinear in q and sigma = 0: nothing in the "
                         "energy identity absorbs the (A1) remainder, so it "
                         "gives no estimate; sigma = 1 dissipates the "
                         "parameter error through (A2)")
        detail = ("f affine in q and sigma = 0: (A2) does not enter the "
                  "energy identity; the parameter error decays through the "
                  "state error it excites, at rates set by the persistent "
                  f"excitation ({pe.status}")
        if (gram is not None and gram.status == OK
                and problem.C_scale is not None and _dirichlet(problem)):
            detail += (f", slowest direction e-folding in about "
                       f"{problem.C_scale / gram.value:.3g} time units")
        detail += ")"
        if a2.status == OK:
            detail += (f"; sigma = 1 would give the decay estimate's rate "
                       f"through (A2), C_coe >= {a2.value:.3g}, at the price "
                       "of D_t z and the model applied to the data in the "
                       "parameter law")
        status = pe.status if pe.status in (OK, FAIL) else UNVERIFIABLE
        return Check(name, status, detail)
    if a2.status == FAIL:
        return Check(name, FAIL,
                     "sigma = 1 with (A2) refuted: the sigma-term pairing is "
                     "indefinite, so the energy grows for eps_q(0) along a "
                     "direction where it is negative, eps_u(0) = 0; no "
                     "estimate")
    hypotheses = {"(A1)": a1, "(A2)": a2, "(A3)": a3}
    open_ = [key for key, check in hypotheses.items() if check.status != OK]
    if not open_:
        return Check(name, OK,
                     f"the decay estimate holds: L = {a1.value:.3g}, C_coe "
                     f">= {a2.value:.3g}, M = {a3.value:.3g}; rate C' = "
                     "min{C_coe, 2 M C_U->H}")
    failed = [key for key in open_ if hypotheses[key].status == FAIL]
    if failed:
        return Check(name, FAIL,
                     f"the decay estimate needs {', '.join(failed)}, "
                     "refuted")
    worst = UNVERIFIABLE if any(
        hypotheses[key].status == UNVERIFIABLE for key in open_) \
        else ESTIMATED
    return Check(name, worst,
                 "the decay estimate, if its hypotheses hold: "
                 + ", ".join(f"{key} {hypotheses[key].status}"
                             for key in open_))


def check_convergence(error_q):
    """The identification must actually descend."""
    if error_q is None or error_q[0] <= 0 or not np.all(np.isfinite(error_q)):
        return Check("reconstruction descends", UNVERIFIABLE,
                     "ground-truth parameter unknown or zero: no relative "
                     "error to track")
    dropped = error_q[-1] < 0.9 * error_q[0]
    return Check("reconstruction descends", OK if dropped else FAIL,
                 f"err_q {error_q[0]:.3g} -> {error_q[-1]:.3g} "
                 f"(final {100 * error_q[-1]:.1f}% relative)")


def run_checks(problem, state, error_q=None, seed=0):
    """Full checklist for one problem instance with data already loaded."""
    samples = _datum_samples(problem)
    affine = (_probe_affine(problem, samples, seed=seed)
              if callable(getattr(problem, "coupling", None)) else None)
    linear = bool(affine and affine[0])
    a1 = check_a1(problem, samples, affine)
    a2 = check_a2(problem, samples, linear)
    a3 = check_a3(problem, a2, linear)
    pe_samples = _datum_samples(problem, PE_SAMPLES)
    gram = check_pe(problem, pe_samples)
    pe = check_pe_baumeister(problem, pe_samples)
    return [
        check_adjoint(problem, state),
        check_pairing(problem, samples, seed=seed),
        a1,
        a2,
        a3,
        pe,
        gram,
        check_convergence_route(problem, linear, a1, a2, a3, pe, gram),
        check_convergence(error_q),
    ]


# -- registry -----------------------------------------------------------

def _registry():
    # built only on demand: the entries execute the example scripts and
    # pull in the whole problem library, which no other checklist call
    # needs.  The examples themselves declare the audit settings (the
    # CHECK marker in each examples/*.py); mras.live collects them.
    from mras.live import example_registry
    return example_registry()


def check_example(name, quick=True, seed=0):
    """Build, run one noise-free run (noise level 0%), and check the
    instance."""
    build, cfg = _registry()[name]
    problem = build(**cfg)
    problem.target = 0.0  # the audit runs to its horizon
    problem.discrepancy = None
    result = problem.run_level(0.0, progress=False)
    checks = run_checks(problem, result.state, result.error_q, seed=seed)
    if quick:
        checks.append(check_reproducible(build, cfg))
    return checks


STATUS_ORDER = {OK: 0, DEVIATION: 1, ESTIMATED: 2, UNVERIFIABLE: 3, FAIL: 4}


def as_markdown(name, checks):
    lines = [f"### {name}", "",
             "| check | status | detail |", "|---|---|---|"]
    for c in checks:
        mark = {OK: "OK", FAIL: "**FAIL**",
                UNVERIFIABLE: "*unverifiable*",
                DEVIATION: "*deviation*",
                ESTIMATED: "*estimated*"}[c.status]
        lines.append(f"| {c.name} | {mark} | {c.detail} |")
    return "\n".join(lines)
