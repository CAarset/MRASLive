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

    ok             confirmed (margin reported)
    fail           contradicted numerically (the honest negative result)
    deviation      expected deviation, documented by ``adjoint_deviation``
    estimated      a numerical bound stands in for an unproven constant;
                   warned, never silently accepted
    unverifiable   no probe exists; flagged, never silently skipped

The checks run on noise-free data at three times (5, 50 and 95 % of the
horizon) and on the example's audit discretization (its CHECK marker),
where the dense eigenproblems fit.  What they need comes in three tiers.

Universal: every problem, from the hooks it declares anyway
-----------------------------------------------------------
``coupling(q, z, s)`` (the q-dependent part of f as a weak density), the
tracking operator the scheme assembles, and the data.

* (A1) Affinity in q by the chord test on ``coupling``: where the chord gap
  vanishes to round-off, the remainder vanishes and L = 0 exactly.  Else
  the remainder R = f(q^dagger + d, z) - f(q^dagger, z) - f_q(q^dagger, z) d
  (q~ = q^dagger, f_q d by central differences) is sampled over random d in
  H_h and measured in U*: its largest ratio ||R||_U* / ||d||_H bounds L
  from below, so it refutes a declared L or stands in as an estimate.
* (A2) The pairing <f(q, z) - f(p, z), E(q - p)> needs q - p as a test
  function of the state equation (the paper's Q subset of U).  The parameter
  spaces the estimator runs on (P0, discontinuous P1) are not in U: on P0 a
  gradient coupling's elementwise pairing is identically zero, and a scalar
  field cannot test a vector state at all, neither of which says anything
  about (A2).  The pairing is therefore evaluated on Q_c, continuous P1 on
  the estimator's mesh, zero on the boundaries where U_h is constrained, so
  that E Q_c lies in U_h, a subspace of U.  A nonpositive value is a
  counterexample in U; a positive minimum bounds C_coe from above only.
* (A3) The scheme's C = c (-Laplace + B): its boundary part B is tested for
  non-negativity on U_h (a negative value refutes the hook's contract), and
  the extreme eigenvalues of C relative to the U-norm on U_h are computed:
  the smallest bounds M from above, the largest bounds N~ from below.  rho_h,
  the smallest eigenvalue of the U-norm relative to H, is the constant of
  the decay rate (1 / C_{U->H}^2 <= rho_h).

Structural: routes a problem takes or not, detected from the problem
--------------------------------------------------------------------
* f affine in q (the chord test): L = 0; (A2) by the exact smallest
  eigenvalue of the pairing matrix on Q_c instead of sampled patch pairs;
  (A3) needs M > 0 only, whatever C_coe.
* A scalar parameter of a vector state (U_h of dimension m > 1): (A2)
  needs an embedding E of the parameter into the state space, which the
  paper's scalar examples never had to choose; the coordinate embeddings
  E_j p = p e_j are each tested, and the best is reported.  (A2) for a
  vector parameter is not probed.
* A Dirichlet part of U_h: ||v||_U = ||grad v|| is then a norm, and with
  B >= 0, <Cv, v> >= c ||grad v||^2 gives M = c analytically.  Without one,
  the U-norm is the H1 norm and M is the discrete eigenvalue: estimated.
* A constant tracking scale c (``C_scale``, ``tracking_scale`` not
  overridden).  A norm-dependent C(||q||) leaves (A3) unverifiable.
* Size: the dense eigenproblems on Q_c take up to DENSE_MAX_DOFS dofs.

Declared: analytic claims only the author can make
--------------------------------------------------
The checklist tests these and never invents them.

* ``L`` (a constant, or a function of ||q||_H): confirms (A1) for f
  nonlinear in q, unless a sampled remainder exceeds it.
* ``c_coe()``: confirms (A2), unless the pairing on Q_c falls below it.
* ``sigma``: the author's choice, not a consequence of f.  The energy
  identity of the error equations decides what each choice gives
  (``check_convergence_route``): sigma = 1, the decay estimate from
  (A1)-(A3); sigma = 0 with f affine in q, convergence through the
  persistent excitation only, (A2) playing no part; sigma = 0 with f
  nonlinear in q, nothing.

Nothing in this module is specific to an example: it reads the hooks and
the spaces, never a name.

The U-norm is ||grad v|| where U_h has a Dirichlet part (a norm by
Poincare-Friedrichs, as in the paper) and the full H1 norm where it has
none; (A1) measures the remainder in its dual, (A3) the coercivity of C
in it, and the persistent excitation the sensitivity in its dual.

The decay rate: testing the error equations with (eps_q, eps_u) gives
1/2 E' + (C_coe / 2) ||eps_q||_H^2 + M ||eps_u||_U^2 <= 0 (KN21 (22)), and
with ||v||_H <= C_{U->H} ||v||_U the rate min{C_coe, 2 M / C_{U->H}^2} =
min{C_coe, 2 M rho}.  KN21 (23), the AMMC paper and the proceedings print
2 M C_{U->H} with C_{U->H} the norm of the embedding; the convergence row
reports the rate the estimate gives (a departure, docs/DEVIATIONS.md).

Persistent excitation asks that, on windows of time, the data see every
parameter direction: the window mean of ||f_q(q~, z) p||_U*^2 bounded
below by gamma ||p||_H^2.  On Q_h, gamma is the smallest eigenvalue of
the windowed Gramian; its eigenvector is the direction the data see
least, and where it lives says where the parameter is identified last.
"""

from collections import namedtuple
import re
import warnings

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from threadpoolctl import threadpool_limits
from ngsolve import (H1, BilinearForm, CoefficientFunction, GridFunction,
                     InnerProduct, Integrate, LinearForm, MultiVector,
                     Preconditioner, dx, x as xc, y as yc, z as zc)
from ngsolve.solvers import BVP

from .adjoint import adjoint_mismatch
# The operators of the persistent excitation and the U-norm, shared with the
# pictures of where the data excite (mras.excitation).
from .excitation import (coupling_matrix as _coupling_matrix,  # noqa: F401
                         gramian_term as _gram,
                         has_dirichlet as _dirichlet,
                         linearisation_point as _linearisation_point,
                         mass_whitening as _q_mass_whitening,
                         u_gram as _u_gram)
from .solvers import factorize

OK, FAIL, UNVERIFIABLE, DEVIATION, ESTIMATED = (
    "ok", "fail", "unverifiable", "deviation", "estimated")

# ``value``: the constant a confirmed check establishes, for the checks
# that build on it ((A3) reads the (A2) constant here); ``extra``: further
# constants by name (the convergence row reads rho_h of (A3) here).
Check = namedtuple("Check", "name status detail value extra",
                   defaults=(None, None))

_TOL = 1e-12
PE_SAMPLES = 8        # datum instants over the horizon: two windows of four
PE_SAMPLES_AFTER = 50  # ... over a finished run's horizon (check_pe_after)
DENSE_MAX_DOFS = 5000  # dense eigenproblems on Q_h and Q_c up to this size
DENSE_U_DOFS = 1500   # U_h eigenproblems dense up to this size, else ARPACK
PE_TOL = 1e-10        # smallest PE eigenvalue relative to the largest: kernel
WITNESS_TOL = 1e-2    # a declared C_coe may exceed the pairing on Q_c by this
LIP_TOL = 1e-6        # a sampled (A1) remainder may exceed a declared L by
                      # this (the central difference in f_q d is O(eta^2))
EMBED_TOL = 1e-8      # E Q_c must lie in U_h: L2 error of the projection
B_TOL = 1e-10         # B >= 0 on U_h, relative to its largest eigenvalue
ARPACK_MAXITER = 300  # Lanczos iterations before the LOBPCG fallback
BLAS_THREADS = 4      # dense linear algebra inside the checks: many
                      # OpenBLAS threads on small matrices are ~40x slower
STRONG_TOL = 5e-2     # model_strong against model on U_h: the face jumps of
                      # the datum's derivatives, not a transcription error


# The checks' dense eigenproblems are small to moderate; NumPy, SciPy and
# NGSolve each bring an OpenBLAS with a thread per core, whose
# synchronisation dominates at these sizes.  The cap changes nothing but
# the order of floating-point sums.
_blas = threadpool_limits.wrap(limits=BLAS_THREADS)


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






def _u_norm_name(problem):
    return "||v||_U = ||grad v||" if _dirichlet(problem) else \
        "||v||_U = H1 norm"








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


def _reference_point(problem):
    """q^dagger of (A1) and (A2), with q~ = q^dagger: P_h q^dagger where the
    truth is known, else the run's start, which then stands in for it."""
    known = problem.q_true is not None
    return (_linearisation_point(problem),
            "q^dagger" if known else "q_0 (q^dagger unknown)")


def _remainder_samples(problem, samples, seed=0, n=8, radii=(1.0, 3.0)):
    """The (A1) remainder R = f(q + d, z) - f(q, z) - f_q(q, z) d at the
    problem's own q = q^dagger, with q~ = q^dagger, over random d in H_h of
    ||d||_H up to each radius: ``[(||R||_U* / ||d||_H, ||q + d||_H)]``.
    f_q(q, z) d is the central difference of the hook, ||R||_U* the Riesz
    norm of the residual functional in the U-norm of (A3).  The largest
    ratio bounds L from below (the theory's L(||q||) at the norm given)."""
    U = problem.Ufes
    v = U.TestFunction()
    inv = factorize(_u_gram(problem).mat, U.FreeDofs(), symmetric=True)
    base, _ = _reference_point(problem)
    rng = np.random.default_rng(seed)
    eta = 1e-4
    out = []
    for radius in radii:
        for z in samples:
            for _ in range(n):
                d = _scaled_q(problem, rng, radius)
                lf = LinearForm(U)
                lf += (problem.coupling(base + d, z, v)
                       - problem.coupling(base, z, v)
                       - (problem.coupling(base + eta * d, z, v)
                          - problem.coupling(base - eta * d, z, v))
                       / (2 * eta)) * dx
                lf.Assemble()
                sol = GridFunction(U)
                sol.vec.data = inv * lf.vec
                rho = float(np.sqrt(max(
                    np.asarray(lf.vec) @ np.asarray(sol.vec), 0.0)))
                out.append((rho / np.sqrt(_l2sq(problem, d)),
                            float(np.sqrt(_l2sq(problem, base + d)))))
    return out


def _scaled_q(problem, rng, radius):
    """Random field of Q_h with ||q||_L2 at most ``radius`` (scale-free probes
    need the bounded sets that the local Lipschitz statement runs on)."""
    gf = _random_q(problem, rng)
    norm = np.sqrt(_l2sq(problem, gf))
    if norm > 0:
        gf.vec.FV().NumPy()[:] *= radius * rng.uniform(0.2, 1.0) / norm
    return gf


def _declared_L(problem, r):
    """The declared (A1) constant at ||q||_H = r: ``L`` itself, or ``L(r)``
    for an L declared as a function of the norm."""
    return float(problem.L(r)) if callable(problem.L) else float(problem.L)


@_blas
def check_a1(problem, samples, affine=None):
    """(A1) Lipschitz remainder: zero exactly where f is affine in q (the
    chord test, universal).  Where it is not, the sampled remainder at
    q~ = q^dagger bounds L from below: it refutes a declared L that it
    exceeds, and otherwise stands in as an estimate.  Which sigma the
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
        note = ""
        if problem.L is not None and (callable(problem.L)
                                      or problem.L != 0.0):
            note = " (declared L unused: affinity gives L = 0)"
        return Check(name, OK,
                     f"f affine in q on sampled chords (gap {gap:.1e}): "
                     f"remainder L = 0 exactly" + note, 0.0)
    draws = _remainder_samples(problem, samples)
    lhat = max(ratio for ratio, _ in draws)
    _, at = _reference_point(problem)
    where = (f"q~ = {at}, ||q - q~||_H <= 3, U* norm "
             f"({_u_norm_name(problem)})")
    if problem.L is not None:
        excess = [(ratio / max(_declared_L(problem, norm), 1e-300), ratio,
                   norm) for ratio, norm in draws
                  if ratio > _declared_L(problem, norm) * (1 + LIP_TOL)]
        declared = ("L(||q||_H) declared as a function" if callable(problem.L)
                    else f"L = {problem.L:g} declared")
        if excess:
            _, ratio, norm = max(excess)
            return Check(name, FAIL,
                         f"f nonlinear in q: {declared}, refuted: a sampled "
                         f"remainder ratio {ratio:.3g} at ||q||_H = "
                         f"{norm:.3g} exceeds L = "
                         f"{_declared_L(problem, norm):.3g} ({where})")
        return Check(name, OK,
                     f"f nonlinear in q: {declared} by the author, not "
                     f"contradicted by the sampled remainder (max ratio "
                     f"{lhat:.3g}; {where})",
                     None if callable(problem.L) else float(problem.L))
    warnings.warn(
        f"{problem.name}: (A1) constant estimated numerically "
        f"(L_hat = {lhat:.3g}) -- declare L on the class to confirm it",
        stacklevel=3)
    return Check(name, ESTIMATED,
                 f"f nonlinear in q: sampled remainder bound L_hat = "
                 f"{lhat:.3g} ({where}); a sampled maximum bounds L "
                 "from below only, declare L to confirm", lhat)


# -- (A2): monotonicity on a subspace of U ----------------------------------

def _csr(mat, shape):
    """An assembled NGSolve matrix as a scipy CSR matrix."""
    rows, cols, vals = mat.COO()
    return sp.csr_matrix((np.array(vals), (np.array(rows), np.array(cols))),
                         shape=shape)


def _dirichlet_regions(problem):
    """The boundaries U_h constrains: those all of whose dofs are fixed."""
    U, mesh = problem.Ufes, problem.mesh
    free = np.array(U.FreeDofs(), dtype=bool)
    names = []
    for name in dict.fromkeys(mesh.GetBoundaries()):
        dofs = np.array(U.GetDofs(mesh.Boundaries(name)), dtype=bool)
        if dofs.any() and not free[dofs].any():
            names.append(name)
    return names


def _conforming_space(problem):
    """Q_c: continuous P1 on the estimator's mesh, zero on the boundaries
    U_h constrains -- the smallest space of parameters that are test
    functions of the state equation, on which the (A2) pairing is defined
    for every coupling.  Returns the space and the boundaries' names."""
    names = _dirichlet_regions(problem)
    if names:
        return H1(problem.mesh, order=1,
                  dirichlet="|".join(re.escape(n) for n in names)), names
    return H1(problem.mesh, order=1), names


def _embeddings(problem, qc, seed=0):
    """E_j: Q_c -> U_h, p -> p e_j (p itself for a scalar state), as dense
    matrices (U_h dofs x Q_c dofs), by the L2 projection onto U_h.  The
    projection is exact where U_h contains Q_c's directions, which is
    checked on a random field; None where it does not."""
    U = problem.Ufes
    dim = GridFunction(U).dim
    p, v = qc.TrialFunction(), U.TestFunction()
    mu = BilinearForm(U)
    mu += InnerProduct(U.TrialFunction(), v) * problem.dx
    mu.Assemble()
    inv = factorize(mu.mat, symmetric=True)
    probe = GridFunction(qc)
    probe.vec.FV().NumPy()[:] = np.random.default_rng(seed).standard_normal(
        qc.ndof)
    out = []
    for j in range(dim):
        def along(field):
            return field if dim == 1 else CoefficientFunction(
                tuple(field if i == j else 0 for i in range(dim)))
        g = BilinearForm(trialspace=qc, testspace=U)
        g += InnerProduct(along(p), v) * problem.dx
        g.Assemble()
        cols = MultiVector(g.mat.CreateColVector(), qc.ndof)
        unit = g.mat.CreateRowVector()
        for k in range(qc.ndof):
            unit[:] = 0.0
            unit[k] = 1.0
            cols[k].data = g.mat * unit
        solved = MultiVector(g.mat.CreateColVector(), qc.ndof)
        solved[:] = inv * cols
        emb = np.array([solved[k].FV().NumPy() for k in range(qc.ndof)]).T
        image = GridFunction(U)
        image.vec.FV().NumPy()[:] = emb @ probe.vec.FV().NumPy()
        gap = image - along(probe)
        if (np.sqrt(_int(problem, InnerProduct(gap, gap)))
                > EMBED_TOL * np.sqrt(_l2sq(problem, probe))):
            return None
        out.append(emb)
    return out


def _pairing_exact(problem, qc, emb, samples):
    """For f affine in q: per embedding E_j, the smallest eigenvalue of the
    symmetrized pairing <f_q(z) p, E_j s> on Q_c relative to its L2 mass,
    minimized over the sampled data, with its eigenvector (a GridFunction
    on Q_c).  None beyond DENSE_MAX_DOFS."""
    U = problem.Ufes
    free = np.array(qc.FreeDofs(), dtype=bool)
    if free.sum() > DENSE_MAX_DOFS:
        return None
    p, s = qc.TnT()
    v = U.TestFunction()
    m = BilinearForm(qc)
    m += p * s * problem.dx
    m.Assemble()
    mass = _csr(m.mat, (qc.ndof, qc.ndof)).toarray()[np.ix_(free, free)]
    best = [None] * len(emb)
    for z in samples:
        b = BilinearForm(trialspace=qc, testspace=U)
        b += problem.coupling(p, z, v) * problem.dx
        b.Assemble()
        coupling = _csr(b.mat, (U.ndof, qc.ndof))
        for j, e in enumerate(emb):
            pairing = (coupling.T @ e).T[np.ix_(free, free)]
            # the whole spectrum (see _eig_extreme on degenerate clusters)
            lam, vec = scipy.linalg.eigh(0.5 * (pairing + pairing.T), mass)
            if best[j] is None or lam[0] < best[j][0]:
                best[j] = (float(lam[0]), vec[:, 0])
    out = []
    for lam, vec in best:
        field = GridFunction(qc)
        field.vec.FV().NumPy()[free] = vec
        out.append((lam, field))
    return out


def _patch_coefficients(free, rng, k=3):
    """Coefficients of a random field of Q_c with k nonzero free dofs, so
    supported on at most k vertex patches, with random signs and
    amplitudes: the pairing ratio then localises the monotonicity density
    on the support, which full-domain pairs would average out."""
    out = np.zeros(len(free))
    idx = np.flatnonzero(free)
    cells = rng.choice(idx, size=min(k, len(idx)), replace=False)
    out[cells] = rng.choice([-1.0, 1.0], size=len(cells)) \
        * rng.uniform(0.1, 3.0, size=len(cells))
    return out


def _pairing_sampled(problem, qc, emb, samples, seed=0, n=12):
    """For f nonlinear in q: per embedding E_j, the smallest ratio
    <f(q_a, z) - f(q_b, z), E_j(q_a - q_b)> / ||q_a - q_b||_H^2 over random
    patch pairs of Q_c and the sampled data, with the difference of the
    minimizing pair (a GridFunction on Q_c)."""
    U = problem.Ufes
    free = np.array(qc.FreeDofs(), dtype=bool)
    rng = np.random.default_rng(seed)
    out = []
    for e in emb:
        low = None
        for z in samples:
            for _ in range(n):
                ca, cb = (_patch_coefficients(free, rng),
                          _patch_coefficients(free, rng))
                qa, qb, diff = (GridFunction(qc) for _ in range(3))
                qa.vec.FV().NumPy()[:] = ca
                qb.vec.FV().NumPy()[:] = cb
                diff.vec.FV().NumPy()[:] = ca - cb
                w = GridFunction(U)
                w.vec.FV().NumPy()[:] = e @ (ca - cb)
                ratio = ((_int(problem, problem.coupling(qa, z, w)
                               * problem.dx)
                          - _int(problem, problem.coupling(qb, z, w)
                                 * problem.dx))
                         / _l2sq(problem, diff))
                if low is None or ratio < low[0]:
                    low = (ratio, diff)
        out.append(low)
    return out


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


@_blas
def check_a2(problem, samples, affine=None):
    """(A2) monotonicity: <f(q,z) - f(p,z), E(q - p)> >= C_coe ||q - p||^2.

    Universal: the pairing on Q_c, a subspace of U (``_conforming_space``),
    exactly (the smallest eigenvalue of the pairing matrix) where f is
    affine in q and by sampled patch pairs where it is not.  A nonpositive
    value is a counterexample.  A positive one bounds C_coe from above
    only, so it stays an estimate; a declared bound ``c_coe()`` is the
    claim, which the pairing tests.  The verdict is the property's; with
    sigma = 0 it is not a hypothesis of convergence (the sigma-term
    carries it, Kaltenbacher-Nguyen 2021, Remark 1.3), which the detail
    says."""
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
    if GridFunction(problem.Qfes).dim != 1:
        return Check(name, UNVERIFIABLE,
                     "a vector parameter: its embedding into the state "
                     "space is not probed")
    qc, names = _conforming_space(problem)
    emb = _embeddings(problem, qc)
    if emb is None:
        return Check(name, UNVERIFIABLE,
                     "U_h does not contain the continuous P1 fields, so the "
                     "parameter differences cannot test the state equation")
    space = ("Q_c = continuous P1, zero on "
             + (", ".join(names) if names else "no boundary"))
    if len(emb) > 1:
        space += f"; E_j p = p e_j, j = 1..{len(emb)}"
    if affine:
        per = _pairing_exact(problem, qc, emb, samples)
        if per is None:
            return Check(name, UNVERIFIABLE,
                         f"Q_c has {int(np.array(qc.FreeDofs()).sum())} free "
                         f"dofs, beyond the dense eigenproblem "
                         f"({DENSE_MAX_DOFS}): check on a coarser mesh")
        how = "exact on Q_c"
    else:
        per = _pairing_sampled(problem, qc, emb, samples)
        how = "min over sampled patch pairs on Q_c"
    best = max(range(len(per)), key=lambda j: per[j][0])
    probe, field = per[best]
    each = ""
    if len(per) > 1:
        each = " (" + ", ".join(f"E_{j + 1}: {val:.3g}"
                                for j, (val, _) in enumerate(per)) + ")"
    if probe <= _TOL:
        every = " for every coordinate embedding" if len(per) > 1 else ""
        return Check(name, FAIL,
                     f"min pairing ratio {probe:.3g}{every}{each} ({how}; "
                     f"{space}): monotonicity violated, a counterexample in "
                     f"U; {_whereabouts(problem, field)}")
    chosen = f", along E_{best + 1}" if len(per) > 1 else ""
    if callable(getattr(problem, "c_coe", None)):
        bound = float(problem.c_coe())
        if bound <= _TOL:
            return Check(name, FAIL,
                         f"declared bound c_coe() = {bound:.3g} <= 0")
        if bound > probe * (1 + WITNESS_TOL):
            return Check(name, FAIL,
                         f"declared bound c_coe() = {bound:.3g} exceeds the "
                         f"pairing ratio {probe:.3g}{each} ({how}; {space}): "
                         "the bound is wrong")
        return Check(name, OK,
                     f"C_coe >= {bound:.3g} (analytic bound c_coe(), "
                     f"evaluated on the data; pairing {how}{chosen}: "
                     f"{probe:.3g}{each}; {space})", bound)
    return Check(name, ESTIMATED,
                 f"no violation found: C_coe <= {probe:.3g}{chosen}{each} "
                 f"({how}; {space}); a subspace of U bounds C_coe from above "
                 "only, declare c_coe() to confirm", probe)


# -- coupling pre-verification ----------------------------------------------

@_blas
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


def _eig_extreme(a, k, which):
    """The smallest (``"min"``) or largest (``"max"``) eigenvalue of the
    symmetric a relative to the SPD k, both scipy sparse.  Dense up to
    DENSE_U_DOFS.  Beyond, ARPACK: shift-inverted about -1 for the smallest
    (nearest -1 is the smallest of a spectrum above -1, and negative
    whenever any eigenvalue is), plain Lanczos for the largest, both capped
    at ARPACK_MAXITER; where ARPACK does not converge -- an eigenvalue of
    high multiplicity at the end sought -- LOBPCG preconditioned by k^-1."""
    n = a.shape[0]
    if n <= DENSE_U_DOFS:
        # the whole spectrum: the subset drivers' inverse iteration fails on
        # the degenerate clusters these pencils have (-Laplace relative to
        # its own Gram matrix is the identity)
        lam = scipy.linalg.eigh(a.toarray(), k.toarray(), eigvals_only=True)
        return float(lam[0] if which == "min" else lam[-1])
    lu = spla.splu(k.tocsc())
    minv = spla.LinearOperator(k.shape, matvec=lu.solve, dtype=float)
    try:
        if which == "min":
            return float(spla.eigsh(a.tocsc(), k=1, M=k.tocsc(), sigma=-1.0,
                                    which="LM", maxiter=ARPACK_MAXITER,
                                    return_eigenvectors=False)[0])
        return float(spla.eigsh(a, k=1, M=k, Minv=minv, which="LA",
                                maxiter=ARPACK_MAXITER,
                                return_eigenvectors=False)[0])
    except spla.ArpackNoConvergence:
        start = np.random.default_rng(0).standard_normal((n, 4))
        lam = spla.lobpcg(a, start, B=k, M=minv, largest=(which == "max"),
                          tol=1e-9, maxiter=1000)[0]
        return float(lam.min() if which == "min" else lam.max())


def _largest_singular(c, k):
    """sup <C v, w> / (||v||_K ||w||_K) for a nonsymmetric C: the square
    root of the largest eigenvalue of C^T K^-1 C relative to K."""
    lu = spla.splu(k.tocsc())
    op = spla.LinearOperator(c.shape, matvec=lambda x: c.T @ lu.solve(c @ x),
                             dtype=float)
    minv = spla.LinearOperator(k.shape, matvec=lu.solve, dtype=float)
    return float(np.sqrt(spla.eigsh(op, k=1, M=k, Minv=minv, which="LA",
                                    return_eigenvectors=False)[0]))


@_blas
def check_a3(problem, a2, affine=None):
    """(A3): <C v, v> >= (L^2 / (2 C_coe) + M) ||v||_U^2 with M > 0, and
    <C v, w> <= N~ ||v||_U ||w||_U.

    Universal: the scheme's C = c (-Laplace + B), as ``_prepare`` assembles
    it, on the free dofs of U_h.  B >= 0, the contract of
    ``tracking_boundary``, is tested: a negative eigenvalue of B relative to
    the U-norm is a counterexample in U_h.  The extreme eigenvalues of C
    relative to the U-norm on U_h bound M from above and N~ from below, and
    rho_h, the smallest eigenvalue of the U-norm relative to H, is reported
    for the decay rate (``check_convergence_route``).

    Structural: with a Dirichlet part of U_h and B >= 0, <C v, v> >= c
    ||grad v||^2 = c ||v||_U^2, so M = c is confirmed analytically; without
    one, M is the discrete eigenvalue, estimated.  With L = 0 (f affine in
    q) the condition is M > 0, whatever C_coe; otherwise it needs a
    declared constant L and a confirmed C_coe.  A tracking scale c that
    depends on q leaves the bound unverifiable."""
    name = "(A3) coercivity"
    U = problem.Ufes
    idx = np.flatnonzero(np.array(U.FreeDofs(), dtype=bool))
    cache = problem._prepare()

    def free(mat):
        return _csr(mat, (U.ndof, U.ndof))[idx][:, idx].tocsr()

    stiff, track, mass_u = free(cache["Au"]), free(cache["Cu"]), free(cache["Mu"])
    gram = free(_u_gram(problem).mat)
    sym = 0.5 * (track + track.T)
    u, v = U.TnT()
    has_b = bool(problem._on_boundary(problem.tracking_boundary(u, v),
                                      problem.order))
    dirichlet = _dirichlet(problem)
    # B's quadratic form sees only the dofs B touches, so its sign is that
    # of B on them, relative to the U-norm there: a small dense problem.
    boundary = (sym - stiff).tocsr()
    boundary.data[np.abs(boundary.data)
                  <= 1e-14 * max(np.abs(sym.data).max(), 1e-300)] = 0.0
    boundary.eliminate_zeros()
    touched = np.unique(boundary.nonzero()[0])
    b_low, b_top = 0.0, 0.0
    if has_b and len(touched):
        lam = scipy.linalg.eigh(boundary[touched][:, touched].toarray(),
                                gram[touched][:, touched].toarray(),
                                eigvals_only=True)
        b_low, b_top = float(lam[0]), float(lam[-1])
    negative = b_low < -B_TOL * max(b_top, 1.0)  # beyond round-off
    if dirichlet and len(touched) < len(idx) and not negative:
        # the U-norm is ||grad .|| = stiff, so C / c = stiff + B: its smallest
        # eigenvalue relative to it is 1 + min(B) = 1, B >= 0 vanishing on
        # the dofs it does not touch -- exact, no eigensolve
        low = 1.0
    else:
        low = _eig_extreme(sym, gram, "min")
    top = _eig_extreme(sym, gram, "max")
    skew = spla.norm(track - track.T) / max(spla.norm(track), 1e-300)
    n_tilde = (max(abs(low), top) if skew <= _TOL
               else _largest_singular(track, gram))
    rho = _eig_extreme(gram, mass_u, "min")
    extra = {"rho": rho}
    norm = ("||v||_U = ||grad v||, a norm by Poincare" if dirichlet
            else "||v||_U = H1 norm, U_h having no Dirichlet part")
    operator = "-Laplace + B" if has_b else "-Laplace"
    spectrum = (f"{operator} on U_h relative to the U-norm: eigenvalues in "
                f"[{low:.3g}, {top:.3g}]")
    if has_b and negative:
        return Check(name, FAIL,
                     f"the boundary part B of C is negative on U_h: <Bv, v> "
                     f"= {b_low:.3g} ||v||_U^2 for some v on the dofs B "
                     "touches, against the contract of tracking_boundary "
                     "(non-negative)", None, extra)
    if low <= _TOL:
        return Check(name, FAIL,
                     f"C is not coercive on U_h in {norm}: {spectrum} "
                     "(-Laplace vanishes on the constants unless a "
                     "Dirichlet part or B holds them)", None, extra)
    if not problem._constant_tracking() or problem.C_scale is None:
        return Check(name, UNVERIFIABLE,
                     f"C(||q||) = c(||q||) ({operator}), {spectrum}: "
                     "coercive for every c > 0, but the bound c(r) >= "
                     "L(r)^2/(2 C_coe) + M needs c and L at every r = "
                     "||q||_H (tracking_scale depends on q)", None, extra)
    c = float(problem.C_scale)
    if dirichlet:
        coerce = c
        basis = (f"<Cv, v> >= c ||grad v||^2 = {c:.3g} ||v||_U^2 "
                 f"analytically, as B >= 0" if has_b else
                 f"<Cv, v> = c ||grad v||^2 = {c:.3g} ||v||_U^2")
        if has_b:
            basis += f" (on U_h: min {b_low:.2g})"
    else:
        coerce = c * low
        basis = (f"<Cv, v> >= {coerce:.3g} ||v||_U^2 on U_h, an upper bound "
                 "for the continuous constant")
    after = (f"N~ >= {c * n_tilde:.3g} (largest on U_h); {spectrum}, "
             f"c = {c:.3g}; {norm}; rho_h = {rho:.3g} (1/C_U->H^2 <= "
             "rho_h)")
    if skew > _TOL:
        after += f"; C nonsymmetric (relative skew {skew:.1e})"
    if affine:
        need, why = 0.0, "L = 0 (f affine in q), so M~ = M"
    else:
        if problem.L is None:
            return Check(name, UNVERIFIABLE,
                         "no (A1) constant available: declare L on the class "
                         "(an estimated L_hat is too weak a basis for M~); "
                         + after, None, extra)
        if callable(problem.L):
            return Check(name, UNVERIFIABLE,
                         "L declared as a function of ||q||_H: with a "
                         "constant c, M~ needs sup L(r)^2/(2 C_coe), not "
                         "checked; " + after, None, extra)
        if a2.status != OK:
            return Check(name, UNVERIFIABLE,
                         f"(A2) not confirmed (status: {a2.status}), so "
                         "M~ = L^2/(2 C_coe) + M is undefined; " + after,
                         None, extra)
        need = problem.L ** 2 / (2 * a2.value)
        why = f"L^2/(2 C_coe) = {need:.3g}"
    margin = coerce - need
    if margin <= _TOL:
        return Check(name, FAIL,
                     f"{why}: {basis} leaves M = {margin:.3g} <= 0; {after}",
                     None, extra)
    return Check(name, OK if dirichlet else ESTIMATED,
                 f"{why}: {basis}, M = {margin:.3g}; {after}", margin, extra)


# -- persistent excitation --------------------------------------------------





def _pe_gram(problem, z, q_lin, inv):
    """||f_q(q~, z) p||_U*^2 as a matrix on Q_h."""
    return _gram(problem, _coupling_matrix(problem, z, q_lin), inv)


@_blas
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


@_blas
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


@_blas
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


@_blas
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
        lip = ("L(||q||_H) declared" if a1.value is None
               else f"L = {a1.value:.3g}")
        rho = (a3.extra or {}).get("rho")
        # KN21 (22) with ||v||_H <= C_U->H ||v||_U: the rate is
        # min{C_coe, 2 M / C_U->H^2}; KN21 (23) and the AMMC paper print
        # 2 M C_U->H (docs/DEVIATIONS.md).
        rate = ("min{C_coe, 2 M / C_U->H^2}" if rho is None else
                f"min{{C_coe, 2 M rho_h}} = "
                f"{min(a2.value, 2 * a3.value * rho):.3g} (rho_h = "
                f"{rho:.3g}, 1/C_U->H^2 <= rho_h; the papers print 2 M "
                "C_U->H)")
        return Check(name, OK,
                     f"the decay estimate holds: {lip}, C_coe >= "
                     f"{a2.value:.3g}, M = {a3.value:.3g}; rate C' = {rate}")
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
