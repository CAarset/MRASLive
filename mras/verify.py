"""Automated theory checklist: every example through the same questions.

The paper's convergence theory (Assumption 2.2: (A1) Lipschitz remainder,
(A2) monotonicity with constant C_coe, (A3) tracking coercivity above
M~ = L^2 / (2 C_coe) + M) plus the adjoint pairing behind the parameter
row make up a fixed list of claims.  ``run_checks`` asks each claim of a
problem instance and reports one of

    ok             confirmed numerically (margin reported)
    fail           contradicted numerically (the honest negative result)
    deviation      expected deviation, documented by ``adjoint_deviation``
    estimated      a numerical bound stands in for an unproven constant;
                   warned, never silently accepted
    unverifiable   no probe exists or the pairing degenerates discretely;
                   flagged, never silently skipped

Everything the ladder needs from the model is the single hook
``coupling(q, z, s)`` (the q-dependent part of f as a weak functional) on
:class:`mras.scheme.Problem`: the same hook drives the affinity probe that
decides between sigma = 0 and sigma = 1, the Lipschitz estimator when f is
nonlinear in q, and the (A2) monotonicity probe.  A second, independently
transcribed hook ``coupling_adjoint(q, z, s)`` lets a pre-verification
compare the declared pairing density against the numerical q-derivative of
``coupling`` before any solving; missing hooks yield ``unverifiable`` with
the reason.  ``scripts/checklist.py`` runs the whole registry and writes
the markdown table that backs the report's theory column.

The (A2) probe evaluates the pairing ``int [f(q,z) - f(p,z)] (q - p) dx``
on random P0 patch pairs and the first, middle and last noise-free datum.
For gradient couplings (Darcy, Fisher, Burgers) this pairing is exactly
zero on the discrete P0 parameter space - grad q = 0 a.e. - which is a
degeneration of the *discretisation*, so the check additionally evaluates
the smoothed relaxation (the pairing tested against the H1-conforming
view of the difference in U_h); a sign-indefinite relaxation reports
``fail`` with that note, matching the paper's own refusal to claim (A2)
there (Remark 4.2).
"""

from collections import namedtuple
import importlib.util
import pathlib
import re
import warnings

import numpy as np
from ngsolve import (BilinearForm, GridFunction, Integrate, LinearForm,
                     Preconditioner, dx, grad)
from ngsolve.solvers import BVP, CG

from .adjoint import adjoint_mismatch

ROOT = pathlib.Path(__file__).resolve().parents[1]

OK, FAIL, UNVERIFIABLE, DEVIATION, ESTIMATED = (
    "ok", "fail", "unverifiable", "deviation", "estimated")

Check = namedtuple("Check", "name status detail")

_TOL = 1e-12


def _int(problem, cf):
    return float(Integrate(cf, problem.mesh))


def _datum_samples(problem, n=3):
    """Noise-free datum as U_h GridFunctions at spread-out times.

    Works for every problem: the datum mechanism ``problem.data`` is
    evaluated on a fresh noise-free state (no noise draw), which covers
    both generated snapshots and analytic references.
    """
    from .scheme import State
    out = []
    for frac in np.linspace(0.05, 0.95, n):
        state = State(problem, 0.0)
        problem.init_run(state)
        state.t.Set(problem.tend * frac)
        state.counter = max(1, int(round(frac * problem.tsteps)))
        problem.data(state)
        out.append(state.z)
    return out


def _datum_array(problem, n=25):
    gf = GridFunction(problem.Ufes)
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
    R = f(q+d, z) - f(q, z) - f_q(q, z) d, measured as max ||R|| / ||d||:
    f_q(q, .) d is the central difference of the hook, and ||R|| the Riesz
    norm of the residual functional in K = mass + dt*stiffness (the dual
    of which dominates the L2 pairing that the scheme applies to)."""
    U = problem.Ufes
    u, v = U.TrialFunction(), U.TestFunction()
    a = BilinearForm(U)
    a += u * v * dx + problem.dt * grad(u) * grad(v) * dx
    a.Assemble()
    pre = Preconditioner(a, "local")
    pre.Update()
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
                BVP(bf=a, lf=lf, gf=sol, pre=pre, print=False)
                rho = float(np.sqrt(max(
                    np.asarray(lf.vec) @ np.asarray(sol.vec), 0.0)))
                lhat = max(lhat, rho / np.sqrt(_l2sq(problem, d)))
    return lhat


def _scaled_q(problem, rng, radius):
    """Random P0 field with ||q||_L2 at most ``radius`` (scale-free probes
    need the bounded sets that the local Lipschitz statement runs on)."""
    gf = _random_q(problem, rng)
    norm = np.sqrt(_l2sq(problem, gf))
    if norm > 0:
        gf.vec.FV().NumPy()[:] *= radius * rng.uniform(0.2, 1.0) / norm
    return gf


def check_a1_sigma(problem, samples, affine=None):
    """(A1) Lipschitz remainder and the sigma it forces (sigma = 0 iff f
    is affine in q)."""
    if problem.theory_exempt:
        return Check("(A1) Lipschitz / sigma", UNVERIFIABLE,
                     f"theory ladder waived: {problem.theory_exempt}")
    if not callable(getattr(problem, "coupling", None)):
        return Check("(A1) Lipschitz / sigma", UNVERIFIABLE,
                     "coupling(q, z, s) not declared")
    if affine is None:
        affine, gap = _probe_affine(problem, samples)
    else:
        affine, gap = affine
    if affine:
        if problem.sigma != 0:
            return Check("(A1) Lipschitz / sigma", FAIL,
                         f"f is affine in q (probe gap {gap:.1e}), so the "
                         f"theory forces sigma = 0 -- declared sigma = "
                         f"{problem.sigma} adds an implicit term beyond the "
                         "affine scheme")
        note = "" if problem.L in (None, 0.0) else \
            f" (declared L = {problem.L} unused: affinity gives L = 0)"
        return Check("(A1) Lipschitz / sigma", OK,
                     f"f affine in q on sampled pairs (probe gap {gap:.1e}): "
                     f"remainder L = 0 exactly" + note)
    if problem.sigma != 1:
        return Check("(A1) Lipschitz / sigma", FAIL,
                     f"f is nonlinear in q, so the scheme requires "
                     f"sigma = 1 -- declared sigma = {problem.sigma}")
    if problem.L is not None:
        return Check("(A1) Lipschitz / sigma", OK,
                     f"f nonlinear in q: constant L = {problem.L} declared "
                     "by the author (local Lipschitz on bounded sets)")
    lhat = _estimate_lipschitz(problem, samples)
    warnings.warn(
        f"{problem.name}: (A1) constant estimated numerically "
        f"(L_hat = {lhat:.3g}) -- declare L on the class to confirm it",
        stacklevel=3)
    return Check("(A1) Lipschitz / sigma", ESTIMATED,
                 f"f nonlinear in q: sampled remainder bound L_hat = "
                 f"{lhat:.3g} on radius-3 balls (K-dual norm, "
                 "K = mass + dt*stiffness); declare L to confirm")


# -- (A2): monotonicity -----------------------------------------------------

def _a2_probe(problem, samples, seed=0, n=12, nrelax=6):
    """Localise the test pairs on a few cells: (A2) must hold for every
    q, i.e. pointwise in the space-time cylinder, and full-domain pairs
    only probe the datum-weighted mean of the monotonicity density, which
    averages out sign-changing data.  A pointwise coupling then pairs the
    difference qa - qb against itself in the raw P0 test slot.  Gradient
    couplings are discretely degenerate there (grad q = 0 a.e. on P0, or
    the gradient operator does not even exist on the parameter space), so
    the probe additionally evaluates the smoothed relaxation: global
    directions tested against their U_h view."""
    rng = np.random.default_rng(seed)
    deg = []
    degenerate = False
    for z in samples:
        for _ in range(n):
            qa, qb = _patch_q(problem, rng), _patch_q(problem, rng)
            nrm = _l2sq(problem, qa - qb)
            try:
                deg.append(
                    (_int(problem, problem.coupling(qa, z, qa - qb) * dx)
                     - _int(problem, problem.coupling(qb, z, qa - qb) * dx))
                    / nrm)
            except Exception:
                # grad does not exist on the P0 test space: that absence is
                # itself the discrete degeneracy of a gradient coupling.
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
                view = GridFunction(problem.Ufes)
                view.Set(qa - qb)  # P0 direction viewed in U_h
                relaxed.append(
                    (_int(problem, problem.coupling(qa, z, view) * dx)
                     - _int(problem, problem.coupling(qb, z, view) * dx))
                    / nrm)
        l = min(relaxed)
    return d, l, degenerate


def _datum_cells_mean(problem):
    """Cell means (L2 projection onto Q_h, P0) of sampled data: nodal
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
    """Random P0 field supported on k cells (the whole space if Q is
    coarser), with random signs and amplitudes: the pairing ratio then
    localises the monotonicity density on the support."""
    q = GridFunction(problem.Qfes)
    fv = q.vec.FV().NumPy()
    fv[:] = 0.0
    cells = rng.choice(len(fv), size=min(k, len(fv)), replace=False)
    fv[cells] = rng.choice([-1.0, 1.0], size=len(cells)) \
        * rng.uniform(0.1, 3.0, size=len(cells))
    return q


def check_a2(problem, samples):
    """(A2) monotonicity: int [f(q,z) - f(p,z)] (q - p) >= C_coe |q - p|^2."""
    if problem.theory_exempt:
        return Check("(A2) monotonicity", UNVERIFIABLE,
                     f"theory ladder waived: {problem.theory_exempt}")
    if callable(getattr(problem, "c_coe", None)):
        est = problem.c_coe()
        if est > _TOL:
            return Check("(A2) monotonicity", OK,
                         f"witnessed C_coe >= {est:.3g} (analytic: weak "
                         "essential lower bound of the datum, c_coe())")
        return Check("(A2) monotonicity", FAIL,
                     f"analytic witness c_coe() = {est:.3g} <= 0")
    if not callable(getattr(problem, "coupling", None)):
        return Check("(A2) monotonicity", UNVERIFIABLE,
                     "coupling(q, z, s) not declared; parameter may be "
                     "absent from the state model entirely")
    d, l, degenerate = _a2_probe(problem, samples)
    if d > _TOL:
        return Check("(A2) monotonicity", OK,
                     f"witnessed C_coe >= {d:.3g} (min over sampled datum "
                     "and patch pairs)")
    if degenerate or abs(d) <= 1e-10:
        if l > _TOL:
            return Check("(A2) monotonicity", UNVERIFIABLE,
                         "pairing vanishes on the discrete P0 space (grad q "
                         "= 0 a.e., or grad is undefined there); smoothed "
                         f"relaxation positive (min {l:.3g}) - continuous "
                         "claim not decidable")
        return Check("(A2) monotonicity", FAIL,
                     "pairing degenerates on P0 and the smoothed relaxation "
                     f"is sign-indefinite (min {l:.3g}): gradient coupling, "
                     "as the paper concedes (Remark 4.2: not claimed for "
                     "this example)")
    return Check("(A2) monotonicity", FAIL,
                 f"min pairing ratio {d:.3g}: monotonicity violated on "
                 "sampled patch pairs")


# -- coupling pre-verification ----------------------------------------------

def check_pairing(problem, samples, n=100, seed=0):
    """The parameter row uses <f_q(q, z) p, s> = int p * coupling_adjoint(
    q, z, s) dx; pre-verify the declared density against the numerical
    q-derivative of coupling() on random draws, before any solving."""
    if problem.theory_exempt:
        return Check("coupling pre-verification", UNVERIFIABLE,
                     f"theory ladder waived: {problem.theory_exempt}")
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
    return Check("coupling pre-verification", OK,
                 f"pairing matches the numerical q-derivative of coupling "
                 f"on {n} draws (max ratio {ratio:.1e}; {how})")


def _lambda1(problem, iters=200, seed=0):
    """Smallest eigenvalue of C = C_scale * (-Laplace) relative to L2 mass,
    by CG-preconditioned inverse iteration with mass normalisation."""
    U = problem.Ufes
    u, v = U.TrialFunction(), U.TestFunction()
    a = BilinearForm(U)
    a += problem.C_scale * grad(u) * grad(v) * dx
    m = BilinearForm(U)
    m += u * v * dx
    a.Assemble()
    m.Assemble()
    am, mm = a.mat, m.mat
    pre = Preconditioner(a, "local")
    pre.Update()
    gf = GridFunction(U)
    rhs = LinearForm(U)
    rhs.Assemble()
    rng = np.random.default_rng(seed)
    gf.vec.FV().NumPy()[:] = rng.standard_normal(U.ndof)
    lam = None
    for _ in range(iters):
        rhs.vec.data = np.asarray(mm.matvec(gf.vec)).copy()
        CG(a.mat, rhs.vec, pre=pre, sol=gf.vec, tol=1e-12, maxsteps=3000,
           printrates=False)
        vv = np.asarray(gf.vec)
        mb = np.asarray(mm.matvec(gf.vec))
        nrm = float(mb @ vv)
        gf.vec.FV().NumPy()[:] /= np.sqrt(nrm)
        av = np.asarray(am.matvec(gf.vec))
        mb = np.asarray(mm.matvec(gf.vec))
        vv = np.asarray(gf.vec)
        lam = float(av @ vv) / float(mb @ vv)
    return lam


def check_a3(problem, a2, affine=None):
    """(A3): lambda_1(C) > L^2 / (2 C_coe) (then M~ can absorb any margin)."""
    if problem.C_scale is None:
        return Check("(A3) coercivity", UNVERIFIABLE,
                     "C is norm-dependent (c_value of |q|_L2^(2/3)): no "
                     "parameter-free coercivity bound to probe")
    if a2.status != OK:
        return Check("(A3) coercivity", UNVERIFIABLE,
                     f"(A2) not confirmed (status: {a2.status}), so "
                     "M~ = L^2/(2 C_coe) + M is undefined")
    if problem.L is not None:
        L = problem.L
    elif affine:
        L = 0.0  # affinity probe: the remainder vanishes identically
    else:
        return Check("(A3) coercivity", UNVERIFIABLE,
                     "no (A1) constant available: declare L on the class "
                     "(an estimated L_hat is too weak a basis for M~)")
    est = float(re.search(r">=\s*([0-9.eE+-]+)", a2.detail).group(1))
    need = L ** 2 / (2 * est)
    lam = _lambda1(problem)
    if lam > need:
        return Check("(A3) coercivity", OK,
                     f"lambda_1(C) = {lam:.3g} > L^2/(2 C_coe) = {need:.3g} "
                     f"(margin {lam - need:.3g} available for M)")
    return Check("(A3) coercivity", FAIL,
                 f"lambda_1(C) = {lam:.3g} <= L^2/(2 C_coe) = {need:.3g}")


def check_pe(problem):
    """Persistent excitation proxy: the datum must move over the horizon."""
    snaps = _datum_array(problem)
    mean = snaps.mean(axis=0)
    scale = float(np.linalg.norm(mean))
    var = float(np.linalg.norm(snaps - mean)) / max(scale, 1e-30)
    if var > 0.02:
        return Check("persistent excitation", OK,
                     f"relative temporal variation of the datum: {var:.3g}")
    return Check("persistent excitation", FAIL,
                 f"datum almost time-invariant (variation {var:.3g}): no "
                 "identifiability can be expected (universal proxy only)")


def check_reproducible(build, cfg, noise_level=0.05):
    """Same seed must reproduce the same error histories exactly."""
    r1 = build(**cfg).run_level(noise_level, progress=False).error_u
    r2 = build(**cfg).run_level(noise_level, progress=False).error_u
    same = r1 is not None and np.array_equal(r1, r2)
    return Check("seeded reproducibility", OK if same else FAIL,
                 "identical error histories across two runs" if same
                 else "noise draws differ between same-seed runs")


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
              if callable(getattr(problem, "coupling", None))
              and not problem.theory_exempt else None)
    a2 = check_a2(problem, samples)
    return [
        check_adjoint(problem, state),
        check_pairing(problem, samples, seed=seed),
        check_a1_sigma(problem, samples, affine),
        a2,
        check_a3(problem, a2, affine),
        check_pe(problem),
        check_convergence(error_q),
    ]


# -- registry -----------------------------------------------------------

def _module(stem):
    spec = importlib.util.spec_from_file_location(
        f"{stem}_verify", ROOT / "examples" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _registry():
    # built only on demand: the entries execute the example scripts and
    # pull in the whole problem library, which no other checklist call
    # needs.
    from mras.problems import (AllenCahnProblem, DarcyProblem, FisherProblem,
                               PotentialProblem)
    from mras.problems.arrhenius import (ArrheniusLinearProblem,
                                         ArrheniusProblem)
    return {
        "darcy": (DarcyProblem, dict(h=0.15, dt=0.05, tend=0.5,
                                     data_h=0.08, data_order=2)),
        "fisher": (FisherProblem, dict(h=0.2, dt=0.05, tend=0.5,
                                       data_h=0.15)),
        "burgers": (_module("burgers_pipe").BurgersPipe,
                    dict(h=0.35, dt=0.05, tend=0.5)),
        "arrhenius": (ArrheniusProblem, dict(h=0.3, dt=0.05, tend=1.5,
                                             data_h=0.15)),
        "arrhenius_linear": (ArrheniusLinearProblem,
                             dict(h=0.3, dt=0.05, tend=1.5, data_h=0.15)),
        "potential": (PotentialProblem, dict(h=0.25, dt=0.05, tend=0.5)),
        "allen_cahn": (AllenCahnProblem, dict(h=0.25, dt=0.05, tend=0.5)),
        "helmholtz": (_module("helmholtz_in_parameter").HelmholtzProblem,
                      dict(h=0.2, tend=0.4, tsteps=10)),
    }


def check_example(name, quick=True, seed=0):
    """Build, run one noise-free run (noise level 0%), and check the
    instance."""
    build, cfg = _registry()[name]
    problem = build(**cfg)
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
