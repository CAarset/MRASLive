"""Automated theory checklist: every example through the same questions.

The paper's convergence theory (Assumption 2.2: (A1) Lipschitz remainder,
(A2) monotonicity with constant C_coe, (A3) tracking coercivity above
M~ = L^2 / (2 C_coe) + M) plus the adjoint pairing behind the parameter
row make up a fixed list of claims.  ``run_checks`` asks each claim of a
problem instance and reports one of

    ok             confirmed numerically (margin reported)
    fail         contradicted numerically (the honest negative result)
    deviation    expected deviation, documented by ``adjoint_deviation``
    unverifiable no probe exists or the pairing degenerates discretely;
                 flagged, never silently skipped

Problems expose the theory through the hooks added to
:class:`mras.scheme.Problem` (``coupling_kernel`` for f affine in q,
``monotone_pair`` for pointwise nonlinear f, ``C_scale``,
``adjoint_deviation``); a missing hook yields ``unverifiable`` with the
reason.  ``scripts/checklist.py`` runs the whole registry and writes the
markdown table that backs the report's theory column.

The (A2) probe evaluates the pairing ``int [f(q,z) - f(p,z)] (q - p) dx``
on random P0 directions and the first, middle and last noise-free datum.
For gradient couplings (Darcy, Fisher, Burgers) this pairing is exactly
zero on the discrete P0 parameter space - grad q = 0 a.e. - which is a
degeneration of the *discretisation*, so the check additionally evaluates
the smoothed relaxation (pairing against the H1-regularised lift of the
direction); a sign-indefinite relaxation reports ``fail`` with that note,
matching the paper's own refusal to claim (A2) there (Remark 4.2).
"""

from collections import namedtuple
import importlib.util
import pathlib
import re

import numpy as np
from ngsolve import (BilinearForm, GridFunction, Integrate, LinearForm,
                     Preconditioner, dx, grad)
from ngsolve.solvers import BVP

from .adjoint import adjoint_mismatch

ROOT = pathlib.Path(__file__).resolve().parents[1]

OK, FAIL, UNVERIFIABLE, DEVIATION = "ok", "fail", "unverifiable", "deviation"

Check = namedtuple("Check", "name status detail")

_TOL = 1e-12


def _int(problem, cf):
    return float(Integrate(cf, problem.mesh))


def _datum_samples(problem, n=3):
    """Noise-free datum as U_h GridFunctions at spread-out times.

    Works for every problem: the datum mechanism ``problem.data`` is
    evaluated on a fresh zero-level state (no noise draw), which covers
    both generated snapshots and analytic references.
    """
    from .scheme import State
    out = []
    for frac in np.linspace(0.05, 0.95, n):
        state = State(problem, 0.0)
        problem.init_level(state)
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


def _l2sq(problem, cf):
    return _int(problem, cf * cf)


def _lift(problem, dq):
    """H1-regularised representative of a P0 direction (mass + dt*stiff)."""
    U = problem.Ufes
    u, v = U.TrialFunction(), U.TestFunction()
    a = BilinearForm(U)
    a += u * v * dx + problem.dt * grad(u) * grad(v) * dx
    lf = LinearForm(U)
    lf += dq * v * dx
    a.Assemble()
    lf.Assemble()
    out = GridFunction(U)
    BVP(bf=a, lf=lf, gf=out, pre=Preconditioner(a, "local"), print=False)
    return out


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


def check_a1_sigma(problem):
    """(A1) Lipschitz remainder and the sigma it forces (sigma = 0 iff f
    is affine in q)."""
    kernel = problem.coupling_kernel if callable(problem.coupling_kernel) \
        else None
    pair = problem.monotone_pair if callable(problem.monotone_pair) else None
    if kernel is not None:
        note = "" if problem.sigma == 0 else \
            f" -- declared sigma = {problem.sigma} adds an implicit term " \
            "beyond the affine scheme"
        return Check("(A1) affine / sigma", OK if problem.sigma == 0 else FAIL,
                     "f affine in q: remainder L = 0 exactly (any L >= 0 "
                     f"admissible; declared L = {problem.L})" + note)
    if pair is not None:
        want = "sigma = 1 required" if problem.sigma == 1 else \
            f"declared sigma = {problem.sigma}"
        return Check("(A1) nonlinear / sigma",
                     OK if problem.sigma == 1 else FAIL,
                     f"f nonlinear in q: local Lipschitz (bounded sets), "
                     f"global constant not probed; {want}; sampled curvature "
                     "bound reported by the (A2) probe detail")
    return Check("(A1) Lipschitz / sigma", UNVERIFIABLE,
                 "no f-hook: f neither affine (coupling_kernel) nor pointwise "
                 "(monotone_pair) as declared by the problem")


def _a2_kernel_probe(problem, samples, seed=0, n=6):
    rng = np.random.default_rng(seed)
    deg, lifted = [], []
    for z in samples:
        for _ in range(n):
            dq = _random_q(problem, rng)
            nrm = _l2sq(problem, dq)
            dq_u = GridFunction(problem.Ufes)
            dq_u.Set(dq)  # P0 direction viewed in U_h
            num = _int(problem, dq * problem.coupling_kernel(z, dq_u))
            deg.append(num / nrm)
            v = _lift(problem, dq)
            lifted.append(_int(problem, dq * problem.coupling_kernel(z, v))
                          / nrm)
    return min(deg), min(lifted)


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


def _a2_pair_probe(problem, samples, seed=0, n=12):
    """Localise the test pairs on a few cells: (A2) must hold for every
    q, i.e. pointwise in the space-time cylinder, and full-domain pairs
    only probe the datum-weighted mean of the monotonicity density, which
    averages out sign-changing data."""
    rng = np.random.default_rng(seed)
    vals = []
    for z in samples:
        for _ in range(n):
            qa, qb = _patch_q(problem, rng), _patch_q(problem, rng)
            nrm = _l2sq(problem, qa - qb)
            vals.append(_int(problem, problem.monotone_pair(qa, qb, z)) / nrm)
    return min(vals)


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
    if callable(getattr(problem, "c_coe", None)):
        est = problem.c_coe()
        if est > _TOL:
            return Check("(A2) monotonicity", OK,
                         f"witnessed C_coe >= {est:.3g} (analytic: weak "
                         "essential lower bound of the datum, c_coe())")
        return Check("(A2) monotonicity", FAIL,
                     f"analytic witness c_coe() = {est:.3g} <= 0")
    if callable(problem.coupling_kernel) and not callable(
            problem.monotone_pair):
        d, l = _a2_kernel_probe(problem, samples)
        if d > _TOL:
            return Check("(A2) monotonicity", OK,
                         f"witnessed C_coe >= {d:.3g} (min over "
                         "sampled datum and directions)")
        if abs(d) <= 1e-10:
            if l > _TOL:
                return Check("(A2) monotonicity", UNVERIFIABLE,
                             "pairing is exactly 0 on the discrete P0 space "
                             "(grad q = 0 a.e.); smoothed relaxation positive "
                             f"(min {l:.3g}) - continuous claim not decidable")
            return Check("(A2) monotonicity", FAIL,
                         "pairing degenerates on P0 (0 a.e.) and the smoothed "
                         f"relaxation is sign-indefinite (min {l:.3g}): "
                         "gradient coupling, as the paper concedes "
                         "(Remark 4.2: not claimed for this example)")
        return Check("(A2) monotonicity", FAIL,
                     f"min pairing ratio {d:.3g}: sign-indefinite pairing, "
                     "gradient coupling - (A2) not claimed by the paper for "
                     "this example (Remark 4.2)")
    if callable(problem.monotone_pair):
        m = _a2_pair_probe(problem, samples)
        if m > _TOL:
            return Check("(A2) monotonicity", OK,
                         f"witnessed C_coe >= {m:.3g} on sampled range")
        return Check("(A2) monotonicity", FAIL,
                     f"sampled pairing reaches {m:.3g} < 0: no monotonicity "
                     "witness on the sampled datum range")
    return Check("(A2) monotonicity", UNVERIFIABLE,
                 "no f-hook declared; parameter may be absent from the "
                 "state model entirely (check the example docstring)")


def _lambda1(problem, iters=200, seed=0):
    """Smallest eigenvalue of C = C_scale * (-Laplace) relative to L2 mass,
    by CG-preconditioned inverse iteration with mass normalisation."""
    from ngsolve.solvers import CG
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


def check_a3(problem, a2):
    """(A3): lambda_1(C) > L^2 / (2 C_coe) (then M~ can absorb any margin)."""
    if problem.C_scale is None:
        return Check("(A3) coercivity", UNVERIFIABLE,
                     "C is norm-dependent (c_value of |q|_L2^(2/3)): no "
                     "parameter-free coercivity bound to probe")
    if a2.status != OK:
        return Check("(A3) coercivity", UNVERIFIABLE,
                     f"(A2) not confirmed (status: {a2.status}), so "
                     "M~ = L^2/(2 C_coe) + M is undefined")
    est = float(re.search(r">=\s*([0-9.eE+-]+)", a2.detail).group(1))
    need = problem.L ** 2 / (2 * est)
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


def check_reproducible(build, cfg, level=0.05):
    """Same seed must reproduce the same error histories exactly."""
    r1 = build(**cfg).run_level(level, progress=False).error_u
    r2 = build(**cfg).run_level(level, progress=False).error_u
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
    a2 = check_a2(problem, samples)
    return [
        check_adjoint(problem, state),
        check_a1_sigma(problem),
        a2,
        check_a3(problem, a2),
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
    """Build, run one noise-free level, and check the instance."""
    build, cfg = _registry()[name]
    problem = build(**cfg)
    result = problem.run_level(0, progress=False)
    checks = run_checks(problem, result.state, result.error_q, seed=seed)
    if quick:
        checks.append(check_reproducible(build, cfg))
    return checks


STATUS_ORDER = {OK: 0, DEVIATION: 1, UNVERIFIABLE: 2, FAIL: 3}


def as_markdown(name, checks):
    lines = [f"### {name}", "",
             "| check | status | detail |", "|---|---|---|"]
    for c in checks:
        mark = {OK: "OK", FAIL: "**FAIL**",
                UNVERIFIABLE: "*unverifiable*",
                DEVIATION: "*deviation*"}[c.status]
        lines.append(f"| {c.name} | {mark} | {c.detail} |")
    return "\n".join(lines)
