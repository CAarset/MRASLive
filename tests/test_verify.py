"""The theory checklist (mras/verify.py) must give the documented verdict
for every example, past or future: confirmed, honestly failed, estimated
from data, documented deviation, or explicitly unverifiable -- never a
silent skip.

The expected verdicts are the audit results of the paper's examples:

* sigma = 0 family (Darcy, Fisher, Burgers 1D): adjoint pairing at
  machine precision, f affine in q, (A2) FAILS as a claim (gradient
  coupling: degenerate on P0, sign-indefinite relaxation; on the P1 Q_h
  of Burgers 1D the exact pairing is indefinite; the paper does not claim
  it, Remark 4.2); L = 0, so (A3) is C's coercivity alone, M~ = M = 1;
* Burgers 2D (vector) shares that family; the vector state is probed
  along each velocity component; its inflow descends from the start
  (0.44 -> 0.34 by its audit horizon t = 1, from qhat_0 = 3/4), and the
  Burgers 1D beams descend (1 -> 0.81 by t = 1);
* persistent excitation over the audit horizon: positive for Darcy,
  Fisher, the Arrhenius pair, potential and Allen-Cahn; zero to the
  accuracy of the solve for both Burgers examples, whose data up to t = 1
  have not reached the top corner of the inlet (1D, beyond the beams) and
  the outlet (2D, before the inflow arrives);
* convergence (what the declared sigma gives): sigma = 0 with f affine in
  q converges through the persistent excitation, so the PE verdict
  carries over (ok for Darcy, Fisher and Arrhenius, fail for both Burgers
  over t <= 1); sigma = 1 (potential, Allen-Cahn) has the decay estimate
  only up to the unconfirmed (A1) and (A3): unverifiable;
* Arrhenius pair: every rung confirmed -- the reference implementation of
  the abstract scheme; its sigma = 1 twin too, with the decay estimate as
  its convergence statement and its strong form verified against the
  weak one;
* potential / allen_cahn: adjoint pairing at machine precision (the
  scheme is built from the declared pair); f nonlinear in q, so the (A1)
  remainder is ESTIMATED numerically (the declaration-free path: the
  checklist warns instead of trusting); (A2) confirmed by the declared
  bounds c_coe() = ess inf u^dagger (potential) and (ess inf u^dagger)^3
  (Allen-Cahn), which the sampled pairings exceed; the state prescribed
  on the boundary, C = c(||q||) (-Laplace) is coercive for every c > 0,
  but the bound c >= L^2/(2 C_coe) + M needs L(||q||): (A3) unverifiable.
"""

import re
from types import SimpleNamespace

import pytest
from ngsolve import IfPos, Integrate, x, y

from mras.domains import darcy
from mras.live import load_example
from mras.scheme import MRASProblem, TheorySpecError
from mras.verify import (ESTIMATED, FAIL, OK, UNVERIFIABLE, Check,
                         _datum_array, _datum_cells_mean, _datum_samples,
                         _strong_gap,
                         check_a2, check_convergence_route, check_example,
                         check_pe_after, persistent_excitation, run_checks)

# The examples live in examples/ and are loaded by the runner, not imported.
AllenCahnProblem = load_example("allen_cahn").PROBLEM
DarcyProblem = load_example("darcy").PROBLEM
_ArrheniusBase = load_example("arrhenius")._ArrheniusBase

EXPECTED = {
    "darcy": {"adjoint": OK, "coupling pre-verif": OK, "(A1)": OK,
              "(A2)": FAIL, "(A3)": OK, "persistent": OK, "Gramian": OK,
              "convergence": OK, "reconstruction": OK, "seeded": OK},
    "fisher": {"adjoint": OK, "coupling pre-verif": OK, "(A1)": OK,
               "(A2)": FAIL, "(A3)": OK, "persistent": OK, "Gramian": OK,
               "convergence": OK, "reconstruction": OK, "seeded": OK},
    "burgers_1d": {"adjoint": OK, "coupling pre-verif": OK, "(A1)": OK,
                   "(A2)": FAIL, "(A3)": OK, "persistent": FAIL, "Gramian": FAIL,
                   "convergence": FAIL, "reconstruction": OK, "seeded": OK},
    "arrhenius": {"adjoint": OK, "coupling pre-verif": OK, "(A1)": OK,
                  "(A2)": OK, "(A3)": OK, "persistent": OK, "Gramian": OK,
                  "convergence": OK, "reconstruction": OK, "seeded": OK},
    "arrhenius_linear": {"adjoint": OK, "coupling pre-verif": OK,
                         "(A1)": OK, "(A2)": OK, "(A3)": OK,
                         "persistent": OK, "Gramian": OK, "convergence": OK,
                         "reconstruction": OK, "seeded": OK},
    "arrhenius_sigma1": {"adjoint": OK, "coupling pre-verif": OK,
                         "(A1)": OK, "(A2)": OK, "(A3)": OK,
                         "persistent": OK, "Gramian": OK, "convergence": OK,
                         "reconstruction": OK, "seeded": OK},
    "potential": {"adjoint": OK, "coupling pre-verif": OK,
                  "(A1)": ESTIMATED, "(A2)": OK, "(A3)": UNVERIFIABLE,
                  "persistent": OK, "Gramian": OK, "convergence": UNVERIFIABLE,
                  "reconstruction": OK, "seeded": OK},
    "allen_cahn": {"adjoint": OK, "coupling pre-verif": OK,
                   "(A1)": ESTIMATED, "(A2)": OK, "(A3)": UNVERIFIABLE,
                   "persistent": OK, "Gramian": OK, "convergence": UNVERIFIABLE,
                   "reconstruction": OK, "seeded": OK},
    "burgers_2d": {"adjoint": OK, "coupling pre-verif": OK, "(A1)": OK,
                   "(A2)": FAIL, "(A3)": OK, "persistent": FAIL, "Gramian": FAIL,
                   "convergence": FAIL, "reconstruction": OK, "seeded": OK},
}


def _by_prefix(checks):
    """Each row under its key; "Gramian" before "persistent", which both
    PE rows contain."""
    keys = sorted(EXPECTED["darcy"], key=lambda k: k != "Gramian")
    out = {}
    for c in checks:
        key = next((k for k in keys if c.name.startswith(k) or k in c.name),
                   None)
        if key is not None:
            out[key] = c.status
    return out


def test_datum_negative_nodal_values_are_interpolation_artifacts():
    """The AC datum's nodal P3 values ring below zero (sine extrema), but
    (A2) is an a.e. statement: cell means of the datum stay positive, so
    the localised pair probe's OK verdict witnesses the true (a.e.) datum
    range, not an interpolation artifact (cf. the weak ess-inf lesson for
    the Arrhenius pair)."""
    p = AllenCahnProblem(h=0.25, dt=0.05, tend=0.5)
    p.run_level(0, progress=False)
    assert (_datum_array(p) < 0).mean() > 0.25      # nodal ringing exists
    assert _datum_cells_mean(p).min() > 0.5         # cell means positive


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_checklist_verdicts_match_the_documented_audit(name):
    checks = check_example(name)
    got = _by_prefix(checks)
    expected = EXPECTED[name]
    assert set(got) == set(expected), (name, [c.name for c in checks])
    bad = {k: (v, got[k]) for k, v in expected.items() if got[k] != v}
    assert not bad, (name, [(c.name, c.status, c.detail) for c in checks])


class QuadraticRateProblem(_ArrheniusBase):
    """A deliberately NEW example for the checklist: rate law
    r(w) = 1 + w + w^2, uniformly positive (min 3/4 at w = -1/2), so
    (A2) holds globally with C_coe = ess inf r(z) = 3/4 whenever the
    datum dips below -1/2 (it does: its weak ess inf is ~ -0.6).  The
    analytic witness must minimise r over the datum range -- r(ess inf)
    alone would over-report (0.76 vs 0.75) -- which is exactly the hook
    a future author implements; nothing in mras/verify.py is modified
    or even informed of this problem."""

    name = "quadratic_rate"

    @staticmethod
    def r(w):
        return 1 + w + w * w

    def c_coe(self):
        lo, hi = self.essinf_datum(), self.data_range()[1]
        cands = [self.r(lo), self.r(hi)]
        if lo < -0.5 < hi:
            cands.append(0.75)
        return float(min(cands))


def _run_quadratic(cls):
    # h = 0.5 keeps Q_h within the dense PE eigenproblem
    p = cls(h=0.5, dt=0.05, tend=1.5)
    res = p.run_level(0, progress=False)
    checks = run_checks(p, res.state, res.error_q)
    keys = ("adjoint", "coupling pre-verif", "(A1)", "(A2)", "(A3)",
            "persistent", "reconstruction")
    out = {}
    for c in checks:
        out.update({k: c for k in keys
                    if c.name.startswith(k) or k in c.name})
    return out


def test_new_example_with_hooks_gets_the_full_ladder_automatically():
    got = _run_quadratic(QuadraticRateProblem)
    assert len(got) == 7, [c.name for c in got.values()]
    for key, c in got.items():
        assert c.status == OK, (key, c.status, c.detail)
    # the (A2) rung must use the analytic witness, at the honest value:
    est = float(re.search(r">=\s*([0-9.eE+-]+)", got["(A2)"].detail).group(1))
    assert 0.7 <= est <= 0.76, got["(A2)"].detail
    assert "analytic" in got["(A2)"].detail
    # adjoint to numerical precision, no author-side effort:
    assert "machine precision" in got["adjoint"].detail


class SquaredQProblem(QuadraticRateProblem):
    """A nonlinear-in-q specimen of the SAME coupling hook: f(q, z) =
    q^2 r(z).  Nothing informs the checklist of the q^2 except the hook
    itself: it must notice (A1) is not affine, ESTIMATE the remainder
    (with a warning, since an estimate must never masquerade as a
    declaration), and refuse to let the estimate feed (A3); the (A2)
    pairing q^2 -> 2 q^3-like cubic form is sign-indefinite over random
    patch pairs, so (A2) must FAIL as a claim; and the coupling
    pre-verification must confirm the independently transcribed adjoint
    2 q r(z) before any solving.  The step itself is the driver's, built
    from the same two hooks."""

    name = "squared_q"
    sigma = 1.0     # f nonlinear in q: the sigma Dt z residual is active
    L = None        # no declaration: force the estimation path
    c_coe = None    # the r-based witness would be a lie for q^2
    # On this coarse mesh the elementwise Laplacian of the datum is 60 %
    # off (the strong-form check rightly fails it); this ladder is about
    # (A1) and (A2), so the problem keeps the weak sigma-bracket.
    model_strong = None

    def initial_guess(self):
        return 0.3  # a q^2 flow starting at 0 has zero drive

    def coupling(self, q, z, s):
        return q * q * self.r(z) * s

    def coupling_adjoint(self, q, z, s):
        return 2 * q * self.r(z) * s


def test_nonlinear_q_ladder_estimates_and_refuses_to_promote():
    p = SquaredQProblem(h=1.2, dt=0.05, tend=1.5, k_q=0)
    res = p.run_level(0, progress=False)
    with pytest.warns(UserWarning, match="estimated numerically"):
        checks = run_checks(p, res.state, res.error_q)
    got = {}
    for c in checks:
        key = ("coupling pre-verification" if c.name.startswith("coupling")
               else c.name.split()[0])
        got[key] = c
    a1 = got["(A1)"]
    assert a1.status == ESTIMATED, a1.detail
    assert "L_hat" in a1.detail and "declare L" in a1.detail
    # an estimate must not feed the (A3) coercion bound:
    a3 = got["(A3)"]
    assert a3.status == UNVERIFIABLE, a3.detail
    a2 = got["(A2)"]
    assert a2.status == FAIL and "monotonicity violated" in a2.detail, a2.detail
    pv = got["coupling pre-verification"]
    assert pv.status == OK and "q-derivative" in pv.detail, pv.detail
    # sigma = 1 on an indefinite (A2) pairing: the energy can grow
    route = got["convergence"]
    assert route.status == FAIL and "indefinite" in route.detail, route.detail


def _route(sigma, linear, a2=OK, pe=OK):
    problem = SimpleNamespace(sigma=sigma, C_scale=None)
    return check_convergence_route(
        problem, linear, Check("(A1)", OK, "", 0.0),
        Check("(A2)", a2, "", 0.75), Check("(A3)", OK, "", 1.0),
        Check("persistent excitation", pe, "", 1e-3))


def test_convergence_follows_the_declared_sigma():
    """sigma is the author's choice; the verdict is what the energy
    identity gives for it."""
    affine0 = _route(0, True)       # through the excitation, not (A2)
    assert affine0.status == OK and "sigma = 1 would give" in affine0.detail
    assert _route(0, True, pe=FAIL).status == FAIL
    affine1 = _route(1, True)       # allowed: the decay estimate
    assert affine1.status == OK and "decay estimate holds" in affine1.detail
    assert _route(1, True, a2=FAIL).status == FAIL
    assert _route(1, False, a2=ESTIMATED).status == ESTIMATED
    nonlinear0 = _route(0, False)   # nothing absorbs the remainder
    assert nonlinear0.status == FAIL and "no estimate" in nonlinear0.detail


class MonotoneQProblem(QuadraticRateProblem):
    """f(q, z) = (q + q^3) r(z): nonlinear in q and monotone (slope >= 1,
    r >= 3/4), so (A2) holds with C_coe >= 3/4.  Sampled pairs can only
    bound C_coe from above: undeclared, the verdict is an estimate."""

    name = "monotone_q"
    sigma = 1.0
    L = None
    c_coe = None

    def coupling(self, q, z, s):
        return (q + q * q * q) * self.r(z) * s

    def coupling_adjoint(self, q, z, s):
        return (1 + 3 * q * q) * self.r(z) * s


def _a2(cls, affine):
    p = cls(h=1.2, dt=0.05, tend=1.5, k_q=0)
    return check_a2(p, _datum_samples(p), affine)


def test_a2_sampled_minimum_is_an_estimate_until_declared():
    undeclared = _a2(MonotoneQProblem, affine=False)
    assert undeclared.status == ESTIMATED, undeclared.detail
    assert "from above only" in undeclared.detail

    class Declared(MonotoneQProblem):
        def c_coe(self):
            return 0.75

    declared = _a2(Declared, affine=False)
    assert declared.status == OK and declared.value == 0.75, declared.detail


def test_a2_declared_bound_above_the_exact_pairing_fails():
    class Overclaimed(QuadraticRateProblem):
        def c_coe(self):
            return 2.0  # r = 1 + w + w^2 dips to 3/4

    got = _a2(Overclaimed, affine=True)
    assert got.status == FAIL and "the bound is wrong" in got.detail, \
        got.detail
    assert "exact on Q_h" in got.detail


def test_pe_weakest_direction_is_where_the_data_do_not_reach():
    """Burgers 1D: up to t = 1 the beams have not lit the top corner of
    the inlet; the PE eigenvector sits there."""
    p = load_example("burgers_1d").PROBLEMS["burgers_1d"](h=0.35, dt=0.05,
                                                         tend=1.0)
    pe = persistent_excitation(p)
    assert pe["gamma"] <= 1e-10 * pe["top"]
    d = pe["direction"]
    corner = Integrate(IfPos(0.5 - x, IfPos(y - 0.8, d * d, 0), 0), p.mesh)
    assert corner >= 0.9 * Integrate(d * d, p.mesh)


def test_strong_form_is_checked_against_the_weak_one():
    """model_strong (-Laplace u for the Arrhenius family) agrees with the
    weak density on U_h's boundary-free test functions; a transcription
    error -- here a flipped sign -- does not."""
    # On the audit's mesh; at h = 1.2 the elementwise Laplacian of the
    # interpolated datum is itself 60 % off, which the check rightly flags.
    p = QuadraticRateProblem(h=0.5, dt=0.05, tend=1.5)
    samples = _datum_samples(p)
    assert _strong_gap(p, samples) < 0.05

    class Flipped(QuadraticRateProblem):
        def model_strong(self, u):
            return -super().model_strong(u)

    flipped = Flipped(h=0.5, dt=0.05, tend=1.5)
    assert _strong_gap(flipped, _datum_samples(flipped)) > 1.5


def test_pe_after_a_run_samples_the_horizon_it_took():
    """Post hoc, at the audit discretization (CHECK, else QUICK), over the
    run's actual horizon."""
    got = check_pe_after("darcy", 1.0)
    assert got.status == OK and "up to t = 1," in got.detail, got.detail
    dark = check_pe_after("burgers_1d_dark", 0.5)  # QUICK only
    assert dark is not None and "up to t = 0.5" in dark.detail, dark.detail


def test_incomplete_problem_raises_naming_every_gap():
    class Stub(MRASProblem):
        name = "stub"

        def __init__(self):
            super().__init__(darcy(maxh=0.5, order_u=2))

    with pytest.raises(TheorySpecError) as excinfo:
        Stub()
    msg = str(excinfo.value)
    assert "not runnable as specified" in msg
    assert "coupling(q, u, w)" in msg and "coupling_adjoint(q, u, w)" in msg
    assert "blank_problem.py" in msg


def test_missing_adjoint_hook_is_refused():
    # The scheme is built from the declared pair, so the adjoint is not
    # optional: without it there is no parameter row.
    class HalfSpecified(DarcyProblem):
        name = "half_specified"
        coupling_adjoint = None

    with pytest.raises(TheorySpecError, match="coupling_adjoint"):
        HalfSpecified()
