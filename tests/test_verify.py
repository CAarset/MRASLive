"""The theory checklist (mras/verify.py) must give the documented verdict
for every example, past or future: confirmed, honestly failed, estimated
from data, documented deviation, or explicitly unverifiable -- never a
silent skip.

The expected verdicts are the audit results of the paper's examples:

* sigma = 0 family (Darcy, Fisher, Burgers pipe): adjoint pairing at
  machine precision, f affine in q, (A2) FAILS as a claim (gradient
  coupling: degenerate on P0, sign-indefinite relaxation; the paper does
  not claim it, Remark 4.2), (A3) therefore unverifiable;
* Burgers pipe additionally: reconstruction does not descend at the
  quick-check horizon (t = 0.5) -- the banded wake field is felt on the
  forcing timescale of several time units, the plateau the live notebook
  documents (persistent-excitation caveat, DEVIATIONS.md);
* Arrhenius pair: every rung confirmed -- the reference implementation of
  the abstract scheme;
* potential / allen_cahn: paper-literal rows -> adjoint DEVIATION; f
  nonlinear in q, so the (A1) remainder is ESTIMATED numerically (the
  declaration-free path: the checklist warns instead of trusting); both
  uniformly monotone on their actual datum range (positive a.e. datum:
  the AC nodal undershoot at sine extrema is a P3 interpolation artifact,
  cell means stay >= 1 - |u_wot| > 0); C norm-dependent -> (A3)
  unverifiable;
* helmholtz: parameter absent from the truth model: the ladder is waived
  (theory_exempt) and the self-step is a documented deviation.
"""

import re

import pytest
from ngsolve import BilinearForm, LinearForm, dx, grad

from mras.domains import darcy
from mras.problems import AllenCahnProblem, DarcyProblem
from mras.problems.arrhenius import _ArrheniusBase
from mras.scheme import Problem, TheorySpecError
from mras.verify import (DEVIATION, ESTIMATED, FAIL, OK, UNVERIFIABLE,
                         _datum_array, _datum_cells_mean, check_example,
                         run_checks)

EXPECTED = {
    "darcy": {"adjoint": OK, "coupling pre-verif": OK, "(A1)": OK,
              "(A2)": FAIL, "(A3)": UNVERIFIABLE, "persistent": OK,
              "reconstruction": OK, "seeded": OK},
    "fisher": {"adjoint": OK, "coupling pre-verif": OK, "(A1)": OK,
               "(A2)": FAIL, "(A3)": UNVERIFIABLE, "persistent": OK,
               "reconstruction": OK, "seeded": OK},
    "burgers": {"adjoint": OK, "coupling pre-verif": OK, "(A1)": OK,
                "(A2)": FAIL, "(A3)": UNVERIFIABLE, "persistent": OK,
                "reconstruction": FAIL, "seeded": OK},
    "arrhenius": {"adjoint": OK, "coupling pre-verif": OK, "(A1)": OK,
                  "(A2)": OK, "(A3)": OK, "persistent": OK,
                  "reconstruction": OK, "seeded": OK},
    "arrhenius_linear": {"adjoint": OK, "coupling pre-verif": OK, "(A1)": OK,
                         "(A2)": OK, "(A3)": OK, "persistent": OK,
                         "reconstruction": OK, "seeded": OK},
    "potential": {"adjoint": DEVIATION, "coupling pre-verif": OK,
                  "(A1)": ESTIMATED, "(A2)": OK, "(A3)": UNVERIFIABLE,
                  "persistent": OK, "reconstruction": OK, "seeded": OK},
    "allen_cahn": {"adjoint": DEVIATION, "coupling pre-verif": OK,
                   "(A1)": ESTIMATED, "(A2)": OK, "(A3)": UNVERIFIABLE,
                   "persistent": OK, "reconstruction": OK, "seeded": OK},
    "helmholtz": {"adjoint": DEVIATION, "coupling pre-verif": UNVERIFIABLE,
                  "(A1)": UNVERIFIABLE, "(A2)": UNVERIFIABLE,
                  "(A3)": UNVERIFIABLE, "persistent": OK,
                  "reconstruction": UNVERIFIABLE, "seeded": OK},
}


def _by_prefix(checks):
    out = {}
    for c in checks:
        for key in EXPECTED["darcy"]:
            if c.name.startswith(key) or key in c.name:
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

    @staticmethod
    def rprime(w):
        return 1 + 2 * w

    def c_coe(self):
        lo, hi = self.essinf_datum(), self.data_range()[1]
        cands = [self.r(lo), self.r(hi)]
        if lo < -0.5 < hi:
            cands.append(0.75)
        return float(min(cands))


class BareQuadraticProblem(QuadraticRateProblem):
    """Same physics, no theory hooks declared and one honest waiver: the
    degradation contract -- every check still fires, and the rungs that
    need declared structure report UNVERIFIABLE instead of passing
    silently.  The adjoint probe must still pass: it probes the ASSEMBLED
    forms, requiring no declaration at all."""

    coupling = None
    coupling_adjoint = None
    c_coe = None
    C_scale = None
    theory_exempt = "degradation probe"


def _run_quadratic(cls):
    p = cls(h=0.4, dt=0.05, tend=1.5, data_h=0.2)
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


def test_new_example_without_hooks_degrades_to_flagged_unverifiable():
    got = _run_quadratic(BareQuadraticProblem)
    expected = {"adjoint": OK,                 # numeric probe of the forms
                "coupling pre-verif": UNVERIFIABLE,  # ladder waived
                "(A1)": UNVERIFIABLE,          # no f-hook to classify
                "(A2)": UNVERIFIABLE,          # no structure to probe
                "(A3)": UNVERIFIABLE,          # no C_scale, no C_coe
                "persistent": OK, "reconstruction": OK}
    got_status = {k: c.status for k, c in got.items()}
    assert got_status == expected, {k: c.detail for k, c in got.items()}


class SquaredQProblem(QuadraticRateProblem):
    """A nonlinear-in-q specimen of the SAME coupling hook: f(q, z) =
    q^2 r(z).  Nothing informs the checklist of the q^2 except the hook
    itself: it must notice (A1) is not affine, ESTIMATE the remainder
    (with a warning, since an estimate must never masquerade as a
    declaration), and refuse to let the estimate feed (A3); the (A2)
    pairing q^2 -> 2 q^3-like cubic form is sign-indefinite over random
    patch pairs, so (A2) must FAIL as a claim; and the coupling
    pre-verification must confirm the independently transcribed adjoint
    2 q r(z) before any solving.  The solver rows exist only to make the
    run possible; the assertions concern the pre-solve ladder."""

    name = "squared_q"
    sigma = 1.0     # f nonlinear in q: the sigma Dt z residual is active
    L = None        # no declaration: force the estimation path
    c_coe = None    # the r-based witness would be a lie for q^2

    def initial_guess(self):
        return 0.3  # a q^2 flow starting at 0 has zero drive

    def coupling(self, q, z, s):
        return q * q * self.r(z) * s

    def coupling_adjoint(self, q, z, s):
        return 2 * q * self.r(z) * s

    def forms(self, state):
        dt = self.dt
        z, z_old, gfu, gfq = state.z, state.z_old, state.gfu, state.gfq
        a = BilinearForm(self.X)
        res = LinearForm(self.X)
        # state row, linearised at the lagged estimate: q^2 -> 2 gfq q - gfq^2
        a += self.u * self.v * dx
        a += dt * grad(self.u) * grad(self.v) * dx
        a += 2 * dt * gfq * self.r(z) * self.q * self.v * dx
        res += gfu * self.v * dx
        res += dt * state.g * self.v * dx
        res += dt * gfq * gfq * self.r(z) * self.v * dx
        # parameter row: Dt q = f_q(q~(gfq), z)^* (u - z) = 2 gfq r(z) (u - z)
        a += self.q * self.s * dx
        res += gfq * self.s * dx
        res += 2 * dt * gfq * self.r(z_old) * (gfu - z_old) * self.s * dx
        return a, res


def test_nonlinear_q_ladder_estimates_and_refuses_to_promote():
    p = SquaredQProblem(h=1.2, dt=0.05, tend=1.5, data_h=0.7)
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


def test_incomplete_problem_raises_naming_every_gap():
    class Stub(Problem):
        name = "stub"

        def __init__(self):
            super().__init__(darcy(maxh=0.5, order_u=2))

    with pytest.raises(TheorySpecError) as excinfo:
        Stub()
    msg = str(excinfo.value)
    assert "not runnable as specified" in msg
    assert "forms(state)" in msg
    assert "data hook" in msg and "forward_forms" in msg
    assert "coupling(q, z, s)" in msg
    assert "blank_problem.py" in msg


def test_missing_adjoint_hook_warns_once():
    class HalfSpecified(DarcyProblem):
        name = "half_specified"
        coupling_adjoint = None

    with pytest.warns(UserWarning, match="coupling_adjoint"):
        HalfSpecified()
