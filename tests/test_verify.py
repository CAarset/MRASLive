"""The theory checklist (mras/verify.py) must give the documented verdict
for every example, past or future: confirmed, honestly failed, documented
deviation, or explicitly unverifiable -- never a silent skip.

The expected verdicts are the audit results of the paper's examples:

* sigma = 0 family (Darcy, Fisher, Burgers pipe): adjoint pairing at
  machine precision, f affine in q, (A2) FAILS as a claim (gradient
  coupling: degenerate on P0, sign-indefinite relaxation; the paper does
  not claim it, Remark 4.2), (A3) therefore unverifiable;
* Arrhenius pair: every rung confirmed -- the reference implementation of
  the abstract scheme;
* potential / allen_cahn: paper-literal rows -> adjoint DEVIATION; f
  nonlinear; both uniformly monotone on their actual datum range
  (positive a.e. datum: the AC nodal undershoot at sine extrema is a P3
  interpolation artifact, cell means stay >= 1 - |u_wot| > 0); C
  norm-dependent -> (A3) unverifiable;
* helmholtz: parameter absent from the truth model: the ladder is
  unverifiable and the self-step is a documented deviation.
"""

import re

import numpy as np
import pytest

from mras.problems import AllenCahnProblem
from mras.problems.arrhenius import _ArrheniusBase
from mras.verify import (DEVIATION, FAIL, OK, UNVERIFIABLE, _datum_array,
                         _datum_cells_mean, check_example, run_checks)

EXPECTED = {
    "darcy": {"adjoint": OK, "(A1)": OK, "(A2)": FAIL, "(A3)": UNVERIFIABLE,
              "persistent": OK, "reconstruction": OK, "seeded": OK},
    "fisher": {"adjoint": OK, "(A1)": OK, "(A2)": FAIL, "(A3)": UNVERIFIABLE,
               "persistent": OK, "reconstruction": OK, "seeded": OK},
    "burgers": {"adjoint": OK, "(A1)": OK, "(A2)": FAIL,
                "(A3)": UNVERIFIABLE, "persistent": OK,
                "reconstruction": OK, "seeded": OK},
    "arrhenius": {"adjoint": OK, "(A1)": OK, "(A2)": OK, "(A3)": OK,
                  "persistent": OK, "reconstruction": OK, "seeded": OK},
    "arrhenius_linear": {"adjoint": OK, "(A1)": OK, "(A2)": OK, "(A3)": OK,
                         "persistent": OK, "reconstruction": OK,
                         "seeded": OK},
    "potential": {"adjoint": DEVIATION, "(A1)": OK, "(A2)": OK,
                  "(A3)": UNVERIFIABLE, "persistent": OK,
                  "reconstruction": OK, "seeded": OK},
    "allen_cahn": {"adjoint": DEVIATION, "(A1)": OK, "(A2)": OK,
                   "(A3)": UNVERIFIABLE, "persistent": OK,
                   "reconstruction": OK, "seeded": OK},
    "helmholtz": {"adjoint": DEVIATION, "(A1)": UNVERIFIABLE,
                  "(A2)": UNVERIFIABLE, "(A3)": UNVERIFIABLE,
                  "persistent": OK, "reconstruction": UNVERIFIABLE,
                  "seeded": OK},
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

    @staticmethod
    def witness(lo):  # unused: c_coe() below takes the range minimum
        return 0.75

    def c_coe(self):
        lo, hi = self.essinf_datum(), self.data_range()[1]
        cands = [self.r(lo), self.r(hi)]
        if lo < -0.5 < hi:
            cands.append(0.75)
        return float(min(cands))


class BareQuadraticProblem(QuadraticRateProblem):
    """Same physics, no theory hooks declared: the honest-degradation
    contract -- every check still fires, and the rungs that need a
    declared structure report UNVERIFIABLE instead of passing silently.
    The adjoint probe must still pass: it probes the ASSEMBLED forms,
    requiring no declaration at all."""

    coupling_kernel = None
    monotone_pair = None
    c_coe = None
    C_scale = None


def _run_quadratic(cls):
    p = cls(h=0.4, dt=0.05, tend=1.5, data_h=0.2)
    res = p.run_level(0, progress=False)
    checks = run_checks(p, res.state, res.error_q)
    keys = ("adjoint", "(A1)", "(A2)", "(A3)", "persistent",
            "reconstruction")
    out = {}
    for c in checks:
        out.update({k: c for k in keys if c.name.startswith(k) or k in c.name})
    return out


def test_new_example_with_hooks_gets_the_full_ladder_automatically():
    got = _run_quadratic(QuadraticRateProblem)
    assert len(got) == 6, [c.name for c in got.values()]
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
    expected = {"adjoint": OK,              # numeric probe of the forms
                "(A1)": UNVERIFIABLE,       # no f-hook to classify
                "(A2)": UNVERIFIABLE,       # no structure to probe
                "(A3)": UNVERIFIABLE,       # no C_scale, no C_coe
                "persistent": OK, "reconstruction": OK}
    got_status = {k: c.status for k, c in got.items()}
    assert got_status == expected, {k: c.detail for k, c in got.items()}

