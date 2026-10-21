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

import numpy as np
import pytest

from mras.problems import AllenCahnProblem
from mras.verify import (DEVIATION, FAIL, OK, UNVERIFIABLE, _datum_array,
                         _datum_cells_mean, check_example)

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
