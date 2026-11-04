"""Adjoint-pairing verification, every example: <A x, y> = <x, A y>.

The harness (mras/adjoint.py) assembles the linear joint step operator A
from problem.forms(state) and randomises the pairing.  The driver builds
both cross blocks from the declared pair -- the state row's coupling from
``coupling_adjoint`` as f_q, the parameter row's tracking term from the
same density as f_q^* -- so the aligned (z_old := z) mismatch sits at
machine precision for every example, sigma = 0 or 1, in every scheme:
the tracking term enters the residual in the semi-implicit scheme and the
matrix in the implicit one, and the harness reads it from either.
"""

import pytest

from mras.adjoint import adjoint_mismatch
from mras.live import build

BUILDERS = {
    "darcy": lambda: build("darcy", h=0.15, dt=0.05, tend=0.5),
    "fisher": lambda: build("fisher", h=0.2, dt=0.05, tend=0.5, q_h=0.15),
    "burgers_1d": lambda: build("burgers_1d", h=0.35, dt=0.05, tend=0.5),
    "arrhenius": lambda: build("arrhenius", h=0.4, dt=0.05, tend=0.5),
    "arrhenius_linear": lambda: build("arrhenius_linear", h=0.4, dt=0.05,
                                      tend=0.5),
    "potential": lambda: build("potential", h=0.25, dt=0.05, tend=0.5),
    "allen_cahn": lambda: build("allen_cahn", h=0.25, dt=0.05, tend=0.5),
}


@pytest.mark.parametrize("scheme", ["implicit", "semi-implicit"])
@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_exact_adjoint_coupling_is_machine_precise(name, scheme):
    problem = BUILDERS[name]()
    problem.scheme = scheme
    result = problem.run_level(0.0, progress=False)
    report = adjoint_mismatch(problem, result.state)
    aligned = report["aligned"]
    assert not aligned["vacuous"], (name, aligned)
    assert aligned["max_ratio"] < 1e-10, (name, report)
    assert aligned["max_ratio_full"] < 1e-10, (name, report)
    # The untouched step form of the last assimilation step: the driver
    # stores z_old := z after every step, so the final forms the probe
    # reads are aligned by construction (stationary Darcy trivially so).
    assert report["lagged"]["max_ratio"] < 1e-10, (name, report)
