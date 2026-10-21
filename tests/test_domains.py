"""Boundary handling of the benchmark meshes.

A mass-matrix solve against the constant load reproduces the one field on
the free degrees of freedom and leaves constrained ones at zero, so
boundary point evaluations reveal which spaces carry constraints.
"""

import pytest
from ngsolve import BilinearForm, GridFunction, LinearForm, solvers, dx

from mras.domains import allen_cahn, darcy, fisher, potential


@pytest.mark.parametrize("make, maxh, ndof_u, ndof_q", [
    (darcy, 0.04, 6703, 1456),
    (potential, 0.1, 34279, 7552),
    (allen_cahn, 0.1, 8542, 1858),
])
def test_dof_anchors(make, maxh, ndof_u, ndof_q):
    dom = make(maxh=maxh)
    assert dom.Ufes.ndof == ndof_u
    assert dom.Qfes.ndof == ndof_q


def _mass_solve(fes):
    u, v = fes.TnT()
    a = BilinearForm(u * v * dx)
    a.Assemble()
    f = LinearForm(1.0 * v * dx)
    f.Assemble()
    gf = GridFunction(fes)
    solvers.BVP(bf=a, lf=f, gf=gf)
    return gf


def test_darcy_fully_dirichlet():
    gf = _mass_solve(darcy(maxh=0.2).Ufes)
    assert gf(0.0, 0.5) == pytest.approx(0.0, abs=1e-12)
    assert gf(0.5, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert gf(0.5, 0.5) == pytest.approx(1.0, abs=0.01)


def test_fisher_regex_quirk_leaves_space_free():
    # The fisher boundary labels never match the Dirichlet regex, so the
    # U_h space is fully free and constants survive a solve.
    gf = _mass_solve(fisher(maxh=0.2).Ufes)
    assert gf(-1.25, 0.0) == pytest.approx(1.0, abs=0.01)
    assert gf(0.0, 1.25) == pytest.approx(1.0, abs=0.01)


def test_allen_cahn_u_dirichlet_q_free():
    dom = allen_cahn(maxh=0.2)
    u = _mass_solve(dom.Ufes)
    assert u(2.0, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert u(0.0, 0.0) == pytest.approx(1.0, abs=0.01)
    q = _mass_solve(dom.Qfes)
    assert q(1.9, 0.0) == pytest.approx(1.0, abs=0.01)
