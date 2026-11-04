"""Boundary handling of the benchmark meshes.

A mass-matrix solve against the constant load reproduces the one field on
the free degrees of freedom and leaves constrained ones at zero, so
boundary point evaluations reveal which spaces carry constraints.
"""

import math

import pytest
from ngsolve import BilinearForm, GridFunction, LinearForm, solvers, dx

from mras.domains import allen_cahn, darcy, fisher, potential


@pytest.mark.parametrize("make, maxh, kwargs, ndof_u, ndof_q", [
    (darcy, 0.04, {}, 6703, 1456),
    (potential, 0.1, {}, 34279, 7552),
    (allen_cahn, 0.1, {"fitted": True}, 8551, 1860),  # follows q's interfaces
])
def test_dof_anchors(make, maxh, kwargs, ndof_u, ndof_q):
    dom = make(maxh=maxh, **kwargs)
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


def test_darcy_u_dirichlet_q_free():
    dom = darcy(maxh=0.2)
    u = _mass_solve(dom.Ufes)
    assert u(0.0, 0.5) == pytest.approx(0.0, abs=1e-12)
    assert u(0.5, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert u(0.5, 0.5) == pytest.approx(1.0, abs=0.01)
    q = _mass_solve(dom.Qfes)
    assert q(0.1, 0.1) == pytest.approx(1.0, abs=0.01)


def test_fisher_dirichlet_walls():
    gf = _mass_solve(fisher(maxh=0.2).Ufes)
    assert gf(-1.25, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert gf(0.0, 1.25) == pytest.approx(0.0, abs=1e-12)
    assert gf(0.0, 0.0) == pytest.approx(1.0, abs=0.01)


def test_potential_dirichlet_circle():
    # The state is prescribed on the circle (paper (26)): U_h is constrained
    # there, the data carry the (nonzero) boundary values.
    gf = _mass_solve(potential(maxh=0.3).Ufes)
    assert gf(0.0, math.pi) == pytest.approx(0.0, abs=1e-12)
    assert gf(0.0, 0.0) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("fitted", [False, True])
def test_allen_cahn_dirichlet_walls(fitted):
    dom = allen_cahn(maxh=0.2, fitted=fitted)
    u = _mass_solve(dom.Ufes)
    assert u(2.0, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert u(0.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert u(0.0, 0.0) == pytest.approx(1.0, abs=0.01)
    q = _mass_solve(dom.Qfes)
    assert q(1.9, 0.0) == pytest.approx(1.0, abs=0.01)


def test_q_meshes_follow_the_jumps_and_the_others_do_not():
    # The true parameter lives on a mesh that follows its jumps, exact
    # there; the estimator's and the data's meshes do not see them, and
    # evaluate it across their elements by point search.
    from mras.domains import burgers_channel
    from mras.fields import subdomain_field
    from ngsolve import Integrate
    for make, kwargs, pieces in (
            (fisher, {"maxh": 0.15}, {"outside", "ring", "core"}),
            (allen_cahn, {"maxh": 0.2}, {"background", "upper", "lower"}),
            (burgers_channel, {"maxh": 0.15},
             {"background", "star", "heart", "diamond"})):
        fitted = make(fitted=True, **kwargs).mesh
        plain = make(**kwargs).mesh
        assert set(fitted.GetMaterials()) == pieces
        assert len(set(plain.GetMaterials())) == 1
        assert "interface" not in plain.GetBoundaries()
        q = subdomain_field(fitted, {name: k for k, name
                                     in enumerate(sorted(pieces))})
        # the same field, integrated on either mesh: exact on its own,
        # a quadrature of its jumps on the other
        assert Integrate(q, plain, order=10) == pytest.approx(
            Integrate(q, fitted), rel=5e-3)
