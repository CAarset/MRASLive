"""Pixel-array field helpers and norms."""

import numpy as np
import pytest
from ngsolve import CoefficientFunction, GridFunction

from mras.fields import l2norm, real_root, to_gridfunc, to_gridfunc0


def test_to_gridfunc_constant(small_darcy):
    fes = small_darcy.Ufes
    gf = to_gridfunc(np.full((6, 6), 2.0), fes)
    assert l2norm(gf - CoefficientFunction(2.0), small_darcy.mesh) < 1e-12


def test_to_gridfunc_subbox_extrapolates(small_darcy):
    # VoxelCoefficient keeps the edge value outside the box, so a constant
    # array restricted to a sub-box is still constant on the whole mesh.
    fes = small_darcy.Ufes
    gf = to_gridfunc(np.full((6, 6), 3.0), fes, box=((0.25, 0.25), (0.75, 0.75)))
    assert l2norm(gf - CoefficientFunction(3.0), small_darcy.mesh) < 1e-12


def test_to_gridfunc0_boundary_ramp(small_darcy):
    # Tested on the L2 coefficient space, which is what the Darcy problem
    # feeds it into. Elements in the plateau average the profile exactly;
    # band elements average strictly between zero and one.
    fes = small_darcy.Qfes
    gf = to_gridfunc0(np.ones((8, 8)), fes, band=0.1)
    assert gf(0.5, 0.5) == pytest.approx(1.0, abs=1e-12)
    assert 0.0 < gf(0.05, 0.5) < 1.0
    assert 0.0 < gf(0.98, 0.5) < 1.0
    assert 0.0 < gf(0.5, 0.05) < 1.0
    assert 0.0 < gf(0.05, 0.05) < 1.0


def test_l2norm_matches_area_scaling(small_darcy):
    mesh = small_darcy.mesh
    assert l2norm(CoefficientFunction(1.0), mesh) == pytest.approx(1.0, rel=1e-10)
    assert l2norm(CoefficientFunction(3.0), mesh) == pytest.approx(3.0, rel=1e-10)


def test_real_root():
    assert real_root(-8.0, 3) == pytest.approx(-2.0)
    assert real_root(8.0, 3) == pytest.approx(2.0)
    out = real_root(np.array([-27.0, 64.0]), 3)
    assert np.allclose(out, [-3.0, 4.0])
