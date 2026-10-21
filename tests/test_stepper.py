"""Backward-Euler stepper: linear solve is satisfied, forms are not mutated."""

import numpy as np
from ngsolve import (BilinearForm, CoefficientFunction, Grad, GridFunction,
                     H1, LinearForm, x as xc, dx)

from mras.fields import l2norm
from mras.stepper import ImplicitStepper


def _free_fes(mesh):
    """H1 space with no constraints, so every matrix row is an equation."""
    return H1(mesh, order=1)


def _heat_forms(fes):
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += Grad(u) * Grad(v) * dx + u * v * dx
    a.Assemble()
    return a


def _shifted_operator(fes, a, dt):
    u, v = fes.TnT()
    m = BilinearForm(fes)
    m += u * v * dx
    m.Assemble()
    m.mat.AsVector().data += dt * a.mat.AsVector()
    return m


def test_step_solves_shifted_system(small_darcy):
    fes = _free_fes(small_darcy.mesh)
    a = _heat_forms(fes)
    dt = 1e-2
    stepper = ImplicitStepper(dt, a)

    gf = GridFunction(fes)
    gf.Set(CoefficientFunction(1.0))
    _, v = fes.TnT()
    rhs = LinearForm((gf + dt * CoefficientFunction(0.5)) * v * dx).Assemble()
    stepper.step(gf, CoefficientFunction(0.5))

    shifted = _shifted_operator(fes, a, dt)
    residual = gf.vec.CreateVector()
    shifted.mat.Mult(gf.vec, residual)
    residual.data -= rhs.vec
    assert np.max(np.abs(np.array(residual))) < 1e-9


def test_step_does_not_mutate_form(small_darcy):
    fes = _free_fes(small_darcy.mesh)
    a = _heat_forms(fes)

    gf = GridFunction(fes)
    gf.Set(CoefficientFunction(1.0))
    ImplicitStepper(1e-2, a).step(gf, CoefficientFunction(0.0))

    # A second stepper built from the same form is still the same operator.
    gf2 = GridFunction(fes)
    gf2.Set(CoefficientFunction(1.0))
    ImplicitStepper(1e-2, a).step(gf2, CoefficientFunction(0.0))
    assert np.array_equal(np.array(gf.vec), np.array(gf2.vec))


def test_gradient_source_changes_result(small_darcy):
    fes = _free_fes(small_darcy.mesh)
    a = _heat_forms(fes)

    # A linear field plays the role of the convecting datum.
    lin = GridFunction(fes)
    lin.Set(CoefficientFunction(2.0) * xc)

    gf_plain = GridFunction(fes)
    gf_grad = GridFunction(fes)
    ImplicitStepper(1e-2, a).step(gf_plain, CoefficientFunction(0.0))
    ImplicitStepper(1e-2, a).step(gf_grad, CoefficientFunction(0.0),
                                  rhs_gradv_factor=1.0, rhs_gradv=lin)
    assert l2norm(gf_grad - gf_plain, fes.mesh) > 1e-6
