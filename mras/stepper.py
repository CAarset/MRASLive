"""Backward-Euler stepper for the ground-truth data generator."""

from ngsolve import BilinearForm, Grad, LinearForm, dx


class ImplicitStepper:
    """Advance (M + dt*K) u^{n+1} = M u^n + dt*F, one solve per call.

    ``a`` is the assembled stiffness form K; build a fresh stepper whenever the
    form changes (e.g. a lagged nonlinear coefficient is re-assembled).
    """

    def __init__(self, dt, a):
        self.dt = dt
        self.fes = a.space
        u, v = self.fes.TnT()
        self.u, self.v = u, v
        mass = BilinearForm(self.fes)
        mass += u * v * dx
        mass.Assemble()
        mass.mat.AsVector().data += dt * a.mat.AsVector()
        self.inv = mass.mat.Inverse(self.fes.FreeDofs())

    def step(self, gf, rhs_v, rhs_gradv_factor=0, rhs_gradv=None):
        """Advance ``gf`` in place.

        ``rhs_v`` is the source coefficient (tested against v); optionally add
        the gradient source ``dt * rhs_gradv_factor * grad(rhs_gradv).grad(v)``.
        """
        expr = (gf + self.dt * rhs_v) * self.v * dx
        if rhs_gradv is not None:
            expr += self.dt * rhs_gradv_factor * Grad(rhs_gradv) * Grad(self.v) * dx
        rhs = LinearForm(expr).Assemble()
        gf.vec.data = self.inv * rhs.vec
