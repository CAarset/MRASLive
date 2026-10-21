from ngsolve import *

# Class solving the implicit equation
# d/dt u + Au = rhs
# Input:
# bilinearform of type ngsolve BilinearForm and linearform
# If None provided for bilinearform, must provide fes

class implicit:
    def __init__(self, dt, bilinearform = None, fes = None):
        self.dt = dt
        self.bilinearform = bilinearform
        assert bilinearform is not None or fes is not None, "Either provide an ngsolve BilinearForm or a fes!"
        if self.bilinearform is not None:
            self.fes = self.bilinearform.space
        else:
            self.fes = fes

        self.u, self.v = self.fes.TnT()

        self.implicitform = BilinearForm(self.fes)
        self.implicitform += self.u * self.v * dx
        self.implicitform.Assemble()

        if self.bilinearform is not None:
            self.implicitform.mat.AsVector().data += self.dt * self.bilinearform.mat.AsVector() # Nice workaround, copied from your code

    def solve_implicit(self, gf, rhs_v, rhs_gradv_factor =0, rhs_gradv = 0): #######later rewrite the code s t we pass linear form, that gets rebuild
        # Solves the implicit equation
        # Requires ngsolve-compatible rhs - we do a rhs for just the test function and one for bilinearforms with gradient + te corresponding factor
        # and a GridFunction u that can be updated

        full_rhs = LinearForm((gf + self.dt * rhs_v) * self.v * dx + self.dt * rhs_gradv_factor * Grad(rhs_gradv) * Grad(self.v) * dx).Assemble()
        gf.vec.data = self.implicitform.mat.Inverse(freedofs = self.fes.FreeDofs()) * full_rhs.vec