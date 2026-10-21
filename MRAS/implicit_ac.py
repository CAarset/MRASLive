from ngsolve import *

# Class solving the implicit equation
# d/dt u + Au = rhs
# Input:
# bilinearform of type ngsolve BilinearForm, acting as A
# If None provided, must provide fes

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
            self.implicitform.mat.AsVector().data += dt * self.bilinearform.mat.AsVector() # Nice workaround, copied from your code

    def solve_implicit(self, gfu, rhs):
        # Solves the implicit equation
        # Requires ngsolve-compatible rhs
        # and a GridFunction u that can be updated

        implicit_rhs = gfu + self.dt * rhs
        
        gfu.vec.data = self.implicitform.mat.Inverse(freedofs = self.fes.FreeDofs()) * LinearForm(implicit_rhs * self.v * dx).Assemble().vec