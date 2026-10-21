from ngsolve import Mesh, H1, L2
from netgen.occ import *
from numpy import pi

class domain:
    def __init__(self, maxh, order = 4):

        self.maxh = maxh
        self.order = order
        
        shape = Rectangle(2*pi,2*pi).Face()
        shape.edges.Min(X).name="left"
        shape.edges.Max(X).name="right"
        shape.edges.Min(Y).name="bottom"
        shape.edges.Max(Y).name="top"
        
        geom = OCCGeometry(shape, dim=2)
        
        self.mesh = Mesh(geom.GenerateMesh(maxh=self.maxh))

        # Set up of the FE Spaces for the two different update steps
        self.Ufes = H1(self.mesh, order = order, dirichlet="top|bottom|left|right")
        self.Qfes = H1(self.mesh, order = order, dirichlet="top|bottom|left|right")
        
        