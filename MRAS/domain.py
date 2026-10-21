from ngsolve import Mesh, H1, L2
from netgen.occ import *
from numpy import pi

class domain:
    def __init__(self, maxh, order_u = 4, order_q = 0):

        self.maxh = maxh
        self.order_u = order_u
        self.order_q = order_q
        
        shape = Rectangle(1,1).Face()
        shape.edges.Min(X).name="left"
        shape.edges.Max(X).name="right"
        shape.edges.Min(Y).name="bottom"
        shape.edges.Max(Y).name="top"
        
        geom = OCCGeometry(shape, dim=2)
        
        self.mesh = Mesh(geom.GenerateMesh(maxh=self.maxh))

        # Set up of the FE Spaces for the two different update steps
        self.Ufes = H1(self.mesh, order = order_u, dirichlet="top|bottom|left|right")
        self.Qfes = L2(self.mesh, order = order_q, dirichlet="top|bottom|left|right")
        
        