from ngsolve import Mesh, H1, L2
from netgen.occ import *
from numpy import pi
from netgen.geom2d import SplineGeometry
class domain:
    def __init__(self, maxh, order = 3):

        self.maxh = maxh
        self.order = order
        
        geo = SplineGeometry()
        geo.AddCircle(c=(0,0),
              r=pi,
              bc="circle",
              leftdomain=1,
              rightdomain=0)        
        #geom = OCCGeometry(shape, dim=2)
        
        self.mesh = Mesh(geo.GenerateMesh(maxh=self.maxh))

        # Set up of the FE Spaces for the two different update steps
        self.Ufes = H1(self.mesh, order = order, dirichlet="circle")
        self.Qfes = L2(self.mesh, order = 0)
        
        