import ngsolve
from netgen.gui import Snapshot
import ngsolve.internal as ngsint

# Resolution for 4K monitors
image_width = 3840
image_height = 2160

# Removes mesh grid, colorbar, logo, xyz-axis marker and sets to a much finer color scale than ngsolve default
ngsint.viewoptions.drawoutline = 0
ngsint.viewoptions.drawcolorbar = 0
ngsint.viewoptions.drawnetgenlogo = 0
ngsint.viewoptions.drawcoordinatecross = 0
ngsint.visoptions.numtexturecols = 256

mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))
V = H1(mesh, order=3)
gfu = GridFunction(V)
gfu.vec.FV().NumPy()[:] = np.load("gfu_over_time_noise0.05.npy")[0]

Draw(gfu); # Will also work with a CF, of course, but then needs mesh argument as usual. Note that this is not the webgui Draw and so has slightly different syntax.
filename = "MY_GFU"
sleep(1)
Snapshot(w = image_width, h = image_height, filename = filename + ".ppm")
sleep(1)
