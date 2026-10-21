"""Additive measurement-noise models for the data feed."""

import numpy as np
from ngsolve import GridFunction

from .fields import l2norm, to_gridfunc


class NoiseModel:
    """Pixel-array noise with exact relative L2 amplitude.

    Each draw builds a ``peaks`` x ``peaks`` amplitude array, maps it onto the
    FE space over ``box``, and rescales it so the noisy datum z = gf + noise
    satisfies ||z - gf||_L2 = level * ||gf||_L2 exactly.

    Parameters
    ----------
    peaks, box :
        resolution and bounding box of the pixel array.
    rng : numpy.random.Generator or None
        source of amplitudes; ``None`` uses the global ``numpy.random`` state.
    distribution : {"uniform", "gaussian"}
        amplitude law before normalisation (uniform matches the historical
        paper runs; gaussian matches the paper's noise description).
    reseed_each_draw : bool
        re-prime the global generator before every draw, producing the same
        noise pattern at every time step (frozen-noise regime).
    """

    def __init__(self, peaks=20, box=((0.0, 0.0), (1.0, 1.0)), rng=None,
                 distribution="uniform", reseed_each_draw=False):
        self.peaks = peaks
        self.box = box
        self.rng = rng
        self.distribution = distribution
        self.reseed_each_draw = reseed_each_draw

    def amplitudes(self):
        if self.reseed_each_draw:
            np.random.seed(0)
            return 2 * np.random.rand(self.peaks, self.peaks) - 1
        if self.rng is not None:
            if self.distribution == "gaussian":
                return self.rng.standard_normal((self.peaks, self.peaks))
            return 2 * self.rng.random((self.peaks, self.peaks)) - 1
        if self.distribution == "gaussian":
            return np.random.randn(self.peaks, self.peaks)
        return 2 * np.random.rand(self.peaks, self.peaks) - 1

    def draw(self, gf, level, mesh):
        """Return a fresh GridFunction z = gf + level-scaled noise."""
        z = GridFunction(gf.space)
        z.vec.data = gf.vec.Copy()
        amps = self.amplitudes()
        if level == 0:
            return z
        unit = to_gridfunc(amps, gf.space, box=self.box)
        scaling = level / l2norm(unit, mesh)
        noise = to_gridfunc(scaling * amps, gf.space, box=self.box)
        z.vec.data += noise.vec.data
        return z
