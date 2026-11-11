"""MRAS: parameter identification for PDEs by model reference adaptation.

The online scheme and the benchmark problems from "Data assimilation via
model reference adaptation for linear and nonlinear dynamical systems".
"""

# torch before NGSolve: NGSolve's bundled OpenBLAS exports MKL's batched GEMM
# names, and a torch loaded after it binds its batched CPU products to them
# and crashes (ngsrng's ImportOrderWarning).  The noise model's ngsrng
# brings torch; loading it first keeps every later import of NGSolve safe.
import torch  # noqa: F401  (import order, see above)

from .checkpoint import load, resume
from .domains import (Domains, allen_cahn, burgers_channel, burgers_pipe,
                      darcy, fisher, pipe_wake_field, pixel_square, potential)
from .fields import l2norm, pixel_field, to_gridfunc
from .noise import GaussianFEMNoise, seeded_rng
from .results import LevelResult
from .movie import render_movie
from .scheme import SCHEMES, MRASProblem, State, TheorySpecError
from .stepper import ImplicitStepper

__all__ = [
    "Domains", "allen_cahn", "burgers_channel", "burgers_pipe", "darcy",
    "fisher", "pixel_square", "potential", "pipe_wake_field", "l2norm", "pixel_field", "to_gridfunc",
    "GaussianFEMNoise", "seeded_rng", "LevelResult", "MRASProblem", "State",
    "SCHEMES", "TheorySpecError", "ImplicitStepper", "load", "resume",
    "render_movie",
]

__version__ = "0.1.0"
