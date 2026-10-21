"""MRAS: parameter identification for PDEs by model reference adaptation.

The online scheme and the benchmark problems from "Data assimilation via
model reference adaptation for linear and nonlinear dynamical systems".
"""

from .domains import Domains, allen_cahn, darcy, fisher, potential
from .fields import l2norm, real_root, to_gridfunc, to_gridfunc0
from .noise import NoiseModel
from .results import LevelResult
from .scheme import Problem, State
from .stepper import ImplicitStepper

__all__ = [
    "Domains", "allen_cahn", "darcy", "fisher", "potential",
    "l2norm", "real_root", "to_gridfunc", "to_gridfunc0",
    "NoiseModel", "LevelResult", "Problem", "State", "ImplicitStepper",
]

__version__ = "0.1.0"
