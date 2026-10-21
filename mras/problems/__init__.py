"""The four paper benchmarks."""

from .allen_cahn import AllenCahnProblem
from .darcy import DarcyProblem
from .fisher import FisherProblem
from .potential import PotentialProblem

__all__ = ["AllenCahnProblem", "DarcyProblem", "FisherProblem",
           "PotentialProblem"]
