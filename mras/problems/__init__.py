"""The problem library: paper benchmarks and the Arrhenius pair."""

from .allen_cahn import AllenCahnProblem
from .arrhenius import ArrheniusLinearProblem, ArrheniusProblem
from .darcy import DarcyProblem
from .fisher import FisherProblem
from .potential import PotentialProblem

__all__ = ["AllenCahnProblem", "ArrheniusLinearProblem", "ArrheniusProblem",
           "DarcyProblem", "FisherProblem", "PotentialProblem"]
