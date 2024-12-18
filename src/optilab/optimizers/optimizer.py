"""
Base class for optimizers.
"""

# pylint: disable=too-few-public-methods

from typing import Any, Dict

from ..data_classes import Bounds, OptimizationRun, OptimizerMetadata
from ..functions import ObjectiveFunction


class Optimizer:
    """
    Base class for optimizers.
    """

    def __init__(
        self, name: str, population_size: int, hyperparameters: Dict[str, Any]
    ) -> None:
        """
        Class constructor.
        """
        self.metadata = OptimizerMetadata(name, population_size, hyperparameters)

    def optimize(self, function: ObjectiveFunction, bounds: Bounds) -> OptimizationRun:
        """
        Optimize a certain objective function.
        """
        raise NotImplementedError
