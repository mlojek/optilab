"""
Base class for optimizers.
"""

from ..data_classes import Bounds, OptimizationRun, OptimizerMetadata
from ..functions import ObjectiveFunction


class Optimizer:
    """
    Base class for optimizers.
    """

    def __init__(self):
        """
        Class constructor.
        """

    def get_metadata(self) -> OptimizerMetadata:
        """
        Get metadata describing the optimizer.
        """

    def optimize(self, function: ObjectiveFunction, bounds: Bounds) -> OptimizationRun:
        """
        Optimize a certain objective function.
        """
