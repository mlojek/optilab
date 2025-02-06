"""
Class that makes an objective function noisy using random normal distribution.
"""

import numpy as np

from ..data_classes import FunctionMetadata, Point
from .objective_function import ObjectiveFunction


class NoisyFunction(ObjectiveFunction):
    """
    Class that makes an objective function noisy using random normal distribution.
    """

    def __init__(self, function: ObjectiveFunction, noise: float) -> None:
        """
        Class constructor.

        Args:
            function (ObjectiveFunction): Objective function to noise.
            noise (float): Noise value of the function.
        """
        super().__init__(f"noisy_{function.name}_{noise}", function.dim)
        self.function = function
        self.noise = noise

    def get_metadata(self) -> FunctionMetadata:
        """
        Get the metadata describing the function.

        Returns:
            FunctionMetadata: The metadata of the function.
        """
        metadata = super().get_metadata()
        metadata.hyperparameters["noise"] = self.noise
        return metadata

    def __call__(self, point: Point) -> Point:
        """
        Evaluate a single point with the objective function.

        Args:
            point (Point): Point to be evaluated.

        Raises:
            ValueError: If dimensionality of x doesn't match the dimensionality of the function.

        Returns:
            Point: Evaluated point.
        """
        super().__call__(point)
        evaluated_point = self.function(point)
        evaluated_point.x *= 1 + self.noise * np.random.normal(0, 1)
        return evaluated_point
