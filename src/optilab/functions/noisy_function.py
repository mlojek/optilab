"""
Class that makes an objective function noisy using random normal distribution.
"""

import numpy as np

from ..data_classes import Point
from .objective_function import ObjectiveFunction


class NoisyFunction(ObjectiveFunction):
    """
    Class that makes an objective function noisy using random normal distribution.
    """

    def __init__(
        self,
        function: ObjectiveFunction,
        noise: float,
    ) -> None:
        """
        Class constructor.

        Args:
            function: Objective function to noise.
            noise: Noise value of the function.
        """
        super().__init__(
            f"noisy_{function.metadata.name}_{noise}",
            function.metadata.dim,
            {"noise": noise},
        )
        self.function = function

    def __call__(self, point: Point) -> Point:
        """
        Evaluate a single point with the objective function.

        Args:
            point: Point to be evaluated.

        Raises:
            ValueError: If dimensionality of x doesn't match the dimensionality of the function.

        Returns:
            Evaluated point.
        """
        super().__call__(point)
        evaluated_point = self.function(point)
        evaluated_point.x *= 1 + self.metadata.hyperparameters[
            "noise"
        ] * np.random.normal(0, 1)
        return evaluated_point
