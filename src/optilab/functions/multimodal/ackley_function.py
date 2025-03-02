"""
The ackley objective function
"""

import numpy as np

from ...data_classes import Point
from ..objective_function import ObjectiveFunction


class AckleyFunction(ObjectiveFunction):
    """
    Ackley objective function.
    """

    def __init__(self, dim: int):
        """
        Class constructor.

        Args:
            dim (int): Dimensionality of the function.
        """
        super().__init__("ackley", dim)

    def __call__(self, point: Point) -> Point:
        """
        Evaluate a single point with the objective function.

        Args:
            point (Point): Point to evaluate.

        Raises:
            ValueError: If dimensionality of x doesn't match self.dim.

        Returns:
            Point: Evaluated point.
        """
        super().__call__(point)
        function_value = (
            20
            - 20
            * np.exp(-0.2 * np.sqrt(sum(x_i**2 for x_i in point.x) / self.metadata.dim))
            + np.e
            - np.exp(
                sum(np.cos(2 * np.pi * x_i) for x_i in point.x) / self.metadata.dim
            )
        )
        return Point(x=point.x, y=function_value, is_evaluated=True)
