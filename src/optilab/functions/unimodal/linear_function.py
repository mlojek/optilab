"""
Linear function. y is the sum of elements of x vector.
"""

import numpy as np

from ...data_classes import Point
from ..objective_function import ObjectiveFunction


class LinearFunction(ObjectiveFunction):
    """
    Linear function. y is the sum of elements of x vector.
    """

    def __init__(self, dim: int):
        """
        Class constructor.

        Args:
            dim: Dimensionality of the function.
        """
        super().__init__("linear", dim)

    def __call__(self, point: Point) -> Point:
        """
        Evaluate a single point with the objective function.

        Args:
            point: Point to evaluate.

        Raises:
            ValueError: If dimensionality of x doesn't match self.dim.

        Returns:
            Evaluated point.
        """
        super().__call__(point)
        assert point.x is not None
        return Point(
            x=point.x,
            y=float(np.sum(point.x)),
            is_evaluated=True,
        )
