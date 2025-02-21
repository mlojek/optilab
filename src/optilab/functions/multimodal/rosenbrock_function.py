"""
The rosenbrock objective function
"""

from ...data_classes import Point
from ..objective_function import ObjectiveFunction


class RosenbrockFunction(ObjectiveFunction):
    """
    Rosenbrock objective function.
    """

    def __init__(self, dim: int):
        """
        Class constructor.

        Args:
            dim (int): Dimensionality of the function.
        """
        super().__init__("rosenbrock", dim)

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

        function_value = sum(
            100 * (x_i**2 - x_i_next) ** 2 + (x_i - 1) ** 2
            for x_i, x_i_next in zip(point.x, point.x[1:])
        )
        return Point(x=point.x, y=function_value, is_evaluated=True)
