"""
Sphere function. y is the sum of squares of elements of x vector.
"""

from ...data_classes import Point
from ..objective_function import ObjectiveFunction


class SphereFunction(ObjectiveFunction):
    """
    Sphere function. y is the sum of squares of elements of x vector.
    """

    def __init__(self, dim: int):
        """
        Class constructor.

        Args:
            dim (int): Dimensionality of the function.
        """
        super().__init__("sphere", dim)

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
        return Point(
            x=point.x,
            y=sum(point.x**2),
            is_evaluated=True,
        )
