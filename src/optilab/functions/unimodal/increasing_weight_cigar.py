"""
Increasing Weight Cigar objective function.
"""

from ...data_classes import Point
from ..objective_function import ObjectiveFunction


class IncreasingWeightCigar(ObjectiveFunction):
    """
    Increasing Weight Cigar objective function.
    """

    def __init__(self, dim: int):
        """
        Class constructor.

        Args:
            dim (int): Dimensionality of the function.
        """
        super().__init__("increasing_weight_cigar", dim)

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
            y=sum(10**i * x_i**2 for i, x_i in enumerate(point.x)),
            is_evaluated=True,
        )
