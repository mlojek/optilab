"""
Bent Cigar objective function. 
"""

# pylint: disable=too-few-public-methods


from ...data_classes import Point
from ..objective_function import ObjectiveFunction


class BentCigarFunction(ObjectiveFunction):
    """
    Bent Cigar objective function.
    """

    def __init__(self, dim: int):
        """
        Class constructor.

        Args:
            dim (int): Dimensionality of the function.
        """
        super().__init__("bent_cigar", dim)

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
            y=point.x[0] ** 2 + sum(point.x[1:] ** 2) * (10**6),
            is_evaluated=True,
        )
