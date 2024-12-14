"""
Base class representing a callable objective function.
"""

from ..data_classes import FunctionMetadata, Point


class ObjectiveFunction:
    """
    Base class representing a callable objective function.
    """

    def __init__(self, name: str, dim: int) -> None:
        """
        Class constructor.

        Args:
            name (str): Name of the objective function.
            dim (int): Dimensionality of the function.
        """
        self.name = name
        self.dim = dim
        self.num_calls = 0

    def get_metadata(self) -> FunctionMetadata:
        """
        Get the metadata describing the function.

        Returns:
            FunctionMetadata: The metadata of the function.
        """
        return FunctionMetadata(name=self.name, dim=self.dim, hyperparameters={})

    def __call__(self, point: Point) -> Point:
        """
        Evaluate a single point with the objective function.

        Args:
            point (Point): Point to evaluate.

        Raises:
            ValueError: If dimensionality of x doesn't match self.dim

        Returns:
            Point: Evaluated point.
        """
        if not len(point.x) == self.dim:
            raise ValueError(
                f"The dimensionality of the provided point is not matching the dimensionality"
                f"of the function. Expected {self.dim}, got {len(point.x)}"
            )
        self.num_calls += 1
