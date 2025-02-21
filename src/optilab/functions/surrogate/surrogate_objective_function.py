"""
Abstract base class for surrogate objective functions.
"""

from ...data_classes import Point, PointList
from ..objective_function import ObjectiveFunction


class SurrogateObjectiveFunction(ObjectiveFunction):
    """
    Abstract base class for surrogate objective functions.
    """

    def __init__(self, name: str, train_set: PointList = None) -> None:
        """
        Class constructor. The dimensionality is deduced from the training points.

        Args:
            name (str): Name of the surrogate function.
            train_set (PointList): Training data for the model.
        """
        self.is_ready = False
        super().__init__(name, 1)

        if train_set:
            self.train(train_set)

    def train(self, train_set: PointList) -> None:
        """
        Train the Surrogate function with provided data.

        Args:
            train_set (PointList): Training data for the model.

        Raises:
            ValueError: If not all points are evaluated.
        """
        if not all((train_point.is_evaluated for train_point in train_set)):
            raise ValueError("Not all points in the training set are evaluated!")

        self.is_ready = True

        dim_set = {point.dim() for point in train_set.points}
        if not len(dim_set) == 1:
            raise ValueError(
                "Provided train set has x-es with different dimensionalities."
            )
        self.dim = list(dim_set)[0]

        self.train_set = train_set

    def __call__(self, point: Point) -> Point:
        """
        Estimate the value of a single point with the surrogate function.

        Args:
            point (Point): Point to estimate.

        Raises:
            ValueError: If dimensionality of x doesn't match self.dim.

        Returns:
            Point: Estimated value of the function in the provided point.
        """
        if not self.is_ready:
            raise NotImplementedError("The surrogate function is not trained!")
        super().__call__(point)
