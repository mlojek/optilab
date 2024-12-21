"""
Surrogate objective function which approximates the function value
with polynomial regression with interactions optimized using least squares.
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from ...data_classes import FunctionMetadata, Point, PointList
from .surrogate_objective_function import SurrogateObjectiveFunction


class PolynomialRegression(SurrogateObjectiveFunction):
    """
    Surrogate objective function which approximates the function value
    with polynomial regression with interactions optimized using least squares.
    """

    def __init__(self, degree: int, train_set: PointList = None) -> None:
        """
        Class constructor.

        Args:
            degree (int): Degree of the polynomial used for approximation.
            train_set (PointList): Training data for the model.
        """
        self.preprocessor = PolynomialFeatures(degree=degree)
        self.degree = degree

        super().__init__(f"polynomial_regression_{degree}_degree", train_set)

    def get_metadata(self) -> FunctionMetadata:
        """
        Get the metadata describing the function.

        Returns:
            FunctionMetadata: The metadata of the function.
        """
        metadata = super().get_metadata()
        metadata.hyperparameters["degree"] = self.degree
        return metadata

    def train(self, train_set: PointList) -> None:
        """
        Train the Surrogate function with provided data

        Args:
            train_set (PointList): Train data for the model.
        """
        super().train(train_set)
        x, y = self.train_set.pairs()
        self.weights = np.linalg.lstsq(self.preprocessor.fit_transform(x), y)[0]

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
        super().__call__(point)
        return Point(
            x=point.x,
            y=sum(self.weights * self.preprocessor.fit_transform([point.x])[0]),
            is_evaluated=False,
        )
