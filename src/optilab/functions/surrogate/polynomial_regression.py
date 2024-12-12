"""
Surrogate objective function which approximates the function value
with polynomial regression with interactions optimized using least squares.
"""

from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from .surrogate_objective_function import SurrogateObjectiveFunction


class PolynomialRegression(SurrogateObjectiveFunction):
    """
    Surrogate objective function which approximates the function value
    with polynomial regression with interactions optimized using least squares.
    """

    def __init__(
        self, degree: int, train_set: List[Tuple[List[float], float]] = None
    ) -> None:
        """
        Class constructor.

        Args:
            degree (int): Degree of the polynomial used for approximation.
            train_set (List[Tuple[List[float], float]]): Training data for the model.
        """
        self.is_ready = False
        self.preprocessor = PolynomialFeatures(degree=degree)

        super().__init__(f"polynomial_regression_{degree}_degree", train_set)

    def train(self, train_set: List[Tuple[List[float], float]]) -> None:
        """
        Train the Surrogate function with provided data

        Args:
            train_set (List[Tuple[List[float], float]]): Train data for the model.
        """
        super().train(train_set)
        x, y = zip(*train_set)
        self.weights = np.linalg.lstsq(self.preprocessor.fit_transform(x), y)[0]

    def __call__(self, x: List[float]) -> float:
        """
        Estimate the value of a single point with the surrogate function.

        Args:
            x (List[float]): Point to estimate.

        Raises:
            ValueError: If dimensionality of x doesn't match self.dim.

        Return:
            float: Estimated value of the function in the provided point.
        """
        super().__call__(x)
        return sum(self.weights * self.preprocessor.fit_transform([x])[0])
