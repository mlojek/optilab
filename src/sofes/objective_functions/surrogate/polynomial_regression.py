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
        Class constructor

        :param degree: the degree of the polynomial used for approximation.
        :param_train_set: training data for the model
        """
        self.is_ready = False
        self.preprocessor = PolynomialFeatures(degree=degree)

        super().__init__(f"polynomial_regression_of_{degree}_degree", train_set)

    def train(self, train_set: List[Tuple[List[float], float]]) -> None:
        """
        Train the Surrogate function with provided data

        :param train_set: train data expressed as list of tuples of x, y values
        """
        super().train(train_set)
        x, y = zip(*train_set)
        self.weights = np.linalg.lstsq(self.preprocessor.fit_transform(x), y)[0]

    def __call__(self, x: List[float]) -> float:
        """
        Predict the value of x with the surrogate function.

        :param x: point to predict the function value of
        :return: predicted function value
        """
        super().__call__(x)
        print(self.weights)
        print(self.preprocessor.fit_transform([x])[0])
        return sum(self.weights * self.preprocessor.fit_transform([x])[0])
