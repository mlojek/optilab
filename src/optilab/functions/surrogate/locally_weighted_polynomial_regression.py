"""
TODO
"""

from typing import Callable, List, Tuple

import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import PolynomialFeatures

from .surrogate_objective_function import SurrogateObjectiveFunction

# pylint: disable=too-many-arguments,too-many-positional-arguments


def biquadratic_kernel_function(x: float) -> float:
    """
    TODO
    """
    if np.abs(x) >= 1:
        return 0
    return (1 - x**2) ** 2


class LocallyWeightedPolynomialRegression(SurrogateObjectiveFunction):
    """
    # TODO
    """

    def __init__(
        self,
        degree: int,
        num_neighbours: float,
        train_set: List[Tuple[List[float], float]] = None,
        covariance_matrix: List[List[float]] = None,
        kernel_function: Callable[[float], float] = biquadratic_kernel_function,
    ) -> None:
        """
        # TODO
        """
        self.is_ready = False
        super().__init__(
            f"locally_weighted_polynomial_regression_{degree}_degree", train_set
        )

        if train_set:
            self.train(train_set)

        self.num_neighbours = num_neighbours

        if covariance_matrix:
            self.set_covariance_matrix(covariance_matrix)
        else:
            self.set_covariance_matrix(np.eye(self.dim))

        self.kernel_function = kernel_function
        self.degree = degree
        self.preprocessor = PolynomialFeatures(degree=degree)
        self.weights = None

    def set_covariance_matrix(self, new_covariance_matrix: List[List[float]]) -> None:
        """
        TODO
        """
        self.reversed_covariance_matrix = np.linalg.inv(np.array(new_covariance_matrix))

    def __call__(self, x: List[float]) -> float:
        """
        TODO
        """
        super().__call__(x)

        distance_points = [
            (
                mahalanobis(x_t, x, self.reversed_covariance_matrix),
                np.array(x_t),
                y_t,
            )
            for x_t, y_t in self.train_set
        ]

        distance_points.sort(key=lambda i: i[0])

        knn_points = distance_points[: self.num_neighbours]

        bandwidth = knn_points[-1][0]

        weights = [
            (np.sqrt(self.kernel_function(d / bandwidth)), x_i, y_i)
            for d, x_i, y_i in knn_points
        ]

        weighted_x, weighted_y = zip(
            *[
                (
                    w * np.array(self.preprocessor.fit_transform([x_i])[0]),
                    w * np.array(y_i),
                )
                for w, x_i, y_i in weights
            ]
        )

        self.weights = np.linalg.lstsq(weighted_x, weighted_y)[0]

        return sum(self.weights * self.preprocessor.fit_transform([x])[0])
