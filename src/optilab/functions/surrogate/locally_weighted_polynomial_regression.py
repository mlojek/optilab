"""
Surrogate function which estimates the objective function with polynomial regression.
Points are weighted based on mahalanobis distance from query points.
"""

from typing import Callable, List, Tuple

import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import PolynomialFeatures

from .surrogate_objective_function import SurrogateObjectiveFunction

# pylint: disable=too-many-arguments,too-many-positional-arguments


def biquadratic_kernel_function(x: float) -> float:
    """
    Biquadratic weighting function.

    Args:
        x (float): Distance between points.

    Returns:
        float: Weight value.
    """
    if np.abs(x) >= 1:
        return 0

    return (1 - x**2) ** 2


class LocallyWeightedPolynomialRegression(SurrogateObjectiveFunction):
    """
    Surrogate function which estimates the objective function with polynomial regression.
    Points are weighted based on mahalanobis distance from query points.
    """

    def __init__(
        self,
        degree: int,
        num_neighbors: float,
        train_set: List[Tuple[List[float], float]] = None,
        covariance_matrix: List[List[float]] = None,
        kernel_function: Callable[[float], float] = biquadratic_kernel_function,
    ) -> None:
        """
        Class constructor.

        Args:
            degree (int): Degree of the polynomial used to approximate function.
            num_neighbors (float): Number of closest points to use in function approximation.
            train_set (List[Tuple[List[float], float]]): Training set for the regressor, optional.
            covariance_matrix (List[List[float]]): Covariance class used in mahalanobis distance,
                optional. When no such matrix is provided an identity matrix is used.
            kernel_function (Callable[[float], float]): Function used to assign weights to points.
        """
        self.is_ready = False
        super().__init__(
            f"locally_weighted_polynomial_regression_{degree}_degree", train_set
        )

        if train_set:
            self.train(train_set)

        self.num_neighbours = num_neighbors

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
        Setter for the covariance matrix.

        Args:
            new_covariance_matrix (List[List[float]]): New covariance matrix to use for mahalanobis
                distance.
        """
        self.reversed_covariance_matrix = np.linalg.inv(np.array(new_covariance_matrix))

    def __call__(self, x: List[float]) -> float:
        """
        Estimate the value of a single point with the surrogate function. Since the surrogate model
        is built for each point independently, this is where the regressor is trained.

        Args:
            x (List[float]): Point to estimate.

        Raises:
            ValueError: If dimensionality of x doesn't match self.dim.

        Return:
            float: Estimated value of the function in the provided point.
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
