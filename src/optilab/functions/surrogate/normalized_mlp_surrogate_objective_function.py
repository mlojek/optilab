"""
Surrogate objective function using sklearn MLPRegressor with z-score normalization.
"""

import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
import numpy as np

from ...data_classes import Point, PointList
from .mlp_surrogate_objective_function import MLPSurrogateObjectiveFunction


class NormalizedMLPSurrogateObjectiveFunction(MLPSurrogateObjectiveFunction):
    """
    MLP surrogate with z-score normalization of both X and y before fitting.

    The base class passes raw values to sklearn with no normalization.
    X inputs and y values may span many orders of magnitude, causing large
    MSE gradients and convergence failure. Standardizing both X and y fixes this.
    Spearman (rank-based) metrics are invariant to the monotone y-transform.
    """

    def train(self, train_set: PointList) -> None:
        """
        Fit scalers on train_set, then train the MLP on normalized data.

        Args:
            train_set: Training data for the model.
        """
        self._x_scaler = StandardScaler()
        self._y_scaler = StandardScaler()

        xs = np.array([p.x for p in train_set.points])
        ys = np.array([[p.y] for p in train_set.points])

        xs_scaled = self._x_scaler.fit_transform(xs)
        ys_scaled = self._y_scaler.fit_transform(ys).ravel()

        scaled = PointList(
            points=[
                Point(x=x, y=float(y), is_evaluated=True)
                for x, y in zip(xs_scaled, ys_scaled)
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            super().train(scaled)

    def __call__(self, point: Point) -> Point:
        """
        Scale input, predict with MLP, then inverse-transform the output.

        Args:
            point: Point to estimate.

        Returns:
            Estimated value of the function at the given point.
        """
        assert point.x is not None

        x_scaled = self._x_scaler.transform([point.x])[0]
        result = super().__call__(
            Point(x=x_scaled, y=point.y, is_evaluated=point.is_evaluated)
        )
        y_unscaled = float(self._y_scaler.inverse_transform([[result.y]])[0][0])

        return Point(x=point.x, y=y_unscaled, is_evaluated=False)
