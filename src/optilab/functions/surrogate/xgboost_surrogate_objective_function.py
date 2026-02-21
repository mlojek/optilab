"""
Surrogate objective function using XGBoost for gradient-boosted tree regression.
"""

import os

# Prevent OpenMP segfault when both FAISS and XGBoost are loaded in the same process
# (known conflict on macOS with duplicate libomp).
# pylint: disable=C0413
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import xgboost as xgb

from ...data_classes import Point, PointList
from .surrogate_objective_function import SurrogateObjectiveFunction


class XGBoostSurrogateObjectiveFunction(SurrogateObjectiveFunction):
    """
    Surrogate objective function using XGBoost for gradient-boosted tree regression.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        train_set: PointList = None,
    ) -> None:
        """
        Class constructor.

        Args:
            n_estimators (int): Number of boosting rounds.
            max_depth (int): Maximum depth of each tree.
            learning_rate (float): Step size shrinkage used to prevent overfitting.
            train_set (PointList): Training data for the model.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None

        super().__init__(
            f"XGBoost_n{n_estimators}_d{max_depth}",
            train_set,
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
            },
        )

    def train(self, train_set: PointList) -> None:
        """
        Train the XGBoost surrogate function with provided data.

        Args:
            train_set (PointList): Training data for the model.
        """
        super().train(train_set)

        x_train, y_train = self.train_set.pairs()
        x_train = np.array(x_train, dtype=np.float64)
        y_train = np.array(y_train, dtype=np.float64)

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            verbosity=0,
        )
        self.model.fit(x_train, y_train)

    def __call__(self, point: Point) -> Point:
        """
        Estimate the function value at a given point using XGBoost regression.

        Args:
            point (Point): Point to estimate.

        Returns:
            Point: Estimated value of the function at the given point.
        """
        super().__call__(point)

        x_query = np.array([point.x], dtype=np.float64)
        y_pred = self.model.predict(x_query)[0]

        return Point(
            x=point.x,
            y=float(y_pred),
            is_evaluated=False,
        )
