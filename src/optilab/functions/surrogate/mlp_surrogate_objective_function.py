"""
Surrogate objective function using sklearn MLPRegressor.
"""

from typing import Tuple

import numpy as np
from sklearn.neural_network import MLPRegressor

from ...data_classes import Point, PointList
from .surrogate_objective_function import SurrogateObjectiveFunction


class MLPSurrogateObjectiveFunction(SurrogateObjectiveFunction):
    """
    Surrogate objective function using sklearn MLPRegressor.
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int] = (32,),
        train_set: PointList | None = None,
        *,
        activation: str = "relu",
        solver: str = "adam",
        learning_rate_init: float = 0.001,
        l2_alpha: float = 0.0001,
        max_iter: int = 200,
        early_stopping: bool = True,
        random_seed: int | None = None,
    ) -> None:
        """
        Class constructor.

        Args:
            hidden_layer_sizes: Shape of hidden layers.
            train_set: Training data for the model.
            activation: Activation function for hidden layers.
            solver: The solver for weight optimization.
            learning_rate_init: Initial learning rate used by optimizer.
            l2_alpha: L2 regularization weight.
            max_iter: Maximum number of optimization iterations.
            early_stopping: Whether to use validation-based early stopping.
            random_seed: Seed for reproducible initialization.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.l2_alpha = l2_alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_seed = random_seed

        self.model: MLPRegressor | None = None

        hidden_layers_repr = "_".join(str(size) for size in hidden_layer_sizes)

        super().__init__(
            f"MLP_{hidden_layers_repr}",
            train_set,
            {
                "hidden_layer_sizes": hidden_layer_sizes,
                "activation": activation,
                "solver": solver,
                "l2_alpha": l2_alpha,
                "learning_rate_init": learning_rate_init,
                "max_iter": max_iter,
                "early_stopping": early_stopping,
                "random_seed": random_seed,
            },
        )

    def train(self, train_set: PointList) -> None:
        """
        Train the MLP surrogate function with provided data.

        Args:
            train_set (PointList): Training data for the model.
        """
        super().train(train_set)

        x_train, y_train = self.train_set.pairs()

        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.l2_alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            random_state=self.random_seed,
        )

        # ignore warnings about overflows and zero divisions when covariance matrix
        # is ill-conditioned
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            self.model.fit(x_train, y_train)

    # pylint: disable=duplicate-code
    def __call__(self, point: Point) -> Point:
        """
        Estimate the function value at a given point using MLP regression.

        Args:
            point (Point): Point to estimate.

        Returns:
            Point: Estimated value of the function at the given point.
        """
        super().__call__(point)

        x_query = np.array([point.x], dtype=np.float64)

        # ignore warnings about overflows and zero divisions when covariance matrix
        # is ill-conditioned
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            y_pred = self.model.predict(x_query)[0]

        return Point(
            x=point.x,
            y=float(y_pred),
            is_evaluated=False,
        )
