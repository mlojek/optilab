"""
Surrogate objective functions, regressors used to estimate objective function values.
"""

from .knn_surrogate_objective_function import KNNSurrogateObjectiveFunction
from .locally_weighted_polynomial_regression import LocallyWeightedPolynomialRegression
from .mlp_surrogate_objective_function import MLPSurrogateObjectiveFunction
from .normalized_mlp_surrogate_objective_function import (
    NormalizedMLPSurrogateObjectiveFunction,
)
from .polynomial_regression import PolynomialRegression
from .surrogate_objective_function import SurrogateObjectiveFunction
from .xgboost_surrogate_objective_function import XGBoostSurrogateObjectiveFunction

__all__ = [
    "KNNSurrogateObjectiveFunction",
    "LocallyWeightedPolynomialRegression",
    "MLPSurrogateObjectiveFunction",
    "NormalizedMLPSurrogateObjectiveFunction",
    "PolynomialRegression",
    "SurrogateObjectiveFunction",
    "XGBoostSurrogateObjectiveFunction",
]
