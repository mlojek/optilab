"""
Component objective_functions.surrogate. It contains surrogate objective function,
which are regressors used to estimate objective function values.
"""

from .knn_surrogate_objective_function import KNNSurrogateObjectiveFunction
from .locally_weighted_polynomial_regression import LocallyWeightedPolynomialRegression
from .polynomial_regression import PolynomialRegression
from .surrogate_objective_function import SurrogateObjectiveFunction
