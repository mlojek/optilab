"""
Component ObjectiveFunction. It's an object that describes an objective function to be optimized.
"""

from .ackley_function import AckleyFunction
from .cec2017_objective_function import CEC2017ObjectiveFunction
from .knn_surrogate_objective_function import KNNSurrogateObjectiveFunction
from .noisy_sphere_function import NoisySphereFunction
from .objective_function import ObjectiveFunction
from .polynomial_regression import PolynomialRegression
from .rastrigin_function import RastriginFunction
from .rosenbrock_function import RosenbrockFunction
from .schwefel_function import SchwefelFunction
from .sphere_function import SphereFunction
from .surrogate_objective_function import SurrogateObjectiveFunction
