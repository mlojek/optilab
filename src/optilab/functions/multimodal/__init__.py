"""
Multimodal objective functions. They have multiple local minima.
"""

from .ackley_function import AckleyFunction
from .rastrigin_function import RastriginFunction
from .rosenbrock_function import RosenbrockFunction

__all__ = [
    "AckleyFunction",
    "RastriginFunction",
    "RosenbrockFunction",
]
