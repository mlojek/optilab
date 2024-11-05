"""
Objective function from CEC 2017 benchmark.
"""

from typing import List, float

from cec2017.functions import all_functions

from .objective_function import ObjectiveFunction


class CEC2017ObjectiveFunction(ObjectiveFunction):
    """
    Objective function from CEC 2017 benchmark.
    """

    def __init__(self, function_num: int, dim: int):
        """
        Class constructor.

        :param function_num: the number of function from range [1, 30].
        :raises AssertionError: when the number of function is invalid.
        :param dim: dimensionality of the function.
        """
        assert function_num in range(1, 31), "Invalid cec2017 function number!"
        super().__init__(f"cec2017_f{function_num}", dim)
        self.function = all_functions[function_num - 1]
        self.minimum = function_num * 100

    def __call__(self, x: List[float]) -> float:
        """
        Evaluate a point.

        :param x: point to evaluate
        :raises AssertionError: when the point dimensionality doesn't match the function's dimensionality
        :return: function value in given point
        """
        assert len(x) == self.dim, f'Given point has invalid dimensions, got {len(x)}, expected {self.dim}'
        super().__call__()
        return self.function(x) - self.minimum
