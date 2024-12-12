"""
Objective function from CEC 2017 benchmark.
"""

# pylint: disable=too-few-public-methods

from typing import List

from cec2017.functions import all_functions

from ..objective_function import ObjectiveFunction


class CEC2017ObjectiveFunction(ObjectiveFunction):
    """
    Objective function from CEC 2017 benchmark.
    """

    def __init__(self, function_num: int, dim: int):
        """
        Class constructor.

        Args:
            function_num (int): The number of CEC2017 function, The range is 1 to 30.
            dim (int): Dimensionality of the function.
        """
        if not function_num in range(1, 31):
            raise ValueError(f"Invalid cec2017 function number {function_num}.")

        super().__init__(f"cec2017_f{function_num}", dim)

        self.function = all_functions[function_num - 1]
        self.minimum = function_num * 100

    def __call__(self, x: List[float]) -> float:
        """
        Evaluate a single point with the objective function.

        Args:
            x (List[float]): Point to evaluate.

        Raises:
            ValueError: If dimensionality of x doesn't match self.dim

        Returns:
            float: Value of the function in the provided point.
        """
        super().__call__(x)
        return float(self.function([x])[0]) - self.minimum
