"""
Objective functions from CEC benchmarks.
"""

import opfunu

from ...data_classes import Point
from ..objective_function import ObjectiveFunction


class CECObjectiveFunction(ObjectiveFunction):
    """
    Objective functions from CEC benchmarks.
    """

    def __init__(self, year: int, function_num: int, dim: int) -> None:
        """
        Class constructor.

        Args:
            year (int): Year of the CEC competition.
            function_num (int): The number of benchmark function.
            dim (int): Dimensionality of the function.
        """
        super().__init__(
            f"cec{year}_f{function_num:02}", dim, {"function_num": function_num}
        )

        self.function = opfunu.get_functions_by_classname(f"F{function_num}{year}")[0](
            ndim=dim, f_bias=0
        )

    def __call__(self, point: Point) -> Point:
        """
        Evaluate a single point with the objective function.

        Args:
            point (Point): Point to evaluate.

        Raises:
            ValueError: If dimensionality of x doesn't match self.dim.

        Returns:
            Point: Evaluated point.
        """
        super().__call__(point)
        return Point(
            x=point.x,
            y=self.function.evaluate(point.x) - self.function.f_global,
            is_evaluated=True,
        )
