"""
Objective functions from CEC benchmarks.
"""

import opfunu

from ...data_classes import FunctionMetadata, Point
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
        super().__init__(f"cec{year}_f{function_num}", dim)

        self.function_num = function_num
        self.function = opfunu.get_functions_by_classname(f"F{function_num}{year}")[0](
            ndim=dim
        )

    def get_metadata(self) -> FunctionMetadata:
        """
        Get the metadata describing the function.

        Returns:
            FunctionMetadata: The metadata of the function.
        """
        metadata = super().get_metadata()
        metadata.hyperparameters["function_num"] = self.function_num
        return metadata

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
