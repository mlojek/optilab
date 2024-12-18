"""
Objective function from CEC 2017 benchmark.
"""

from cec2017.functions import all_functions

from ...data_classes import FunctionMetadata, Point
from ..objective_function import ObjectiveFunction


class CEC2017ObjectiveFunction(ObjectiveFunction):
    """
    Objective function from CEC 2017 benchmark.
    """

    def __init__(self, function_num: int, dim: int) -> None:
        """
        Class constructor.

        Args:
            function_num (int): The number of CEC2017 function, The range is 1 to 30.
            dim (int): Dimensionality of the function.
        """
        if not function_num in range(1, 31):
            raise ValueError(f"Invalid cec2017 function number {function_num}.")

        super().__init__(f"cec2017_f{function_num}", dim)

        self.function_num = function_num
        self.function = all_functions[function_num - 1]
        self.minimum = function_num * 100

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
            y=float(self.function([point.x])[0]) - self.minimum,
            is_evaluated=True,
        )
