"""
Base class for optimizers.
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments


from typing import Any, Dict

from tqdm import tqdm

from ..data_classes import Bounds, OptimizationRun, OptimizerMetadata, PointList
from ..functions import ObjectiveFunction


class Optimizer:
    """
    Base class for optimizers.
    """

    def __init__(
        self, name: str, population_size: int, hyperparameters: Dict[str, Any]
    ) -> None:
        """
        Class constructor.

        Args:
            name (str): Name of this optimizer.
            population_size (int): Size of the population.
            hyperparameters (Dict[str, Any]): Dictionary with the metadata of the optimizer.
        """
        self.metadata = OptimizerMetadata(name, population_size, hyperparameters)

    def optimize(
        self,
        function: ObjectiveFunction,
        bounds: Bounds,
        call_budget: int,
        target: float = 0.0,
        tolerance: float = 1e-8,
    ) -> PointList:
        """
        Run a single optimization of provided objective function.

        Args:
            function (ObjectiveFunction): Objective function to optimize.
            bounds (Bounds): Search space of the function.
            call_budget (int): Max number of calls to the objective function.
            target (float): Objective function value target, default 0.
            tolerance (float): Tolerance of y value to count a solution as acceptable.

        Returns:
            PointList: Results log from the optimization.
        """
        raise NotImplementedError

    def run_optimization(
        self,
        num_runs: int,
        function: ObjectiveFunction,
        bounds: Bounds,
        call_budget: int,
        target: float = 0.0,
        tolerance: float = 1e-8,
    ) -> OptimizationRun:
        """
        Optimize a provided objective function.

        Args:
            num_runs (int): Number of optimization runs to perform.
            function (ObjectiveFunction): Objective function to optimize.
            bounds (Bounds): Search space of the function.
            call_budget (int): Max number of calls to the objective function.
            target (float): Objective function value target, default 0.
            tolerance (float): Tolerance of y value to count a solution as acceptable.

        Returns:
            OptimizationRun: Metadata of optimization run.
        """
        logs = [
            self.optimize(function, bounds, call_budget, target, tolerance)
            for _ in tqdm(range(num_runs), desc="Optimizing...", unit="run")
        ]

        return OptimizationRun(
            model_metadata=self.metadata,
            function_metadata=function.get_metadata(),
            bounds=bounds,
            tolerance=tolerance,
            logs=logs,
        )
