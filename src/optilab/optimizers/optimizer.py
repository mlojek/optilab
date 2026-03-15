"""
Base class for optimizers.
"""

from multiprocessing.pool import Pool
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
            name: Name of this optimizer.
            population_size: Size of the population.
            hyperparameters: Dictionary with the hyperparameters of the optimizer.
        """
        self.metadata = OptimizerMetadata(name, population_size, hyperparameters)

    # Stop checker methods
    @staticmethod
    def _stop_budget(
        log: PointList,
        population_size: int,
        call_budget: int,
    ) -> bool:
        """
        Check if the call budget will allow for another algorithm generation.

        Args:
            log: Log with all points evaluated so far.
            population_size: Number of points in a single generation.
            call_budget: Number of allowed point evaluations.

        Returns:
            If true, the call budget has been expended and another generation
                cannot be evaluated.
        """
        return len(log) + population_size > call_budget

    @staticmethod
    def _stop_target_found(
        log: PointList,
        target: float,
        tolerance: float,
    ) -> bool:
        """
        Check if the optimal function value has been found.

        Args:
            log: Log with all points evaluated so far.
            target: Global optimum value of the optimized function.
            tolerance: Allowed error value from global optimum value.

        Returns:
            If true, the global optimum has been found.
        """
        return log.best_y() < target + tolerance

    def _stop_external(
        self,
        log: PointList,
        population_size: int,
        call_budget: int,
        target: float,
        tolerance: float,
    ) -> bool:
        """
        Check if the external stop criteria has been met, i.e. if the budget has been expended
        or the the global optimum has been found.

        Args:
            log: Log with all points evaluated so far.
            population_size: Number of points in a single generation.
            call_budget: Number of allowed point evaluations.
            target: Global optimum value of the optimized function.
            tolerance: Allowed error value from global optimum value.

        Returns:
            If true, the external stop criteria has been met and the optimization
                should be stopped.
        """
        return self._stop_budget(
            log,
            population_size,
            call_budget,
        ) or self._stop_target_found(
            log,
            target,
            tolerance,
        )

    # Optimization methods
    def optimize(
        self,
        function: ObjectiveFunction,
        bounds: Bounds,
        call_budget: int,
        tolerance: float,
        target: float = 0.0,
    ) -> PointList:
        """
        Run a single optimization of provided objective function.

        Args:
            function: Objective function to optimize.
            bounds: Search space of the function.
            call_budget: Max number of calls to the objective function.
            tolerance: Tolerance of y value to count a solution as acceptable.
            target: Objective function value target, default 0.

        Returns:
            Results log from the optimization.
        """
        raise NotImplementedError

    def run_optimization(
        self,
        num_runs: int,
        function: ObjectiveFunction,
        bounds: Bounds,
        call_budget: int,
        tolerance: float,
        target: float = 0.0,
        *,
        num_processes: int = 1,
    ) -> OptimizationRun:
        """
        Optimize a provided objective function.

        Args:
            num_runs: Number of optimization runs to perform.
            function: Objective function to optimize.
            bounds: Search space of the function.
            call_budget: Max number of calls to the objective function.
            tolerance: Tolerance of y value to count a solution as acceptable.
            target: Objective function value target, default 0.
            num_processes(int): Number of concurrent processes to use to speed up the
                optimization. By default only one is used.

        Returns:
            Metadata of optimization run.
        """
        with Pool(num_processes) as pool:
            tasks = [
                pool.apply_async(
                    self.optimize, (function, bounds, call_budget, tolerance, target)
                )
                for _ in range(num_runs)
            ]

            logs = [
                task.get() for task in tqdm(tasks, desc="Optimizing...", unit="run")
            ]

        return OptimizationRun(
            model_metadata=self.metadata,
            function_metadata=function.metadata,
            bounds=bounds,
            tolerance=tolerance,
            logs=logs,
        )
