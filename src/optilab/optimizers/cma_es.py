"""
CMA-ES optimizer: Covariance Matrix Adaptation Evolution Strategy.
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments

import cma

from ..data_classes import Bounds, PointList
from ..functions import ObjectiveFunction
from .optimizer import Optimizer


class CmaEs(Optimizer):
    """
    CMA-ES optimizer: Covariance Matrix Adaptation Evolution Strategy.
    """

    def __init__(self, population_size: int, sigma0: float):
        """
        Class constructor.

        Args:
            population_size (int): Size of the population.
            sigma0 (float): Starting value of the sigma,
        """
        super().__init__("cma-es", population_size, {"sigma0": sigma0})

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
            function (ObjectiveFunction): Objective function to optimize.
            bounds (Bounds): Search space of the function.
            call_budget (int): Max number of calls to the objective function.
            tolerance (float): Tolerance of y value to count a solution as acceptable.
            target (float): Objective function value target, default 0.

        Returns:
            PointList: Results log from the optimization.
        """
        x0 = bounds.random_point(function.dim).x

        res_log = PointList(points=[])

        es = cma.CMAEvolutionStrategy(
            x0,
            self.metadata.hyperparameters["sigma0"],
            {
                "popsize": self.metadata.population_size,
                "bounds": bounds.to_list(),
                "maxfevals": call_budget,
                "ftarget": target,
                "verbose": -9,
                "tolfun": tolerance,
            },
        )

        while (
            not es.stop()
            and len(res_log) < call_budget
            and res_log.best_y() > target + tolerance
        ):
            solutions = PointList.from_list(es.ask())
            results = PointList(points=[function(x) for x in solutions.points])
            res_log.extend(results)
            x, y = results.pairs()
            es.tell(x, y)

        return res_log
