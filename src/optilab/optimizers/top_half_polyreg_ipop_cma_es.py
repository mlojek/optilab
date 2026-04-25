"""
Top-half PolyReg-IPOP-CMA-ES optimizer.

IPOP-CMA-ES enhanced with a polynomial-regression-based top-half metamodel:
each generation, the surrogate estimates all candidates and only the best half
is evaluated with the real objective function.
"""

from ..data_classes import Bounds, PointList
from ..functions import ObjectiveFunction
from ..functions.surrogate import PolynomialRegression
from ..metamodels import TopHalfMetamodel
from .cma_es import CmaEs
from .optimizer import Optimizer


class TopHalfPolyregIpopCmaEs(CmaEs):
    """
    Top-half PolyReg-IPOP-CMA-ES optimizer.

    Uses the :class:`TopHalfMetamodel` with a degree-2 polynomial regression
    surrogate: every generation the surrogate pre-screens all *lambda*
    candidates and only the best *mu* = *lambda* / 2 are evaluated with the
    real objective.  Penalised values for the remaining candidates guarantee
    that CMA-ES updates its state exclusively from real evaluations.
    """

    def __init__(
        self,
        population_size: int,
        buffer_size: int,
    ):
        """
        Class constructor.

        Args:
            population_size: Starting size of the population.
            buffer_size: Number of last evaluated points provided to the
                polynomial regression surrogate for training.
        """
        Optimizer.__init__(
            self,
            f"th-polyreg2b{buffer_size}-ipop-cma-es",
            population_size,
            {"buffer_size": buffer_size},
        )

    def optimize(
        self,
        function: ObjectiveFunction,
        bounds: Bounds,
        call_budget: int,
        tolerance: float,
        target: float = 0.0,
    ) -> PointList:
        """
        Run a single optimisation of the provided objective function.

        Args:
            function: Objective function to optimise.
            bounds: Search space of the function.
            call_budget: Max number of calls to the objective function.
            tolerance: Tolerance of y value to accept a solution.
            target: Objective function value target, default 0.

        Returns:
            Results log from the optimisation.
        """
        current_population_size = self.metadata.population_size

        metamodel = TopHalfMetamodel(
            self.metadata.population_size,
            self.metadata.population_size // 2,
            function,
            PolynomialRegression(degree=2),
            buffer_size=self.metadata.hyperparameters["buffer_size"],
        )

        dim = function.metadata.dim
        min_train = (dim + 1) * (dim + 2) // 2

        while not self._stop_external(
            metamodel.get_log(),
            current_population_size,
            call_budget,
            target,
            tolerance,
        ):
            es = self._spawn_cmaes(
                bounds,
                function.metadata.dim,
                current_population_size,
                len(bounds) / 2,
            )

            while not self._stop(
                es,
                metamodel.get_log(),
                current_population_size,
                call_budget,
                target,
                tolerance,
            ):
                solutions = PointList.from_list(es.ask())

                if len(metamodel.train_set) < min_train:
                    xy_pairs = metamodel.evaluate(solutions)
                else:
                    metamodel.adapt(solutions)
                    xy_pairs = metamodel(solutions)

                x, y = xy_pairs.pairs()
                es.tell(x, y)

            current_population_size *= 2
            metamodel.population_size *= 2
            metamodel.mu *= 2

        return metamodel.get_log()
