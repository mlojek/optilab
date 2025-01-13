"""
LMM-CMA-ES optimizer: CMA-ES with local polynomial regression metamodel.
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments

from ..data_classes import Bounds, PointList
from ..functions import ObjectiveFunction
from ..functions.surrogate import LocallyWeightedPolynomialRegression
from ..metamodels import ApproximateRankingMetamodel
from .cma_es import CmaEs
from .optimizer import Optimizer


class LmmCmaEs(CmaEs):
    """
    LMM-CMA-ES optimizer: CMA-ES with local polynomial regression metamodel.
    """

    def __init__(self, population_size: int, sigma0: float, polynomial_dim: int):
        """
        Class constructor.

        Args:
            population_size (int): Size of the population.
            sigma0 (float): Starting value of the sigma,
            polynomial_dim (int): Dimension of the polynomial regression.
        """
        # Skipping super().__init__ and calling grandparent init instead.
        # pylint: disable=super-init-not-called, non-parent-init-called
        Optimizer.__init__(
            self,
            "lmm-cma-es",
            population_size,
            {"sigma0": sigma0, "polynomial_dim": polynomial_dim},
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
        num_neighbors = function.dim * (function.dim + 3) + 2

        metamodel = ApproximateRankingMetamodel(
            self.metadata.population_size,
            self.metadata.population_size // 2,
            function,
            LocallyWeightedPolynomialRegression(
                self.metadata.hyperparameters["polynomial_dim"], num_neighbors
            ),
        )

        es = self._spawn_cmaes(bounds, function.dim)

        while not self._stop(es, metamodel.get_log(), call_budget, target, tolerance):
            solutions = PointList.from_list(es.ask())
            metamodel.surrogate_function.set_covariance_matrix(es.C)
            metamodel.adapt(solutions)
            xy_pairs = metamodel(solutions)
            x, y = xy_pairs.pairs()
            es.tell(x, y)

        return metamodel.get_log()
