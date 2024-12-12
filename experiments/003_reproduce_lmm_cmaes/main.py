"""
Benchmarking CMA-ES algorithm on CEC 2017
"""

# pylint: disable=too-many-arguments
# pylint: disable=import-error, duplicate-code

from typing import List, Tuple

import cma
import numpy as np

# from matplotlib import pyplot as plt
from tqdm import tqdm

from optilab.data_classes import ExperimentMetadata, ExperimentResults
from optilab.metamodels import ApproximateRankingMetamodel
from optilab.functions import ObjectiveFunction
from optilab.functions.benchmarks.cec2017_objective_function import (
    CEC2017ObjectiveFunction,
)
from optilab.functions.surrogate import LocallyWeightedPolynomialRegression
from optilab.functions.unimodal import BentCigarFunction, SphereFunction

# from optilab.objective_functions.unimodal.sphere_function import SphereFunction
from optilab.plotting import plot_ecdf_curves


def lmm_cma_es(  # pylint: disable=too-many-positional-arguments, too-many-locals
    function: ObjectiveFunction,
    dims: int,
    population_size: int,
    call_budget: int,
    bounds: Tuple[float, float],
    sigma0: float,
    num_neighbours: int = 5,
    debug: bool = False,
) -> List[float]:
    """
    TODO
    """
    metamodel = ApproximateRankingMetamodel(
        population_size,
        population_size // 2,
        function,
        LocallyWeightedPolynomialRegression(num_neighbours, 2),
    )

    x0 = np.random.uniform(low=bounds[0], high=bounds[1], size=dims)

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            "popsize": population_size,
            "bounds": [bounds[0], bounds[1]],
            "verbose": -9,
        },
    )

    while (
        min(metamodel.get_log(), default=1) > 1e-8
        and len(metamodel.get_log()) <= call_budget
    ):
        solutions = [x.tolist() for x in es.ask()]
        metamodel.surrogate_function.set_covariance_matrix(es.C)
        metamodel.adapt(solutions)
        xy_pairs = metamodel(solutions)
        x, y = zip(*xy_pairs)
        es.tell(x, y)

        print(min(metamodel.get_log()), len(metamodel.get_log()))

    print(es.best)

    if debug:
        print(dict(es.result._asdict()))

    return metamodel.get_log()


if __name__ == "__main__":
    # hyperparams:
    DIM = 2
    POPSIZE = 6
    NUM_RUNS = 5

    CALL_BUDGET = 1e4 * DIM

    metadata = ExperimentMetadata(
        method_name="cmaes",
        method_hyperparameters={
            "sigma0": 10,
            "popsize": "4*dim",
            "call_budget": CALL_BUDGET,
            "bounds": (-100, 100),
        },
        metamodel_name="approximate_ranking_metamodel_knn",
        metamodel_hyperparameters={"num_neighbours": 5},
        benchmark_name="cec2017",
    )
    results = ExperimentResults(metadata)

    func = SphereFunction(DIM)
    func = CEC2017ObjectiveFunction(1, DIM)
    func = BentCigarFunction(DIM)

    logs_armknn5 = [
        lmm_cma_es(
            func,
            DIM,
            POPSIZE,
            metadata.method_hyperparameters["call_budget"],
            metadata.method_hyperparameters["bounds"],
            10,
            debug=True,
        )
        for _ in tqdm(range(NUM_RUNS))
    ]

    plot_ecdf_curves({"LMM-CMA-ES": logs_armknn5}, n_dimensions=DIM)



# CumulativeSquaredSums (2, 6), (4, 8), (8, 10), (16, 12) [-10, 10]
# RosenbrockFunction [-5, 5]
# Ackley [1, 30]
# 1e-10
# noisy sphere [-3, 7]