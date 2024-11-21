"""
Benchmarking CMA-ES algorithm on CEC 2017
"""

# pylint: disable=too-many-arguments
# pylint: disable=import-error

from typing import List, Tuple

import cma
import numpy as np
from tqdm import tqdm

from sofes.data_classes import ExperimentMetadata, ExperimentResults
from sofes.metamodels import ApproximateRankingMetamodel
from sofes.objective_functions import (
    CEC2017ObjectiveFunction,
    KNNSurrogateObjectiveFunction,
    ObjectiveFunction,
    SphereFunction,
)
from sofes.plotting import plot_ecdf_curves


def run_cmaes_on_cec(  # pylint: disable=too-many-positional-arguments, too-many-locals
    function: ObjectiveFunction,
    dims: int,
    population_size: int,
    call_budget: int,
    bounds: Tuple[float, float],
    sigma0: float,
    armknn_metamodel: bool = False,
    num_neighbours: int = 5,
    debug: bool = False,
) -> List[float]:
    """
    TODO
    """
    if armknn_metamodel:
        metamodel = ApproximateRankingMetamodel(
            population_size * 2,
            population_size,
            function,
            KNNSurrogateObjectiveFunction(num_neighbours),
        )

    x0 = np.random.uniform(low=bounds[0], high=bounds[1], size=dims)

    res_log = []

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            "popsize": population_size,
            "bounds": [bounds[0], bounds[1]],
            "maxfevals": call_budget,
            "ftarget": 0,
            "verbose": -9,
        },
    )

    while not es.stop():
        if armknn_metamodel:
            solutions = es.ask(population_size * 2)
            xy_pairs = metamodel(solutions)
            x, y = zip(*xy_pairs)
            es.tell(x, y)
        else:
            solutions = es.ask()
            y = [function(x) for x in solutions]
            res_log.extend(y)
            es.tell(solutions, y)

    if debug:
        print(es.stop())
        print(dict(es.result._asdict()))

    if armknn_metamodel:
        return metamodel.get_log()
    return res_log


if __name__ == "__main__":
    # hyperparams:
    DIM = 10
    POPSIZE = 40
    NUM_RUNS = 5

    metadata = ExperimentMetadata(
        method_name="cmaes",
        method_hyperparameters={
            "sigma0": 10,
            "popsize": "4*dim",
            "call_budget": 1e6,
            "bounds": (-100, 100),
        },
        metamodel_name="approximate_ranking_metamodel_knn",
        metamodel_hyperparameters={"num_neighbours": 5},
        benchmark_name="cec2017",
    )
    results = ExperimentResults(metadata)

    func = SphereFunction(DIM)
    func = CEC2017ObjectiveFunction(15, DIM)
    logs_vanilla = [
        run_cmaes_on_cec(
            func,
            DIM,
            POPSIZE,
            metadata.method_hyperparameters["call_budget"],
            metadata.method_hyperparameters["bounds"],
            30,
            debug=True,
        )
        for _ in tqdm(range(NUM_RUNS))
    ]

    logs_armknn5 = [
        run_cmaes_on_cec(
            func,
            DIM,
            POPSIZE,
            metadata.method_hyperparameters["call_budget"],
            metadata.method_hyperparameters["bounds"],
            30,
            armknn_metamodel=True,
            debug=True,
        )
        for _ in tqdm(range(NUM_RUNS))
    ]

    plot_ecdf_curves(
        {"vanilla": logs_vanilla, "armknn5": logs_armknn5}, n_dimensions=DIM
    )
