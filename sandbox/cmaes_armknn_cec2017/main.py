"""
CMA-ES on CEC2017 with metamodel from lmm-CMA-ES also known as Approximate Ranking Metamodel
"""

# pylint: disable=too-many-arguments


from typing import List, Tuple

import cma
import numpy as np
from tqdm import tqdm

from sofes.data_classes import ExperimentMetadata, ExperimentResults
from sofes.metamodels import ApproximateRankingMetamodel
from sofes.objective_functions import (
    CEC2017ObjectiveFunction,
    KNNSurrogateObjectiveFunction,
)

NUM_NEIGHBOURS = 5
EXPERIMENT_NAME = f"cmaes_armknn{NUM_NEIGHBOURS}_cec2017"


MAX_FES = 1e6
BOUNDS = [-100, 100]
SIGMA0 = 10
DIMS = [10]
POPSIZE_PER_DIM = 4
TOLERANCE = 1e-8
NUM_RUNS = 3
F_NUMS = [1]


def run_cmaes_on_cec(
    cec_function: CEC2017ObjectiveFunction,
    dims: int,
    population_size: int,
    call_budget: int,
    bounds: Tuple[float, float],
    tolerance: float,
    target: float,
    sigma0: float,
) -> List[float]:
    """
    Runs the CMA-ES algorithm on a CEC function.

    :param cec_function: The CEC benchmark function to optimize.
    :param dims: The dimensionality of the problem.
    :param population_size: The population size for the CMA-ES algorithm.
    :param call_budget: The maximum number of function evaluations.
    :param bounds: The lower and upper bounds for the search space.
    :param sigma0: The starting value of the sigma parameter.
    :return: The list of error values achieved through the optimization
    """
    metamodel = ApproximateRankingMetamodel(
        population_size * 2,
        population_size,
        cec_function,
        KNNSurrogateObjectiveFunction(NUM_NEIGHBOURS),
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
            "tolfun": tolerance,
            "ftarget": target,
            "verbose": -1,
        },
    )

    while not es.stop():
        solutions = es.ask(population_size * 2)
        xy_pairs = metamodel(solutions)
        x, y = zip(*xy_pairs)
        res_log.extend(y)
        es.tell(x, y)

    return metamodel.get_log()


if __name__ == "__main__":
    metadata = ExperimentMetadata(
        method_name="cmaes",
        metamodel_hyperparameters={"sigma0": SIGMA0, "popsize_per_dim": 4},
        metamodel_name="approximate_ranking_metamodel_knn",
        method_hyperparameters={"num_neighbours": NUM_NEIGHBOURS},
        benchmark_name="cec2017",
        time_begin="",
        time_end="",
    )
    results = ExperimentResults(metadata)

    for n in F_NUMS:
        for dimension in DIMS:
            print(f"{n} dim {dimension}")
            func = CEC2017ObjectiveFunction(n, dimension)
            maxes = [
                run_cmaes_on_cec(
                    func,
                    dimension,
                    POPSIZE_PER_DIM * dimension,
                    MAX_FES * dimension,
                    BOUNDS,
                    TOLERANCE,
                    0,
                    SIGMA0,
                )
                for _ in tqdm(range(NUM_RUNS))
            ]

            results.add_data(func.name, dimension, maxes)

    print(results.print_stats())
    results.save_to_json(f"{EXPERIMENT_NAME}.json")
    results.save_stats(f"{EXPERIMENT_NAME}cmaes_vanilla_cec2017.csv")
    results.plot_ecdf_curve(dim=10, savepath=f"{EXPERIMENT_NAME}_ecdf_10.png")
    results.plt_box_plot(savepath=f"{EXPERIMENT_NAME}_box.png")
