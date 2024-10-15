"""
Benchmarking CMA-ES algorithm on CEC 2017
"""
import json
from typing import Tuple

import cma
import numpy as np
from cec2017.functions import all_functions
from tqdm import tqdm

from visualize import plot_ecdf_curves

MAX_FES = 1e4
BOUNDS = [-100, 100]
SIGMA0 = 10
DIMS = [10, 30]
POPSIZE_PER_DIM = 4
TOLERANCE = 1e-8
NUM_RUNS = 10


def run_cmaes_on_cec(
    cec_function: callable,
    dims: int,
    population_size: int,
    call_budget: int,
    bounds: Tuple[float, float],
    tolerance: float,
    target: float,
    sigma0: float
):
    """
    Runs the CMA-ES algorithm on a CEC function.

    :param cec_function: The CEC benchmark function to optimize.
    :param dims: The dimensionality of the problem.
    :param population_size: The population size for the CMA-ES algorithm.
    :param call_budget: The maximum number of function evaluations.
    :param bounds: The lower and upper bounds for the search space.
    :param sigma0: The starting value of the sigma parameter.
    :return: The best solution found by the CMA-ES algorithm.
    """
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
        solutions = es.ask()
        fitness_values = [cec_function([x.tolist()])[0] for x in solutions]
        res_log.extend(fitness_values)
        es.tell(solutions, fitness_values)
        es.logger.add()

    res_log = [x - target for x in res_log]

    return res_log, es.result.fbest - target


if __name__ == "__main__":
    functions = [
        (f"f{i+1}", func, (i + 1) * 100) for i, func in enumerate(all_functions)
    ]

    results = []

    for name, function, target in functions:
        for dimension in DIMS:
            print(f"{name} dim {dimension}")
            maxes = [
                run_cmaes_on_cec(
                    function,
                    dimension,
                    POPSIZE_PER_DIM * dimension,
                    MAX_FES * dimension,
                    BOUNDS,
                    TOLERANCE,
                    target,
                    SIGMA0
                )
                for _ in tqdm(range(NUM_RUNS))
            ]

            log = [x[0] for x in maxes]
            plot_ecdf_curves({'cmaes': log})
            exit(0)
            maxes = [x[1] for x in maxes]

            results.append(
                {
                    "name": name,
                    "dimensions": dimension,
                    "results": maxes,
                    "mean": np.mean(maxes),
                    "stdev": np.std(maxes),
                    "min": np.min(maxes),
                    "max": np.max(maxes),
                }
            )

    with open("results.json", "w") as results_handle:
        json.dump(results, results_handle, indent=4)
