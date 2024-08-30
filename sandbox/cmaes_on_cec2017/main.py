from typing import Tuple
import json

import cma
import numpy as np
from tqdm import tqdm

from config import MAX_FES, FUNCTIONS, NUM_RUNS


def run_cmaes_on_cec(
    cec_function: callable,
    dims: int,
    population_size: int,
    call_budget: int,
    constraints: Tuple[float, float],
):
    """
    Runs the CMA-ES algorithm on a CEC function.
    
    :param cec_function: The CEC benchmark function to optimize.
    :param dims: The dimensionality of the problem.
    :param population_size: The population size for the CMA-ES algorithm.
    :param call_budget: The maximum number of function evaluations.
    :param constraints: The lower and upper bounds for the search space.
    :return: The best solution found by the CMA-ES algorithm.
    """
    x0 = np.random.uniform(low=constraints[0], high=constraints[1], size=dims)
    # print(x0)
    # exit(0)
    sigma0 = 0.3
    
    es = cma.CMAEvolutionStrategy(
        x0, 
        sigma0, 
        {
            'popsize': population_size, 
            'bounds': [constraints[0], constraints[1]], 
            'maxfevals': call_budget
        }
    )
    # TODO does the cmaes stick to bounds and maxfes by itself?

    while not es.stop():
        solutions = es.ask()
        # TODO odbijanie lamarcka
        # TODO better call budget tracking
        fitness_values = [cec_function([x.tolist()])[0] for x in solutions]

        if es.result.evaluations >= call_budget:
            break

        es.tell(solutions, fitness_values)

    return es.result.fbest


# 4 + 3sqrt(dim)
# 4dim
if __name__ == "__main__":
    results = {}

    for name, function, dimensions, constraints in FUNCTIONS:
        for dimension in dimensions:
            print(f"{name} dim {dimensions}")
            maxes = [
                run_cmaes_on_cec(
                    function,
                    dimension,
                    4 * dimension,
                    MAX_FES * dimension,
                    constraints
                )
                for _ in tqdm(range(NUM_RUNS))
            ]
            try:
                results[name][str(dimension)] = maxes
            except KeyError:
                results[name] = {
                    str(dimension): maxes
                }
        break

    with open("results.json", "w") as results_handle:
        json.dump(results, results_handle, indent=4)
