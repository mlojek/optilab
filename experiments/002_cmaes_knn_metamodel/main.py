"""
Benchmarking CMA-ES algorithm on CEC 2017
"""

# pylint: disable=import-error

import numpy as np

from optilab.data_classes import Bounds
from optilab.functions.unimodal import SphereFunction
from optilab.plotting import plot_ecdf_curves
from optilab.utils import dump_to_pickle
from optilab.optimizers import CmaEs, KnnCmaEs

if __name__ == "__main__":
    # hyperparams:
    DIM = 2
    POPSIZE = DIM * 4
    NUM_NEIGHBORS = POPSIZE * 5
    NUM_RUNS = 51
    CALL_BUDGET = 1e4 * DIM
    TOL = 1e-8
    SIGMA0 = 1

    # optimized problem
    BOUNDS = Bounds(-100, 100)
    FUNC = SphereFunction(DIM)

    cmaes_optimizer = CmaEs(POPSIZE, SIGMA0)
    cmaes_results = cmaes_optimizer.run_optimization(NUM_RUNS, FUNC, BOUNDS, CALL_BUDGET, TOL)

    knn_optimizer = KnnCmaEs(POPSIZE, SIGMA0, NUM_NEIGHBORS)
    knn_results = knn_optimizer.run_optimization(NUM_RUNS, FUNC, BOUNDS, CALL_BUDGET, TOL)

    # print stats
    vanilla_times = [len(log) for log in cmaes_results.logs]
    knn_times = [len(log) for log in knn_results.logs]

    print(f'vanilla {np.average(vanilla_times)} {np.std(vanilla_times)}')
    print(f'knn {np.average(knn_times)} {np.std(knn_times)}')
    
    # plot results
    plot_ecdf_curves(
        {
            "cma-es": cmaes_results.logs,
            "knn-cma-es": knn_results.logs,
        },
        n_dimensions=DIM,
        savepath=f"ecdf_{FUNC.name}_{DIM}.png",
        allowed_error=TOL
    )

    dump_to_pickle([cmaes_results, knn_results], f'knn_reproduction_{FUNC.name}_{DIM}.pkl')
