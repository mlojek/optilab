"""
Benchmarking CMA-ES algorithm on CEC 2017
"""

# pylint: disable=import-error

from cmaes_variations import cma_es, knn_cma_es
from tqdm import tqdm
import numpy as np

from optilab.data_classes import Bounds
from optilab.functions.unimodal import SphereFunction
from optilab.functions.multimodal import RosenbrockFunction
from optilab.plotting import plot_ecdf_curves
from optilab.data_classes import OptimizationRun, OptimizerMetadata
from optilab.utils import dump_to_pickle

if __name__ == "__main__":
    # hyperparams:
    DIM = 10
    POPSIZE = DIM * 2
    NUM_NEIGHBORS = POPSIZE * 10
    NUM_RUNS = 51
    CALL_BUDGET = 1e4 * DIM
    TOL = 1e-8

    # optimized problem
    BOUNDS = Bounds(-100, 100)
    FUNC = SphereFunction(DIM)

    # perform optimization with vanilla cmaes
    cmaes_logs = [
        cma_es(FUNC, POPSIZE, CALL_BUDGET, BOUNDS, tolerance=TOL)
        for _ in tqdm(range(NUM_RUNS), unit="run")
    ]

    cmaes_run = OptimizationRun(
        model_metadata=OptimizerMetadata(
            name='CMA_ES',
            population_size=POPSIZE,
            hyperparameters={
                'sigma0': 1
            }
        ),
        function_metadata=FUNC.get_metadata(),
        bounds=BOUNDS,
        tolerance=TOL,
        logs=cmaes_logs
    )

    # perform optimization with knn cmaes
    knn_cmaes_logs = [
        knn_cma_es(FUNC, POPSIZE, CALL_BUDGET, BOUNDS, tolerance=TOL, num_neighbors=NUM_NEIGHBORS)
        for _ in tqdm(range(NUM_RUNS), unit="run")
    ]

    knn_cmaes_run = OptimizationRun(
        model_metadata=OptimizerMetadata(
            name='knn_CMA_ES',
            population_size=POPSIZE,
            hyperparameters={
                'sigma0': 1,
                'num_neighbor': NUM_NEIGHBORS
            }
        ),
        function_metadata=FUNC.get_metadata(),
        bounds=BOUNDS,
        tolerance=TOL,
        logs=knn_cmaes_logs
    )

    # print stats
    vanilla_times = [len(log) for log in cmaes_logs]
    knn_times = [len(log) for log in knn_cmaes_logs]

    print(f'vanilla {np.average(vanilla_times)} {np.std(vanilla_times)}')
    print(f'knn {np.average(knn_times)} {np.std(knn_times)}')
    
    # plot results
    plot_ecdf_curves(
        {
            "cma-es": cmaes_logs,
            "knn-cma-es": knn_cmaes_logs,
        },
        n_dimensions=DIM,
        savepath=f"ecdf_{FUNC.name}_{DIM}.png",
        allowed_error=TOL
    )

    dump_to_pickle([cmaes_run, knn_cmaes_run], f'knn_reproduction_{FUNC.name}_{DIM}.pkl')
