"""
Benchmarking CMA-ES algorithm on CEC 2017
"""

# pylint: disable=import-error

from cmaes_variations import cma_es, lmm_cma_es
from tqdm import tqdm
import numpy as np

from optilab.data_classes import Bounds
from optilab.functions.unimodal import CumulativeSquaredSums
from optilab.functions.multimodal import RosenbrockFunction
from optilab.plotting import plot_ecdf_curves
from optilab.data_classes import OptimizationRun, OptimizerMetadata
from optilab.utils import dump_to_pickle
from optilab.optimizers import CMAES

if __name__ == "__main__":
    # hyperparams:
    DIM = 2
    POPSIZE = 6
    NUM_RUNS = 51
    CALL_BUDGET = 1e4 * DIM
    TOL = 1e-10

    # optimized problem
    BOUNDS = Bounds(-10, 10)
    FUNC = CumulativeSquaredSums(DIM)

    precooked = CMAES(POPSIZE, 1)
    precooked_cmaes_run = precooked.run_optimization(NUM_RUNS, FUNC, BOUNDS, CALL_BUDGET, 0.0, TOL)

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

    # perform optimization with lmm cmaes
    lmm_cmaes_logs = [
        lmm_cma_es(FUNC, POPSIZE, CALL_BUDGET, BOUNDS, tolerance=TOL)
        for _ in tqdm(range(NUM_RUNS), unit="run")
    ]

    lmm_cmaes_run = OptimizationRun(
        model_metadata=OptimizerMetadata(
            name='LMM_CMA_ES',
            population_size=POPSIZE,
            hyperparameters={
                'sigma0': 1,
                'degree': 2,
                'num_neighbor': DIM * (DIM + 3) + 2
            }
        ),
        function_metadata=FUNC.get_metadata(),
        bounds=BOUNDS,
        tolerance=TOL,
        logs=lmm_cmaes_logs
    )

    # print stats
    vanilla_times = [len(log) for log in cmaes_logs]
    precooked_times = [len(log) for log in precooked_cmaes_run.logs]
    lmm_times = [len(log) for log in lmm_cmaes_logs]

    print(f'vanilla {np.average(vanilla_times)} {np.std(vanilla_times)}')
    print(f'precooked {np.average(precooked_times)} {np.std(precooked_times)}')
    print(f'lmm {np.average(lmm_times)} {np.std(lmm_times)}')
    
    # plot results
    plot_ecdf_curves(
        {
            "cma-es": cmaes_logs,
            "lmm-cma-es": lmm_cmaes_logs,
            'precooked': precooked_cmaes_run.logs
        },
        n_dimensions=DIM,
        savepath=f"ecdf_{FUNC.name}_{DIM}.png",
        allowed_error=TOL
    )

    dump_to_pickle([cmaes_run, lmm_cmaes_run, precooked_cmaes_run], f'lmm_reproduction_{FUNC.name}_{DIM}.pkl')
