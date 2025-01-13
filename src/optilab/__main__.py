"""
Entrypoint for CLI functionality of optilab.
"""

import argparse
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from .data_classes import OptimizationRun
from .plotting import plot_box_plot, plot_convergence_curve, plot_ecdf_curves
from .utils.pickle_utils import load_from_pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Optilab CLI utility.")
    parser.add_argument(
        "pickle_path", type=Path, help="Path to pickle file with optimization runs."
    )
    args = parser.parse_args()

    filename_stem = args.pickle_path.stem

    data = load_from_pickle(args.pickle_path)

    assert isinstance(data, list)
    for run in data:
        assert isinstance(run, OptimizationRun)

    plot_convergence_curve(
        data={run.model_metadata.name: run.logs for run in data},
        savepath=f"{filename_stem}.convergence.png",
    )

    plot_ecdf_curves(
        data={run.model_metadata.name: run.logs for run in data},
        n_dimensions=data[0].function_metadata.dim,
        n_thresholds=100,
        allowed_error=data[0].tolerance,
        savepath=f"{filename_stem}.ecdf.png",
    )

    plot_box_plot(
        data={run.model_metadata.name: run.bests_y() for run in data},
        savepath=f"{filename_stem}.box_plot.png",
    )

    stats = pd.concat([run.stats() for run in data], ignore_index=True)

    stats.to_csv(f"{filename_stem}.stats.csv")
    print(tabulate(stats, headers="keys", tablefmt="github"))
