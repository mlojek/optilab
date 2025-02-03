"""
Entrypoint for CLI functionality of optilab.
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from scipy.stats import mannwhitneyu
from tabulate import tabulate

from .data_classes import OptimizationRun
from .plotting import plot_box_plot, plot_convergence_curve, plot_ecdf_curves
from .utils.pickle_utils import load_from_pickle


def mann_whitney_u_test_grid(data_lists: List[List[float]]) -> str:
    """
    TODO
    """
    n = len(data_lists)
    results_table = [["-" for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                p_value = mannwhitneyu(
                    data_lists[i], data_lists[j], alternative="less"
                )[1]
                results_table[i][j] = f"{p_value:.4f}"

    header = list(range(n))
    return tabulate(results_table, headers=header, showindex="always", tablefmt="grid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Optilab CLI utility.")
    parser.add_argument(
        "pickle_path",
        type=Path,
        help="Path to pickle file or directory with optimization runs.",
    )
    parser.add_argument(
        "--hide_plots", action="store_true", help="Hide plots when running the script."
    )
    args = parser.parse_args()

    file_path_list = []

    if args.pickle_path.is_file():
        file_path_list.append(args.pickle_path)
    elif args.pickle_path.is_dir():
        for file_path in sorted(args.pickle_path.iterdir()):
            if file_path.is_file() and file_path.suffix == ".pkl":
                file_path_list.append(file_path)

    for file_path in file_path_list:
        print(f"file {file_path}")
        filename_stem = file_path.stem

        data = load_from_pickle(file_path)

        assert isinstance(data, list)
        for run in data:
            assert isinstance(run, OptimizationRun)

        plot_convergence_curve(
            data={run.model_metadata.name: run.logs for run in data},
            savepath=f"{filename_stem}.convergence.png",
            show=not args.hide_plots,
        )

        plot_ecdf_curves(
            data={run.model_metadata.name: run.logs for run in data},
            n_dimensions=data[0].function_metadata.dim,
            n_thresholds=100,
            allowed_error=data[0].tolerance,
            savepath=f"{filename_stem}.ecdf.png",
            show=not args.hide_plots,
        )

        plot_box_plot(
            data={run.model_metadata.name: run.bests_y() for run in data},
            savepath=f"{filename_stem}.box_plot.png",
            show=not args.hide_plots,
        )

        stats = pd.concat([run.stats() for run in data], ignore_index=True)

        stats.to_csv(f"{filename_stem}.stats.csv")
        print(tabulate(stats, headers="keys", tablefmt="github"))

        print()
        print("Mann Whitney U test of optimization results (y).")
        print("p-values for alternative hypothesis row < column")
        best_data = [run.bests_y() for run in data]
        print(mann_whitney_u_test_grid(best_data))
