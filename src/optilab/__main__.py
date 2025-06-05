"""
Entrypoint for CLI functionality of optilab.
"""

import argparse
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from .data_classes import OptimizationRun
from .plotting import plot_box_plot, plot_convergence_curve, plot_ecdf_curves
from .utils.aggregate_pvalues import aggregate_pvalues
from .utils.aggregate_stats import aggregate_stats
from .utils.pickle_utils import list_all_pickles, load_from_pickle
from .utils.stat_test import display_test_grid, mann_whitney_u_test_grid


def main():
    """
    Main function of the CLI utility. It's called when using optilab CLI command.
    """
    parser = argparse.ArgumentParser(description="Optilab CLI utility.", prog="optilab")
    parser.add_argument(
        "pickle_path",
        type=Path,
        help="Path to pickle file or directory with optimization runs.",
    )
    parser.add_argument(
        "--aggregate_pvalues",
        action="store_true",
        help="Aggregate pvalues of stat tests against run 0 in each pickle file into one table.",
    )
    parser.add_argument(
        "--aggregate_stats",
        action="store_true",
        help="Aggregate median and iqr for all processed runs into one table.",
    )
    parser.add_argument(
        "--entries",
        nargs="+",
        type=int,
        help="Space separated list of indexes of entries to include in analysis.",
    )
    parser.add_argument(
        "--hide_outliers",
        action="store_true",
        help="If specified, outliers will not be shown in the box plot.",
    )
    parser.add_argument(
        "--hide_plots",
        action="store_true",
        help="Hide plots when running the script.",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="If specified, no artifacts will be saved.",
    )
    parser.add_argument(
        "--raw_values",
        action="store_true",
        help="If specified, y values below tolerance are not substituted by tolerance value.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=Path.cwd(),
        help="Path to directory to save the artifacts. Default is the user's working directory.",
    )
    parser.add_argument(
        "--siginificance",
        type=float,
        default=0.05,
        help="Statistical significance of the U tests. Default value is 0.05.",
    )
    parser.add_argument(
        "--test_evals",
        action="store_true",
        help="Perform Mann-Whitney U test on eval values.",
    )
    parser.add_argument(
        "--test_y",
        action="store_true",
        help="Perform Mann-Whitney U test on y values.",
    )
    args = parser.parse_args()

    stats_to_aggregate_df = pd.DataFrame(
        columns=["model", "function", "y_median", "y_iqr"]
    )

    y_pvalues_to_aggregate_df = pd.DataFrame(
        columns=["model", "function", "alternative", "pvalue"]
    )
    evals_pvalues_to_aggregate_df = pd.DataFrame(
        columns=["model", "function", "alternative", "pvalue"]
    )

    file_path_list = list_all_pickles(args.pickle_path)

    for file_path in file_path_list:
        print(f"# File {file_path}")
        filename_stem = file_path.stem

        data = load_from_pickle(file_path)

        if args.entries:
            data = [data[i] for i in args.entries if 0 <= i < len(data)]

        assert isinstance(data, list)
        for run in data:
            assert isinstance(run, OptimizationRun)

        # plots
        plot_convergence_curve(
            data={run.model_metadata.name: run.logs for run in data},
            savepath=(
                (args.save_path / f"{filename_stem}.convergence.png")
                if not args.no_save
                else None
            ),
            show=not args.hide_plots,
            function_name=data[0].function_metadata.name,
        )

        plot_ecdf_curves(
            data={run.model_metadata.name: run.logs for run in data},
            n_dimensions=data[0].function_metadata.dim,
            n_thresholds=100,
            allowed_error=data[0].tolerance,
            savepath=(
                (args.save_path / f"{filename_stem}.ecdf.png")
                if not args.no_save
                else None
            ),
            show=not args.hide_plots,
            function_name=data[0].function_metadata.name,
        )

        plot_box_plot(
            data={
                run.model_metadata.name: run.bests_y(args.raw_values) for run in data
            },
            savepath=(
                (args.save_path / f"{filename_stem}.box_plot.png")
                if not args.no_save
                else None
            ),
            show=not args.hide_plots,
            function_name=data[0].function_metadata.name,
            hide_outliers=args.hide_outliers,
        )

        # stats
        stats = pd.concat(
            [run.stats(args.raw_values) for run in data], ignore_index=True
        )

        if args.aggregate_stats:
            stats_to_concat = pd.DataFrame(stats, columns=stats_to_aggregate_df.columns)
            stats_to_aggregate_df = pd.concat(
                [stats_to_aggregate_df, stats_to_concat], axis=0
            )

        stats_evals = stats.filter(like="evals_", axis=1)
        stats_y = stats.filter(like="y_", axis=1)
        stats_df = stats.drop(columns=stats_evals.columns.union(stats_y.columns))

        if not args.no_save:
            stats.to_csv(args.save_path / f"{filename_stem}.stats.csv")

        print(tabulate(stats_df, headers="keys", tablefmt="github"), "\n")
        print(tabulate(stats_y, headers="keys", tablefmt="github"), "\n")
        print(tabulate(stats_evals, headers="keys", tablefmt="github"), "\n")

        # stat tests
        if args.test_y:
            pvalues_y = mann_whitney_u_test_grid([run.bests_y() for run in data])

            if args.aggregate_pvalues:
                better_df = pd.DataFrame(
                    [
                        {
                            "model": stats.model,
                            "function": stats.function,
                            "alternative": "better",
                            "pvalue": row[0],
                        }
                        for row, (_, stats) in zip(
                            pvalues_y[1:], list(stats_df.iterrows())[1:]
                        )
                    ]
                )
                worse_df = pd.DataFrame(
                    [
                        {
                            "model": stats.model,
                            "function": stats.function,
                            "alternative": "worse",
                            "pvalue": pvalue,
                        }
                        for pvalue, (_, stats) in zip(
                            pvalues_y[0][1:], list(stats_df.iterrows())[1:]
                        )
                    ]
                )
                y_pvalues_to_aggregate_df = pd.concat(
                    [y_pvalues_to_aggregate_df, better_df, worse_df], axis=0
                )

            print("## Mann Whitney U test on optimization results (y).")
            print("p-values for alternative hypothesis row < column")
            print(display_test_grid(pvalues_y), "\n")

            if not args.no_save:
                pvalues_y_df = pd.DataFrame(
                    columns=list(range(len(data))), data=pvalues_y
                )
                pvalues_y_df.to_csv(args.save_path / f"{filename_stem}.pvalues_y.csv")

        if args.test_evals:
            pvalues_evals = mann_whitney_u_test_grid(
                [run.log_lengths() for run in data]
            )

            if args.aggregate_pvalues:
                better_df = pd.DataFrame(
                    [
                        {
                            "model": stats.model,
                            "function": stats.function,
                            "alternative": "better",
                            "pvalue": row[0],
                        }
                        for row, (_, stats) in zip(
                            pvalues_evals[1:],
                            list(stats_df.iterrows())[1:],
                        )
                    ]
                )
                worse_df = pd.DataFrame(
                    [
                        {
                            "model": stats.model,
                            "function": stats.function,
                            "alternative": "worse",
                            "pvalue": pvalue,
                        }
                        for pvalue, (_, stats) in zip(
                            pvalues_evals[0][1:],
                            list(stats_df.iterrows())[1:],
                        )
                    ]
                )
                evals_pvalues_to_aggregate_df = pd.concat(
                    [
                        evals_pvalues_to_aggregate_df,
                        better_df,
                        worse_df,
                    ],
                    axis=0,
                )

            print("## Mann Whitney U test on number of objective function evaluations.")
            print("p-values for alternative hypothesis row < column")
            print(display_test_grid(pvalues_evals), "\n")

            if not args.no_save:
                pvalues_evals_df = pd.DataFrame(
                    columns=list(range(len(data))), data=pvalues_evals
                )
                pvalues_evals_df.to_csv(
                    args.save_path / f"{filename_stem}.pvalues_evals.csv"
                )

    if args.aggregate_stats:
        aggregated_stats = aggregate_stats(stats_to_aggregate_df)

        print("# Aggregated stats")
        print(tabulate(aggregated_stats, headers="keys", tablefmt="github"), "\n")

        if not args.no_save:
            aggregated_stats.to_csv(
                args.save_path / "aggregated_stats.csv", index=False
            )

    if args.aggregate_pvalues:
        if args.test_y:
            aggregated_y_pvalues = aggregate_pvalues(y_pvalues_to_aggregate_df)

            print("# Aggregated y pvalues")
            print(
                tabulate(aggregated_y_pvalues, headers="keys", tablefmt="github"), "\n"
            )

            if not args.no_save:
                aggregated_y_pvalues.to_csv(
                    args.save_path / "aggregated_y_pvalues.csv", index=False
                )

        if args.test_evals:
            aggregated_evals_pvalues = aggregate_pvalues(evals_pvalues_to_aggregate_df)

            print("# Aggregated evals pvalues")
            print(
                tabulate(aggregated_evals_pvalues, headers="keys", tablefmt="github"),
                "\n",
            )

            if not args.no_save:
                aggregated_evals_pvalues.to_csv(
                    args.save_path / "aggregated_evals_pvalues.csv", index=False
                )


if __name__ == "__main__":
    main()
