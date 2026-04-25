"""
CLI runner for analyzing optilab optimization result pickles.
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


class OptilabCLI:
    """CLI runner that analyzes optilab optimization result pickles."""

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize the CLI runner from parsed argparse arguments.

        Args:
            args: Parsed argument namespace from argparse.
        """
        self.pickle_path: Path = args.pickle_path
        self.aggregate_pvalues_flag: bool = args.aggregate_pvalues
        self.aggregate_stats_flag: bool = args.aggregate_stats
        self.entries: list[int] | None = args.entries
        self.hide_outliers: bool = args.hide_outliers
        self.hide_plots: bool = args.hide_plots
        self.no_save: bool = args.no_save
        self.raw_values: bool = args.raw_values
        self.save_path: Path = args.save_path
        self.significance: float = args.significance
        self.test_evals: bool = args.test_evals
        self.test_y: bool = args.test_y

        self.stats_to_aggregate_df = pd.DataFrame(
            columns=["model", "function", "y_median", "y_iqr"]  # type: ignore
        )
        self.y_pvalues_to_aggregate_df = pd.DataFrame(
            columns=["model", "function", "alternative", "pvalue"]  # type: ignore
        )
        self.evals_pvalues_to_aggregate_df = pd.DataFrame(
            columns=["model", "function", "alternative", "pvalue"]  # type: ignore
        )

    def run(self) -> None:
        """Iterate over all pickle files and analyze each one, then print aggregated results."""
        for file_path in list_all_pickles(self.pickle_path):
            self._analyze_file(file_path)
        self._finalize()

    def _analyze_file(self, file_path: Path) -> None:
        """
        Load and analyze a single pickle file.

        Args:
            file_path: Path to the pickle file containing optimization runs.
        """
        print(f"# File {file_path}")
        filename_stem = file_path.stem.split(".")[0]

        data = load_from_pickle(file_path)

        if self.entries:
            data = [data[i] for i in self.entries if 0 <= i < len(data)]

        assert isinstance(data, list)
        for run in data:
            assert isinstance(run, OptimizationRun)

        self._plot(data, filename_stem)
        stats_df = self._report_stats(data, filename_stem)

        if self.test_y:
            self._test_y(data, stats_df, filename_stem)
        if self.test_evals:
            self._test_evals(data, stats_df, filename_stem)

    def _plot(self, data: list[OptimizationRun], filename_stem: str) -> None:
        """
        Generate convergence curve, ECDF, and box plot for the given runs.

        Args:
            data: List of optimization runs to plot.
            filename_stem: Base name used for output file names.
        """
        plot_convergence_curve(
            data={run.model_metadata.name: run.logs for run in data},
            savepath=(
                (self.save_path / f"{filename_stem}.convergence.png")
                if not self.no_save
                else None
            ),
            show=not self.hide_plots,
            function_name=data[0].function_metadata.name,
        )

        plot_ecdf_curves(
            data={run.model_metadata.name: run.logs for run in data},
            n_dimensions=data[0].function_metadata.dim,
            n_thresholds=100,
            allowed_error=data[0].tolerance,
            savepath=(
                (self.save_path / f"{filename_stem}.ecdf.png")
                if not self.no_save
                else None
            ),
            show=not self.hide_plots,
            function_name=data[0].function_metadata.name,
        )

        plot_box_plot(
            data={
                run.model_metadata.name: run.bests_y(self.raw_values) for run in data
            },
            savepath=(
                (self.save_path / f"{filename_stem}.box_plot.png")
                if not self.no_save
                else None
            ),
            show=not self.hide_plots,
            function_name=data[0].function_metadata.name,
            hide_outliers=self.hide_outliers,
        )

    def _report_stats(
        self, data: list[OptimizationRun], filename_stem: str
    ) -> pd.DataFrame:
        """
        Compute, print, and optionally save descriptive statistics for the given runs.

        Args:
            data: List of optimization runs to compute statistics for.
            filename_stem: Base name used for the output CSV file name.

        Returns:
            DataFrame with non-evals/y columns, used as a label source for stat tests.
        """
        stats = pd.concat(
            [run.stats(self.raw_values) for run in data], ignore_index=True
        )

        if self.aggregate_stats_flag:
            stats_to_concat = pd.DataFrame(
                stats, columns=self.stats_to_aggregate_df.columns
            )
            self.stats_to_aggregate_df = pd.concat(
                [self.stats_to_aggregate_df, stats_to_concat], axis=0
            )

        stats_evals = stats.filter(like="evals_", axis=1)
        stats_y = stats.filter(like="y_", axis=1)
        stats_df = stats.drop(columns=stats_evals.columns.union(stats_y.columns))

        if not self.no_save:
            stats.to_csv(self.save_path / f"{filename_stem}.stats.csv")

        print(tabulate(stats_df, headers="keys", tablefmt="github"), "\n")
        print(tabulate(stats_y, headers="keys", tablefmt="github"), "\n")
        print(tabulate(stats_evals, headers="keys", tablefmt="github"), "\n")

        return stats_df

    def _test_y(
        self, data: list[OptimizationRun], stats_df: pd.DataFrame, filename_stem: str
    ) -> None:
        """
        Run Mann-Whitney U test on best y values and print the p-value grid.

        Args:
            data: List of optimization runs to test.
            stats_df: Stats DataFrame used to label models and functions in the aggregate.
            filename_stem: Base name used for the output CSV file name.
        """
        pvalues_y = mann_whitney_u_test_grid([run.bests_y() for run in data])

        if self.aggregate_pvalues_flag:
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
            self.y_pvalues_to_aggregate_df = pd.concat(
                [self.y_pvalues_to_aggregate_df, better_df, worse_df], axis=0
            )

        print("## Mann Whitney U test on optimization results (y).")
        print("p-values for alternative hypothesis row < column")
        print(display_test_grid(pvalues_y), "\n")

        if not self.no_save:
            pvalues_y_df = pd.DataFrame(
                columns=list(range(len(data))),  # type: ignore
                data=pvalues_y,
            )
            pvalues_y_df.to_csv(self.save_path / f"{filename_stem}.pvalues_y.csv")

    def _test_evals(
        self, data: list[OptimizationRun], stats_df: pd.DataFrame, filename_stem: str
    ) -> None:
        """
        Run Mann-Whitney U test on objective function evaluation counts and print the p-value grid.

        Args:
            data: List of optimization runs to test.
            stats_df: Stats DataFrame used to label models and functions in the aggregate.
            filename_stem: Base name used for the output CSV file name.
        """
        pvalues_evals = mann_whitney_u_test_grid([run.log_lengths() for run in data])

        if self.aggregate_pvalues_flag:
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
            self.evals_pvalues_to_aggregate_df = pd.concat(
                [
                    self.evals_pvalues_to_aggregate_df,
                    better_df,
                    worse_df,
                ],
                axis=0,
            )

        print("## Mann Whitney U test on number of objective function evaluations.")
        print("p-values for alternative hypothesis row < column")
        print(display_test_grid(pvalues_evals), "\n")

        if not self.no_save:
            pvalues_evals_df = pd.DataFrame(
                columns=list(range(len(data))),  # type: ignore
                data=pvalues_evals,
            )
            pvalues_evals_df.to_csv(
                self.save_path / f"{filename_stem}.pvalues_evals.csv"
            )

    def _finalize(self) -> None:
        """Print and optionally save aggregated stats and p-values across all processed files."""
        if self.aggregate_stats_flag:
            aggregated_stats = aggregate_stats(self.stats_to_aggregate_df)

            print("# Aggregated stats")
            print(tabulate(aggregated_stats, headers="keys", tablefmt="github"), "\n")

            if not self.no_save:
                aggregated_stats.to_csv(
                    self.save_path / "aggregated_stats.csv", index=False
                )

        if self.aggregate_pvalues_flag:
            if self.test_y:
                aggregated_y_pvalues = aggregate_pvalues(
                    self.y_pvalues_to_aggregate_df, self.significance
                )

                print("# Aggregated y pvalues")
                print(
                    tabulate(aggregated_y_pvalues, headers="keys", tablefmt="github"),
                    "\n",
                )

                if not self.no_save:
                    aggregated_y_pvalues.to_csv(
                        self.save_path / "aggregated_y_pvalues.csv", index=False
                    )

            if self.test_evals:
                aggregated_evals_pvalues = aggregate_pvalues(
                    self.evals_pvalues_to_aggregate_df, self.significance
                )

                print("# Aggregated evals pvalues")
                print(
                    tabulate(
                        aggregated_evals_pvalues, headers="keys", tablefmt="github"
                    ),
                    "\n",
                )

                if not self.no_save:
                    aggregated_evals_pvalues.to_csv(
                        self.save_path / "aggregated_evals_pvalues.csv", index=False
                    )
