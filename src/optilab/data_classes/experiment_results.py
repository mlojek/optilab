"""
A data class to store experiment results and easily plot and export/import them.
"""

import json
from dataclasses import asdict
from typing import Any, Dict, List

import jsonschema
import numpy as np
import pandas as pd
from tabulate import tabulate

from ..plotting.box_plot import plot_box_plot
from ..plotting.ecdf_curve import plot_ecdf_curves
from .experiment_metadata import ExperimentMetadata
from .results_json_schema import results_json_schema


class ExperimentResults:
    """
    Class to easily store and analyze results of an experiment.
    """

    def __init__(
        self,
        metadata: ExperimentMetadata,
        data: List[Dict[str, str | List[List[float]]]] = None,
    ) -> None:
        """
        Class constructor.

        Args:
            metadata (ExperimentMetadata): Experiment metadata.
            data (List[Dict[str, str | List[List[float]]]]): Data to initialize the class with,
                optional.
        """
        self.metadata = metadata

        if not self.metadata.time_begin:
            self.metadata.begin_now()

        if data:
            self.validate_data_format(data)
            self.data = data
        else:
            self.data = []

    def validate_data_format(self, data: Any) -> None:
        """
        Validate data to check if it matches expected JSON schema.

        Args:
            data (Any): Data to be checked.

        Raises:
            jsonschema.exceptions.ValidationError: If data doesn't comply with expected schema.
        """
        jsonschema.validate(instance=data, schema=results_json_schema)

    def add_data(self, name: str, dim: int, logs: List[List[str]]) -> None:
        """
        Add another series of data.

        Args:
            name (str): The name of the series.
            dim (int): Dimensionality of optimized function.
            logs (List[List[str]]): Result or error logs from the optimization run.
        """
        self.data.append({"name": name, "dim": dim, "logs": logs})

    def save_to_json(self, savepath: str, indent: int = 4) -> None:
        """
        Dumps the content of the object to a json file.

        Args:
            savepath (str): Path to file to dump the data into.
            indent (int): JSON indent value, default is 4.
        """
        if not self.metadata.time_end:
            self.metadata.end_now()

        with open(savepath, "w", encoding="utf-8") as output_file_handle:
            json.dump(
                {"metadata": asdict(self.metadata), "data": self.data},
                output_file_handle,
                indent=indent,
            )

    @classmethod
    def from_json(cls, filepath: str):
        """
        Alternative constructor that reads the data directly from a JSON file.

        Args:
            filepath (str): Path to the JSON file with data.
        """
        with open(filepath, "r", encoding="utf-8") as input_file_handle:
            file_content = json.load(input_file_handle)
        return cls(ExperimentMetadata(**file_content["metadata"]), file_content["data"])

    def stats(self) -> pd.DataFrame:
        """
        Calculate stats of the data and expresses them as a pandas DataFrame.

        Returns:
            pd.DataFrame: A pandas dataframe with stats for each run in the data.
        """
        stats_data = [
            {
                "name": item["name"],
                "dim": item["dim"],
                "runs": len(item["logs"]),
                "min": min(min(log) for log in item["logs"]),
                "mean": np.mean([min(log) for log in item["logs"]]),
                "max": max(min(log) for log in item["logs"]),
                "std": np.std([min(log) for log in item["logs"]]),
            }
            for item in self.data
        ]
        return pd.DataFrame(stats_data)

    def print_stats(self) -> str:
        """
        Return printable stats of the data.

        Returns:
            str: Tabulated stats of the data.
        """
        return tabulate(
            self.stats(), headers="keys", tablefmt="github", showindex=False
        )

    def save_stats(self, savepath: str) -> None:
        """
        Saves stats to a CSV file.

        Args:
            savepath (str): Path to csv file to save the data in.
        """
        df = self.stats()
        df.to_csv(savepath, index=False)

    def plot_ecdf_curve(
        self,
        dim: int,
        savepath: str = None,
        n_thresholds: int = 100,
        allowed_error: float = 1e-8,
    ) -> None:
        """
        Plots ecdf curve for the data in this object.

        Args:
            dim (int): The dimensionality of problems to plot, only entries with
                matching dimensionalities will be plotted.
            savepath (str): Path to save the plot to, optional.
            n_thresholds (int): Number of ecdf value thresholds, default 100.
            allowed_error (float): Acceptable error value, default 1e-8.
        """
        plot_ecdf_curves(
            {item["name"]: item["logs"] for item in self.data if item["dim"] == dim},
            n_dimensions=dim,
            n_thresholds=n_thresholds,
            allowed_error=allowed_error,
            savepath=savepath,
        )

    def plt_box_plot(self, savepath: str = None) -> None:
        """
        Plots a box plot of the results.

        Args:
            savepath (str): Path to save the plot to, optional.
        """
        plot_box_plot(
            {item["name"]: [min(log) for log in item["logs"]] for item in self.data},
            savepath=savepath,
        )
