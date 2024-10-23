"""
A data class to store experiment results and easily plot and export/import them.
"""

import json
from typing import Any, Dict, List, Union

import jsonschema
import pandas as pd

from ..plotting.box_plot import plot_box_plot
from ..plotting.ecdf_curve import plot_ecdf_curves
from .results_json_schema import results_json_schema

DataSeries = Dict[str, Union[str, List[List[float]]]]


class ExperimentResults:
    """
    Class to easily store and analyze results of an experiment.
    """

    def __init__(self, data: List[DataSeries] = None) -> None:
        """
        Class constructor.

        :param data: optional, data to initialize the class with.
        """
        if data:
            self.validate_data_format(data)
            self.data = data
        else:
            self.data = []

    def validate_data_format(self, data: Any) -> None:
        """
        Validate data to check if it matches expected JSON schema.

        :param data: data to be checked
        :raises jsonschema.exceptions.ValidationError: if data doesn't comply with expected schema
        """
        jsonschema.validate(instance=data, schema=results_json_schema)

    def add_data(self, name: str, dim: int, logs: List[List[str]]) -> None:
        """
        Add another series of data.

        :param name: name of the series
        :param dim: dimensionality of solved problem
        :param logs: result or error logs from the optimization run
        """
        self.data.append({"name": name, "dim": dim, "logs": logs})

    def save_to_json(self, savepath: str, indent: int = 4) -> None:
        """
        Dumps the content of the object to a json file.

        :param savepath: path to file to dump the data into
        :param indent: json indent, default is 4
        """
        with open(savepath, "w") as output_file_handle:
            json.dump(self.data, output_file_handle, indent=indent)

    @classmethod
    def from_json(cls, filepath: str):
        """
        Alternate constructor that reads the data directly from a JSON file.

        :param filepath: path to file with data
        :return: instance of ExperimentResults
        """
        with open(filepath, "r") as input_file_handle:
            data = json.load(input_file_handle)
        return cls(data)

    # # csv
    # def stats(self):
    #     # TODO
    #     pass

    # def print_stats(self):
    #     # TODO tabulate and print
    #     pass

    # def stats_csv(self):
    #     # TODO
    #     pass

    def plot_ecdf_curve(
        self,
        dim: int,
        savepath: str = None,
        n_thresholds: int = 100,
        allowed_error: float = 1e-8,
    ) -> None:
        """
        Plots ecdf curve for the data in this object.

        :param dim: dimensionality of problems to plot, only entries with matching dimensionalities will be plotted
        :param savepath: optional, path to save the plot to
        :param n_thresholds: optional, number of ecdf value thresholds, default 100
        :param allowed_error: optional, acceptable error value, default 1e-8
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

        :param savepath: optional, path to save the plot to
        """
        plot_box_plot(
            {item["name"]: [max(log) for log in item["logs"]] for item in self.data},
            savepath=savepath,
        )
