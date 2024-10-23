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
    def __init__(self, data: List[DataSeries] = None):
        """
        TODO
        """
        if data:
            self.validate_data_format(data)
            self.data = data
        else:
            self.data = []

    def validate_data_format(self, data: Any) -> None:
        """
        TODO

        :raises jsonschema.exceptions.ValidationError: if data doesn't comply with expected schema
        """
        jsonschema.validate(instance=data, schema=results_json_schema)

    def add_data(self, name: str, dim: int, logs: List[List[str]]) -> None:
        """
        TODO
        """
        self.data.append({"name": name, "dim": dim, "logs": logs})

    def save_to_json(self, savepath: str, indent: int = 4):
        """
        TODO
        """
        with open(savepath, "w") as output_file_handle:
            json.dump(self.data, output_file_handle, indent=indent)

    @classmethod
    def from_json(cls, filepath: str):
        """
        TODO
        """
        with open(filepath, "r") as input_file_handle:
            data = json.load(input_file_handle)
        return cls(data)

    # csv
    def stats(self):
        # TODO
        pass

    def print_stats(self):
        # TODO tabulate and print
        pass

    def stats_csv(self):
        # TODO
        pass

    # plotting
    def plot_ecdf_curve(self) -> None:
        """
        TODO
        """
        plot_ecdf_curves({item["name"]: item["logs"] for item in self.data})

    def plt_box_plot(self) -> None:
        """
        TODO
        """
        plot_box_plot(
            {item["name"]: [max(log) for log in item["logs"]] for item in self.data}
        )
