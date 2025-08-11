"""
Class containing information about an optimization run.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import scipy

from .bounds import Bounds
from .function_metadata import FunctionMetadata
from .optimizer_metadata import OptimizerMetadata
from .point_list import PointList


@dataclass
class OptimizationRun:
    """
    Dataclass containing information about an optimization run.
    """

    model_metadata: OptimizerMetadata
    "Metadata describing the model used in optimization."

    function_metadata: FunctionMetadata
    "Metadata describing the optimized function."

    bounds: Bounds
    "Bounds of the search space."

    tolerance: float
    "Tolerated error value to stop the search."

    logs: List[PointList]
    "Logs of points from the optimization runs."

    def bests_y(self, raw_values: bool = False) -> List[float]:
        """
        Get a list of best y values from each log.

        Args:
            raw_values (bool): If false, values below tolerance values are set to tolerance, else
                return real y values. Default is false.

        Returns:
            List[float]: List of the best values from each log.
        """
        tolerance = -np.inf if raw_values else self.tolerance
        return [max(log.best_y(), tolerance) for log in self.logs]

    def log_lengths(self) -> List[float]:
        """
        Get a list of log lengths.

        Returns:
            List[float]: List of the lengths of logs.
        """
        return [len(log) for log in self.logs]

    def stats(self, raw_values: bool = False) -> pd.DataFrame:
        """
        Make a summary of the run.

        Args:
            raw_values (bool): If false, values below tolerance values are set to tolerance, else
                return real y values. Default is false.

        :Returns:
            pd.DataFrame: Dataframe containing stats and summary of the run.
        """
        return pd.DataFrame(
            {
                "model": [self.model_metadata.name],
                "function": [self.function_metadata.name],
                "runs": [len(self.logs)],
                "dim": [self.function_metadata.dim],
                "popsize": [self.model_metadata.population_size],
                "bounds": [str(self.bounds)],
                "tolerance": [self.tolerance],
                "evals_min": [min(self.log_lengths())],
                "evals_max": [max(self.log_lengths())],
                "evals_mean": [np.mean(self.log_lengths())],
                "evals_std": [np.std(self.log_lengths())],
                "y_min": [min(self.bests_y(raw_values))],
                "y_max": [max(self.bests_y(raw_values))],
                "y_mean": [np.mean(self.bests_y(raw_values))],
                "y_std": [np.std(self.bests_y(raw_values))],
                "y_median": [np.median(self.bests_y(raw_values))],
                "y_iqr": [scipy.stats.iqr(self.bests_y(raw_values))],
            }
        )

    def remove_x(self) -> None:
        """
        Set x values of points in the logs to None. This is done to save memory and storage since
        x values are rarely used.
        """
        for log in self.logs:
            log.remove_x()
