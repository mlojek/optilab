"""
Class containing information about an optimization run.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

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

    def bests_y(self) -> List[float]:
        """
        Get a list of best y values from each log.

        Returns:
            List[float]: List of the best values from each log.
        """
        return [log.best_y() for log in self.logs]

    def log_lengths(self) -> List[float]:
        """
        Get a list of log lengths.

        Returns:
            List[float]: List of the lengths of logs.
        """
        return [len(log) for log in self.logs]

    def stats(self) -> pd.DataFrame:
        """
        Make a summary of the run.

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
                "y_min": [min(self.bests_y())],
                "y_max": [max(self.bests_y())],
                "y_mean": [np.mean(self.bests_y())],
                "y_std": [np.std(self.bests_y())],
            }
        )
