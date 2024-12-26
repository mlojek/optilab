"""
Class containing information about an optimization run.
"""

from dataclasses import dataclass
from typing import List

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
        """
        return [log.best_y() for log in self.logs]
