"""
Point and PointList classes
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Point:
    """
    Point dataclass, representing a point in the search space.
    """

    x: np.ndarray
    "1D vector representing the point from the search space"

    y: float = None
    "Function value for this point."

    is_evaluated: bool = False
    "Wheather the value was generated by the objective function."


# pylint: disable=too-few-public-methods
class PointList:
    """
    Class holding a list of points.
    """

    def __init__(self, points: List[Point]) -> None:
        """
        Class constructor
        """
        self.points = points


#     def only_evaluated(self) -> List[Point]:
#         """
#
#         """
#         pass

#     def __getitem__(self, index: int) -> Point:
#         """
#
#         """
#         return self.points[index]

#     def __len__(self) -> int:
#         """
#
#         """
#         return len(self.points)
