"""
Class holding a list of points.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .point import Point


@dataclass
class PointList:
    """
    Class holding a list of points. Might be used as optimization run result log
    or as a train set for surrogate function.
    """

    points: List[Point]
    "The list of points"

    def only_evaluated(self) -> List[Point]:
        """
        Return list of only those points that have been evaluated.

        Returns:
            List[Point]: List containing evaluated points.
        """
        return filter(lambda point: point.is_evaluated, self.points)

    def pairs(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Return the contents of this point list as list of x and list of y values.
        This is potentially useful for quickly accesing point values for training surrofates.

        Returns:
            Tuple[List[np.ndarray], List[float]]: Lists of x and y values.
        """
        return zip(*[[point.x, point.y] for point in self.points])

    def __getitem__(self, index: int) -> Point:
        """
        Allows to index this object like a list.

        Args:
            index (int): The index of object to fetch.

        Returns:
            Point: The object at given index.
        """
        return self.points[index]

    def __len__(self) -> int:
        """
        Return number of points stored in the list.

        Returns:
            int: Number of points stored in the list.
        """
        return len(self.points)
