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

    def x_difference(self, other) -> List[Point]:
        """
        Return list of points in self that do not appear in other based on their x values.

        Args:
            other (PointList): Another PointList to compare against.

        Returns:
            List[Point]: List of points in self that are not in other.
        """
        return PointList(
            points=[
                point_self
                for point_self in self.points
                if not any(
                    np.array_equal(point_self.x, point_other.x)
                    for point_other in other.points
                )
            ]
        )

    @classmethod
    def from_list(cls, xs: List[np.ndarray]):
        """
        Alternative constructor that takes a list of x values.

        Args:
            xs (List[np.ndarray]): List of x values.

        Returns:
            PointList: Object of class PointList containing points with given x.
        """
        return PointList(
            points=[Point(x=point, y=None, is_evaluated=False) for point in xs]
        )

    def append(self, new_point: Point) -> None:
        """
        Add new point to the list.

        Args:
            new_point (Point): Point to append to this object.
        """
        self.points.append(new_point)

    def extend(self, new_points) -> None:
        """
        Append a list of points to this PointList.

        Args:
            new_points (PointList): A list of point to append to this object.
        """
        self.points.extend(new_points.points)

    def pairs(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Return the contents of this point list as list of x and list of y values.
        This is potentially useful for quickly accesing point values for training surrofates.

        Returns:
            Tuple[List[np.ndarray], List[float]]: Lists of x and y values.
        """
        return [point.x for point in self.points], [point.y for point in self.points]

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

    def rank(self) -> None:
        """
        Sort the point in this list by y, ascending.
        """
        self.points = list(sorted(self.points, key=lambda point: point.y))

    def best_y(self) -> float:
        """
        Get the best y value found. If list is empty, infinity is returned.

        Returns:
            float: The best y value found.
        """
        return min((point.y for point in self.points), default=np.inf)
