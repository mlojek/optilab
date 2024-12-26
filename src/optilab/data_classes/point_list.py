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

    # alternative constructors
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

    # adding points to the list
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

    # getting point values
    def x(self) -> List[np.ndarray]:
        """
        Get all x values of points in this list.

        Returns:
            List[np.ndarray]: List containing x values of all points.
        """
        return [point.x for point in self.points]

    def y(self) -> List[float]:
        """
        Get all y values of points in this list.

        Returns:
            List[float]: List of y values of all points.
        """
        return [point.y for point in self.points]

    def pairs(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Return the contents of this point list as list of x and list of y values.
        This is potentially useful for quickly accesing point values for training surrofates.

        Returns:
            Tuple[List[np.ndarray], List[float]]: Lists of x and y values.
        """
        return self.x(), self.y()

    def only_evaluated(self):
        """
        Return list of only those points that have been evaluated.

        Returns:
            PointList: List containing evaluated points.
        """
        return PointList(list(filter(lambda point: point.is_evaluated, self.points)))

    # magic methods for list abstraction
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

    # fancy methods
    def rank(self, *, reverse: bool = False) -> None:
        """
        Sort points by y value in place ascending.

        Args:
            reverse (bool): If true, sorting is done descending. Default False.
        """
        self.points = list(
            sorted(self.points, key=lambda point: point.y, reverse=reverse)
        )

    def x_difference(self, other):
        """
        Return list of points in self that do not appear in other based on their x values.

        Args:
            other (PointList): Another PointList to compare against.

        Returns:
            PointList: List of points in self that are not in other.
        """
        return PointList(
            points=[
                point_self
                for point_self in self.points
                if not point_self in other.points
            ]
        )

    # best value getters and similar methods
    def best(self) -> Point:
        """
        Get the best point by y value from the PointList.

        Returns:
            Point: The Point with the lowest y value in the list.
        """
        return min(self.points, key=lambda point: point.y)

    def best_index(self) -> int:
        """
        Get the index of the best point by y value in the PointList.

        Returns:
            int: The index of the point with the lowest y value.
        """
        return min(range(len(self.points)), key=lambda i: self.points[i].y)

    def best_y(self) -> float:
        """
        Get the best y value found. If list is empty, infinity is returned.

        Returns:
            float: The best y value found.
        """
        return min((point.y for point in self.points), default=np.inf)

    def slice_to_best(self):
        """
        Return a list of all points up to the best in the list, including the best.

        Returns:
            PointList: List of points up to the best point.
        """
        return PointList(self[: self.best_index() + 1])
