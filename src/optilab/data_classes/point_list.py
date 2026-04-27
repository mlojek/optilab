"""
Class holding a list of points.
"""

from __future__ import annotations

from typing import Iterator, overload

import numpy as np
from pydantic import BaseModel

from .point import Point


class PointList(BaseModel):
    """
    Class holding a list of points. Might be used as optimization run result log
    or as a train set for surrogate function.
    """

    points: list[Point]
    "The list of points"

    # alternative constructors
    @classmethod
    def from_list(cls, xs: list[np.ndarray]) -> PointList:
        """
        Alternative constructor that takes a list of x values.

        Args:
            xs: List of x values.

        Returns:
            Object of class PointList containing points with given x.
        """
        return PointList(
            points=[Point(x=point, y=None, is_evaluated=False) for point in xs]
        )

    # adding points to the list
    def append(self, new_point: Point) -> None:
        """
        Add new point to the list.

        Args:
            new_point: Point to append to this object.
        """
        self.points.append(new_point)

    def extend(self, new_points: PointList) -> None:
        """
        Append a list of points to this PointList.

        Args:
            new_points: A list of point to append to this object.
        """
        self.points.extend(new_points.points)

    # getting point values
    def x(self) -> np.ndarray:
        """
        Get all x values of points in this list.

        Returns:
            List containing x values of all points.
        """
        return np.array([point.x for point in self.points], dtype=np.float64)

    def y(self) -> np.ndarray:
        """
        Get all y values of points in this list.

        Returns:
            List of y values of all points.
        """
        return np.array([point.y for point in self.points], dtype=np.float64)

    def pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the contents of this point list as list of x and list of y values.
        This is potentially useful for quickly accesing point values for training surrofates.

        Returns:
            Lists of x and y values.
        """
        return self.x(), self.y()

    def only_evaluated(self) -> PointList:
        """
        Return list of only those points that have been evaluated.

        Returns:
            List containing evaluated points.
        """
        return PointList(
            points=list(filter(lambda point: point.is_evaluated, self.points))
        )

    # magic methods for list abstraction
    def __iter__(self) -> Iterator[Point]:  # type: ignore
        return iter(self.points)

    @overload
    def __getitem__(self, index: int) -> Point: ...
    @overload
    def __getitem__(self, index: slice) -> PointList: ...
    def __getitem__(self, index: int | slice) -> Point | PointList:
        """
        Allows indexing and slicing this object like a list.

        Args:
            index: The index or slice of objects to fetch.

        Returns:
            A single Point object for integer index,
                or a new PointList instance for slicing.
        """
        if isinstance(index, slice):
            return PointList(points=self.points[index])
        return self.points[index]

    def __len__(self) -> int:
        """
        Return number of points stored in the list.

        Returns:
            Number of points stored in the list.
        """
        return len(self.points)

    # fancy methods
    def rank(self, *, reverse: bool = False) -> None:
        """
        Sort points by y value in place ascending.

        Args:
            reverse: If true, sorting is done descending. Default False.
        """
        self.points = list(
            sorted(self.points, key=lambda point: point.y, reverse=reverse)
        )

    def x_difference(self, other: PointList) -> PointList:
        """
        Return list of points in self that do not appear in other based on their x values.

        Args:
            other: Another PointList to compare against.

        Returns:
            List of points in self that are not in other.
        """
        return PointList(
            points=[
                point_self
                for point_self in self.points
                if point_self not in other.points
            ]
        )

    def remove_x(self) -> None:
        """
        Set x values of points to None. This is done to save memory since xs are rarely used.
        """
        for point in self:
            point.remove_x()

    # best value getters and similar methods
    def best(self) -> Point:
        """
        Get the best point by y value from the PointList.

        Returns:
            The Point with the lowest y value in the list.
        """
        return min(self.points, key=lambda point: point.y)

    def best_index(self) -> int:
        """
        Get the index of the best point by y value in the PointList.

        Returns:
            The index of the point with the lowest y value.
        """
        return min(range(len(self.points)), key=lambda i: self.points[i].y)

    def best_y(self) -> float:
        """
        Get the best y value found. If list is empty, infinity is returned.

        Returns:
            The best y value found.
        """
        return min((point.y for point in self.points), default=np.inf)

    def slice_to_best(self) -> PointList:
        """
        Return a list of all points up to the best in the list, including the best.

        Returns:
            List of points up to the best point.
        """
        return self[: self.best_index() + 1]
