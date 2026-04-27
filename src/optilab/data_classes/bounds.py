"""
Class representing bounds of the search space.
"""

from copy import deepcopy

import numpy as np
from pydantic import BaseModel

from .point import Point
from .point_list import PointList


class Bounds(BaseModel):
    """
    Class representing bounds of the search space.
    """

    lower: float
    "The lower bound of search space."

    upper: float
    "The upper bound of search space."

    def to_list(self) -> list[float]:
        """
        Return the bounds as a list of two floats.

        Returns:
            List containing the lower and upper bound.
        """
        return [self.lower, self.upper]

    def __len__(self) -> int:
        """
        Returns the width of the search space - the distance between the lower and upper bound.

        Returns:
            The width of the search space.
        """
        return int(self.upper - self.lower)

    def __str__(self) -> str:
        """
        Express the bounds as a string.

        Returns:
            Simple, printable string representation of this object.
        """
        return f"{self.lower} {self.upper}"

    def is_valid(self) -> bool:
        """
        Check if the bounds are valid, i.e. if lower bound is below upper bound.

        Returns:
            True if bounds are valid, false otherwise.
        """
        return self.lower < self.upper

    def __contains__(self, point: Point) -> bool:
        """
        Check if a point lies in the bounds. This method overrides the "in" operator.

        Returns:
            True if point lies in the bounds.
        """
        assert point.x is not None
        return bool(np.all((point.x >= self.lower) & (point.x <= self.upper)))

    def random_point(self, dim: int) -> Point:
        """
        Sample the bounds for a random point of given dimensionality.

        Args:
            dim: The dimensionality of the point.

        Returns:
            Randomly sampled point from the search space.
        """
        return Point(x=np.random.uniform(low=self.lower, high=self.upper, size=dim))

    def random_point_list(self, num_points: int, dim: int) -> PointList:
        """
        Sample the bounds for a list of random points of given dimensionality.

        Args:
            num_points: The number of points to sample.
            dim: The dimensionality of the points.

        Returns:
            List of randomly sampled points from the search space.
        """
        return PointList(points=[self.random_point(dim) for _ in range(num_points)])

    # search space bounds handling methods
    def reflect(self, point: Point) -> Point:
        """
        Handle bounds by reflecting the point back into the
        search area.

        Args:
            point: The point to handle.

        Returns:
            Reflected point.
        """
        assert point.x is not None
        new_x = []

        for val in point.x:
            if val < self.lower or val > self.upper:
                val -= self.lower
                remainder = val % (self.upper - self.lower)
                relative_distance = val // (self.upper - self.lower)

                if relative_distance % 2 == 0:
                    new_x.append(self.lower + remainder)
                else:
                    new_x.append(self.upper - remainder)
            else:
                new_x.append(val)

        new_point = deepcopy(point)
        new_point.x = np.array(new_x, dtype=np.float64)
        return new_point

    def wrap(self, point: Point) -> Point:
        """
        Handle bounds by wrapping the point around the
        search area.

        Args:
            point: The point to handle.

        Returns:
            Wrapped point.
        """
        assert point.x is not None
        new_x = []

        for val in point.x:
            if val < self.lower or val > self.upper:
                val -= self.lower
                val %= self.upper - self.lower
                val += self.lower
                new_x.append(val)
            else:
                new_x.append(val)

        new_point = deepcopy(point)
        new_point.x = np.array(new_x, dtype=np.float64)
        return new_point

    def project(self, point: Point) -> Point:
        """
        Handle bounds by projecting the point onto the bounds
        of the search area.

        Args:
            point: The point to handle.

        Returns:
            Projected point.
        """
        assert point.x is not None
        new_point = deepcopy(point)
        new_point.x = np.clip(point.x, self.lower, self.upper)
        return new_point

    def handle_bounds(self, point: Point, mode: str) -> Point:
        """
        Function to choose the bound handling method by name of the method.

        Args:
            point: The point to handle.
            mode: Bound handling mode to use, choose from reflect, wrap or project.

        Returns:
            Handled point.
        """
        methods = {"reflect": self.reflect, "wrap": self.wrap, "project": self.project}
        try:
            return methods[mode](point)
        except KeyError as err:
            raise ValueError(f"Invalid mode {mode} in Bounds.handle_bounds!") from err
