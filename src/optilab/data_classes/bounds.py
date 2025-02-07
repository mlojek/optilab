"""
Class representing bounds of the search space.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import List

import numpy as np

from .point import Point
from .point_list import PointList


@dataclass
class Bounds:
    """
    Class representing bounds of the search space.
    """

    lower: float
    "The lower bound of search space."

    upper: float
    "The upper bound of search space."

    def to_list(self) -> List[float]:
        """
        Return the bounds as a list of two floats.

        Returns:
            List[float]: List containing the lower and upper bound.
        """
        return [self.lower, self.upper]

    def __str__(self) -> str:
        """
        Express the bounds as a string.

        Returns:
            str: Simple, printable string representation of this object.
        """
        return f"{self.lower} {self.upper}"

    def is_valid(self) -> bool:
        """
        Check if the bounds are valid, i.e. if lower bound is below upper bound.

        Returns:
            bool: True if bounds are valid, false otherwise.
        """
        return self.lower < self.upper

    def __contains__(self, point: Point) -> bool:
        """
        Check if a point lies in the bounds. This method overrides the "in" operator.

        Returns:
            bool: True if point lies in the bounds.
        """
        return np.all((point.x >= self.lower) & (point.x <= self.upper))

    def random_point(self, dim: int) -> Point:
        """
        Sample the bounds for a random point of given dimensionality.

        Args:
            dim (int): The dimensionality of the point.

        Returns:
            Point: Randomly sampled point from the search space.
        """
        return Point(np.random.uniform(low=self.lower, high=self.upper, size=dim))

    def random_point_list(self, num_points: int, dim: int) -> PointList:
        """
        Sample the bounds for a list of random points of given dimensionality.

        Args:
            num_points (int): The number of points to sample.
            dim (int): The dimensionality of the points.

        Returns:
            Point: List of randomly sampled points from the search space.
        """
        return PointList([self.random_point(dim) for _ in range(num_points)])

    # search space bounds handling methods
    def reflect(self, point: Point) -> Point:
        """
        Handle bounds by reflecting the point back into the
        search area.

        Args:
            point (Point): The point to handle.

        Returns:
            Point: Reflected point.
        """
        raise NotImplementedError

    def wrap(self, point: Point) -> Point:
        """
        Handle bounds by wrapping the point around the
        search area.

        Args:
            point (Point): The point to handle.

        Returns:
            Point: Wrapped point.
        """
        raise NotImplementedError

    def project(self, point: Point) -> Point:
        """
        Handle bounds by projecting the point onto the bounds
        of the search area.

        Args:
            point (Point): The point to handle.

        Returns:
            Point: Projected point.
        """
        new_x = []

        for val in point.x:
            if val < self.lower:
                new_x.append(deepcopy(self.lower))
            elif val > self.upper:
                new_x.append(deepcopy(self.upper))
            else:
                new_x.append(val)

        point.x = new_x
        return point

    def handle_bounds(self, point: Point, mode: str) -> Point:
        """
        Function to choose the bound handling method by name of the method.

        Args:
            point (Point): The point to handle.
            mode (str): Bound handling mode to use, choose from reflect, wrap or project.

        Returns:
            Point: Handled point.
        """
        methods = {"reflect": self.reflect, "wrap": self.wrap, "project": self.project}
        try:
            return methods[mode](point)
        except KeyError as err:
            raise ValueError(f"Invalid mode {mode} in Bounds.handle_bounds!") from err
