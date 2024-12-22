"""
Inter-extra-polation surrogate metamodel.
It uses different surrogate functions for interpolation and extrapolation.
"""

# pylint: disable=no-name-in-module

from scipy.spatial import ConvexHull
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon

from ..data_classes import Point, PointList
from ..functions.surrogate.surrogate_objective_function import (
    SurrogateObjectiveFunction,
)


class IEPolationSurrogate(SurrogateObjectiveFunction):
    """
    Inter-extra-polation surrogate metamodel.
    It uses different surrogate functions for interpolation and extrapolation.
    """

    def __init__(
        self,
        interpolation_surrogate: SurrogateObjectiveFunction,
        extrapolation_surrogate: SurrogateObjectiveFunction,
        train_set: PointList = None,
    ) -> None:
        """
        Class constructor.

        Args:
            interpolation_surrogate (SurrogateObjectiveFunction): Surrogate used for interpolation.
            extrapolation_surrogate (SurrogateObjectiveFunction): Surrogate used for extrapolation.
            train_set (PointList): Initial training set for the surrogates.
        """

        super().__init__("iepolation", train_set)

        self.interpolation_surrogate = interpolation_surrogate
        self.extrapolation_surrogate = extrapolation_surrogate

        self.convex_hull = None

        if train_set:
            self.build_convex_hull(train_set)

    def build_convex_hull(self, train_set: PointList) -> None:
        """
        Builds a convex hull from the train set. This convex hull is then used to determine
        wheather the point value should be interpolated or extrapolated.

        Args:
            train_set (PointList): Training set.
        """
        hull = ConvexHull(train_set.x())
        hull_points = hull.points[hull.vertices]
        self.convex_hull = Polygon(hull_points)

    def train(self, train_set: PointList) -> None:
        """
        Train both surrogate functions with provided data.

        Args:
            train_set (PointList): Training data for the model.
        """
        super().train(train_set)
        self.interpolation_surrogate.train(self.train_set)
        self.extrapolation_surrogate.train(self.train_set)
        self.build_convex_hull(self.train_set)

    def is_in_convex_hull(self, point: Point) -> bool:
        """
        Check if the given point lies inside of the convex hull of training points.

        Args:
            point (Point): The point to be checked.

        Returns:
            bool: True if the given point lies inside the convex hull, False otherwise.
        """
        return self.convex_hull.contains(ShapelyPoint(point.x))

    def __call__(self, point: Point) -> Point:
        """
        Estimate the value of a single point with the surrogate function.

        Args:
            x (Point): Point to estimate.

        Raises:
            ValueError: If dimensionality of x doesn't match self.dim.

        Return:
            Point: Estimated point.
        """
        super().__call__(point)

        if self.is_in_convex_hull(point):
            return self.interpolation_surrogate(point)

        return self.extrapolation_surrogate(point)
