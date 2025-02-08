"""
Unit tests for Bounds.reflect method.
"""

import pytest


class TestBoundsReflect:
    """
    Unit tests for Bounds.reflect method.
    """

    def test_point_in_bounds(self, example_bounds, point_in_bounds):
        """
        Test if reflection works as expected when the point lies within bounds.
        """
        handled_point = example_bounds.reflect(point_in_bounds)
        assert handled_point.x == [14]

    def test_point_equal_lower_bound(self, example_bounds, point_equal_lower_bound):
        """
        Test if reflection works as expected when the point lies on the lower bound.
        """
        handled_point = example_bounds.reflect(point_equal_lower_bound)
        assert handled_point.x == [10]

    def test_point_equal_upper_bound(self, example_bounds, point_equal_upper_bound):
        """
        Test if reflection works as expected when the point lies on the upper bound.
        """
        handled_point = example_bounds.reflect(point_equal_upper_bound)
        assert handled_point.x == [20]

    def test_point_below_bounds(self, example_bounds, point_below_bounds):
        """
        Test if reflection works as expected when the point lies below the lower bound.
        """
        handled_point = example_bounds.reflect(point_below_bounds)
        assert handled_point.x == [18]

    def test_point_above_bounds(self, example_bounds, point_above_bounds):
        """
        Test if reflection works as expected when the point lies above the upper bound.
        """
        handled_point = example_bounds.reflect(point_above_bounds)
        assert handled_point.x == [17]

    def test_point_twice_below_bounds(self, example_bounds, point_twice_below_bounds):
        """
        Test if reflection works as expected when the point lies far below the lower bound.
        """
        handled_point = example_bounds.reflect(point_twice_below_bounds)
        assert handled_point.x == [16]

    def test_point_twice_above_bounds(self, example_bounds, point_twice_above_bounds):
        """
        Test if reflection works as expected when the point lies far above the upper bound.
        """
        handled_point = example_bounds.reflect(point_twice_above_bounds)
        assert handled_point.x == [12]

    def test_multidimensional(self, example_bounds, point_multidimensional):
        """
        Test if reflection works as expected for a multidimensional point.
        """
        handled_point = example_bounds.reflect(point_multidimensional)
        assert handled_point.x == [14, 10, 20, 18, 17, 16, 12]

    def test_evaluated_point(self, example_bounds, evaluated_point):
        """
        Test if reflecting a point leaves y and is_evaluated members unchanged.
        """
        handled_point = example_bounds.reflect(evaluated_point)
        assert handled_point.y == 10.1
        assert handled_point.is_evaluated
