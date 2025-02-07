"""
Unit tests for Bounds.wrap method.
"""

import pytest


@pytest.mark.xfail
class TestBoundswrap:
    """
    Unit tests for Bounds.wrap method.
    """

    def test_point_in_bounds(self, example_bounds, point_in_bounds):
        """
        Test if wrapping works as expected when the point lies within bounds.
        """
        handled_point = example_bounds.wrap(point_in_bounds)
        assert handled_point.x == [14]

    def test_point_equal_lower_bound(self, example_bounds, point_equal_lower_bound):
        """
        Test if wrapping works as expected when the point lies on the lower bound.
        """
        handled_point = example_bounds.wrap(point_equal_lower_bound)
        assert handled_point.x == [10]

    def test_point_equal_upper_bound(self, example_bounds, point_equal_upper_bound):
        """
        Test if wrapping works as expected when the point lies on the upper bound.
        """
        handled_point = example_bounds.wrap(point_equal_upper_bound)
        assert handled_point.x == [20]

    def test_point_below_bounds(self, example_bounds, point_below_bounds):
        """
        Test if wrapping works as expected when the point lies below the lower bound.
        """
        handled_point = example_bounds.wrap(point_below_bounds)
        assert handled_point.x == [2]

    def test_point_above_bounds(self, example_bounds, point_above_bounds):
        """
        Test if wrapping works as expected when the point lies above the upper bound.
        """
        handled_point = example_bounds.wrap(point_above_bounds)
        assert handled_point.x == [14]

    def test_point_twice_below_bounds(self, example_bounds, point_twice_below_bounds):
        """
        Test if wrapping works as expected when the point lies far below the lower bound.
        """
        handled_point = example_bounds.wrap(point_twice_below_bounds)
        assert handled_point.x == [16]

    def test_point_twice_above_bounds(self, example_bounds, point_twice_above_bounds):
        """
        Test if wrapping works as expected when the point lies far above the upper bound.
        """
        handled_point = example_bounds.wrap(point_twice_above_bounds)
        assert handled_point.x == [12]

    def test_multidimensional(self, example_bounds, point_multidimensional):
        """
        Test if wrapping works as expected for a multidimensional point.
        """
        handled_point = example_bounds.wrap(point_multidimensional)
        assert handled_point.x == [14, 10, 20, 2, 14, 16, 12]
