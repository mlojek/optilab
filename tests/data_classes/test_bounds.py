"""
Bounds dataclass unit tests.
"""

import numpy as np

from optilab.data_classes import Bounds, Point


class TestBounds:
    """
    Bounds dataclass unit tests class.
    """

    def test_valid_bounds(self):
        """
        Check if bounds where lower bound is below upper bound are valid.
        """
        bounds_object = Bounds(-100, 100)
        assert bounds_object.is_valid()

    def test_invalid_bounds(self):
        """
        Check if bounds where upper bound is below lower bound are invalid.
        """
        bounds_object = Bounds(100, -100)
        assert not bounds_object.is_valid()

    def test_invalid_equal_bounds(self):
        """
        Check if bounds where lower and upper bound are equal is invalid.
        """
        bounds_object = Bounds(100, 100)
        assert not bounds_object.is_valid()

    def test_to_list(self):
        """
        Check if bounds are correctly converted to a list.
        """
        bounds_object = Bounds(-100, 100)
        assert bounds_object.to_list() == [-100, 100]

    def test_point_in_bounds_1d(self):
        """
        Check if point laying in the bounds is correctly checked to lie in the bounds.
        """
        bounds_object = Bounds(-100, 100)
        simple_point = Point(np.array([10]))
        assert simple_point in bounds_object

    def test_point_in_bounds_1d_equal(self):
        """
        Check if a point laying on the edge of the bounds is considered lying in the bounds.
        """
        bounds_object = Bounds(-100, 100)
        simple_point = Point(np.array([100]))
        assert simple_point in bounds_object

    def test_point_in_bounds_5d(self):
        """
        Check if a 5D point is correctly marked to belong in the bounds.
        """
        bounds_object = Bounds(-100, 100)
        simple_point = Point(np.array([10, 0, -10, 44, 90]))
        assert simple_point in bounds_object

    def test_point_in_bounds_one_invalid(self):
        """
        Check if a point where one coordinate is outside of bounds is considered out of bounds.
        """
        bounds_object = Bounds(-100, 100)
        simple_point = Point(np.array([10, 0, -101, 44, 90]))
        assert simple_point not in bounds_object

    def test_point_in_bounds_all_invalid(self):
        """
        Check if a point where all coordinates are outside of bounds is considered out of bounds.
        """
        bounds_object = Bounds(30, 100)
        simple_point = Point(np.array([10, 0, -101, -10, 700]))
        assert simple_point not in bounds_object

    def test_random_point_in_bounds(self):
        """
        Check if a point randomly sampled from the bounds is considered in bounds.
        """
        bounds_object = Bounds(-1, 1)
        for _ in range(1000):
            sampled_point = bounds_object.random_point(dim=10)
            assert sampled_point in bounds_object

    def test_random_point_list_in_bounds(self):
        """
        Check if a randomly sample point list contains points from the bounds.
        """
        bounds_object = Bounds(-1, 1)
        sampled_point_list = bounds_object.random_point_list(1000, dim=10)
        for sampled_point in sampled_point_list:
            assert sampled_point in bounds_object
