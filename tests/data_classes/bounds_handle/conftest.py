"""
Pytest fixtures used in Bounds handling unit tests.
"""

import pytest

from optilab.data_classes import Bounds, Point


@pytest.fixture()
def example_bounds() -> Bounds:
    """
    Example bounds used in bounds handling unit tests.
    """
    return Bounds(10, 20)


@pytest.fixture()
def point_in_bounds() -> Point:
    """
    Point that lies within the example_bounds fixture.
    Expected values for bounds handlers is the same as the point.
    """
    return Point([14])


@pytest.fixture()
def point_equal_lower_bound() -> Point:
    """
    Point that lies on the lower bound of example_bounds fixture.
    Expected values for all bounds handlers are equal to the point.
    """
    return Point([10])


@pytest.fixture()
def point_equal_upper_bound() -> Point:
    """
    Point that lies on the upper bound of example_bounds fixture.
    Expected values for all bounds handlers are equal to the point.
    """
    return Point([20])


@pytest.fixture()
def point_below_bounds() -> Point:
    """
    Point that lies below the example_bounds fixture.
    Expected values for bounds handlers are:
    - reflect: 18
    - wrap: 12
    - project: 10
    """
    return Point([2])


@pytest.fixture()
def point_above_bounds() -> Point:
    """
    Point that lies above the example_bounds fixture.
    Expected values for bounds handlers are:
    - reflect: 17
    - wrap: 13
    - project: 20
    """
    return Point([23])


@pytest.fixture
def point_twice_below_bounds() -> Point:
    """
    Point that lies below the lower bound of the example_bounds, and the difference
    in distance from the lower bound is bigger than the length of the bounds.
    Expected values for bounds handlers are:
    - reflect: 16
    - wrap: 16
    - project: 10
    """
    return Point([-4])


@pytest.fixture
def point_twice_above_bounds() -> Point:
    """
    Point that lies below the lower bound of the example_bounds, and the difference
    in distance from the lower bound is bigger than the length of the bounds.
    Expected values for bounds handlers are:
    - reflect: 12
    - wrap: 12
    - project: 20
    """
    return Point([32])


@pytest.fixture
def point_multidimensional() -> Point:
    """
    A multidimensional point that has all the values of the previous fixtures.
    Expected values for bounds handlers are:
    - reflect: [14, 10, 20, 18, 17, 16, 12]
    - wrap: [14, 10, 20, 12, 13, 16, 12]
    - project: [14, 10, 20, 10, 20, 10, 20]
    """
    return Point([14, 10, 20, 2, 23, -4, 32])


@pytest.fixture
def evaluated_point() -> Point:
    """
    An evaluated point, with y value and is_evaluated set to True.
    Used to check if the handled point has the same y and is_evaluated values.
    """
    return Point(x=[14, 10, 20, -4], y=10.1, is_evaluated=True)
