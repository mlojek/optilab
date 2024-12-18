"""
Simple unit tests for polynomial regression
"""

import numpy as np

from optilab.data_classes import Point, PointList
from optilab.functions.surrogate import PolynomialRegression
from optilab.functions.surrogate.polynomial_regression import PolynomialFeatures


def test_polynomial_regression():
    """
    Test if polynomial regression correctly calculated polynomial coefficients.
    """

    def polynomial(x: Point) -> Point:
        """
        Function simulating a polynomial with fixed weights
        """
        coeffs = np.array([1, 2, 3, 4, 5, 6])
        poly = PolynomialFeatures(2)
        features = poly.fit_transform([x.x])[0]
        return Point(x=x.x, y=sum(coeffs * features), is_evaluated=True)

    x = PointList.from_list(np.random.random((1000, 2)))
    train_set = PointList([polynomial(i) for i in x])
    p = PolynomialRegression(2, train_set)
    t = np.isclose(p.weights, np.array([1, 2, 3, 4, 5, 6]))
    print(p.weights)
    assert t.all()
