# """
# Simple unit tests for polynomial regression
# """

# from typing import List

# import numpy as np

# from optilab.functions.surrogate import PolynomialRegression
# from optilab.functions.surrogate.polynomial_regression import PolynomialFeatures


# def test_polynomial_regression():
#     """
#     Test if polynomial regression correctly calculated polynomial coefficients.
#     """

#     def polynomial(x: List[float]) -> float:
#         """
#         Function simulating a polynomial with fixed weights
#         """
#         coeffs = np.array([1, 2, 3, 4, 5, 6])
#         poly = PolynomialFeatures(2)
#         features = poly.fit_transform([x])[0]
#         return sum(coeffs * features)

#     x = np.random.random((1000, 2))
#     y = [polynomial(i) for i in x]
#     train_set = list(zip(x, y))
#     p = PolynomialRegression(2, train_set)
#     t = np.isclose(p.weights, np.array([1, 2, 3, 4, 5, 6]))
#     print(p.weights)
#     assert t.all()
