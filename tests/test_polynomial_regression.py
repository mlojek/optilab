# from typing import List

# import numpy as np

# from sofes.objective_functions import PolynomialRegression
# from sofes.objective_functions.polynomial_regression import PolynomialFeatures


# def polynomial(x: List[float]) -> float:
#     coeffs = np.array([1, 2, 3, 4, 5, 6])
#     poly = PolynomialFeatures(2)
#     vars = poly.fit_transform([x])[0]
#     return sum(coeffs * vars)


# def test_polynomial_regression():
#     x = np.random.random((1000, 2))
#     y = [polynomial(i) for i in x]
#     train_set = list(zip(x, y))
#     p = PolynomialRegression(2, train_set)
#     t = np.isclose(p.weights, np.array([1, 2, 3, 4, 5, 6]))
#     print(p.weights)
#     assert t.all()
