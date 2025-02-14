#Code written by AryanNaik24
#Formulas used from https://en.wikipedia.org/wiki/Polynomial_regression



import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv


def polynomial_regression(X, Y, degree):
    X_poly = np.column_stack([X**i for i in range(degree + 1)])
    B = pinv(X_poly.T @ X_poly) @ X_poly.T @ Y
    return B


def predict_values(B, X, degree):
    X_poly = np.column_stack([X**i for i in range(degree + 1)])
    return X_poly @ B


np.random.seed(42) 
k = [2, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324]
k.reverse()  # Don't ask why I reversed it.
j = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]

X = np.array(j)
Y = np.array(k) + np.random.normal(0, 15, size=len(k)) 

degree = 2

B = polynomial_regression(X, Y, degree)
X_range = np.linspace(min(X), max(X), 100)
Y_pred = predict_values(B, X_range, degree)
X_test = np.array([47])
Y_test_pred = predict_values(B, X_test, degree)[0]
print(f"Predicted value for Xtest: {Y_test_pred:.2f}")


# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='black', label='Data points', alpha=0.7)
plt.plot(X_range, Y_pred, color='red', label=f'Polynomial Regression (degree {degree})')
plt.scatter(X_test, Y_test_pred, color='blue', marker='o', s=100, label=f'Prediction for X=39')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polynomial Regression")
plt.legend()
plt.grid()
plt.show()
