#Code written by AryanNaik24
#Formulas used from https://www.geeksforgeeks.org/linear-regression-formula/


import numpy as np
import matplotlib.pyplot as plt

def linearR(x, y):
    # Y = mx + b
    n = len(x)
    sumY = sum(y)
    sumX = sum(x)
    sumX2 = sum(i * i for i in x)
    sumY2 = sum(i * i for i in y)
    sumXY = sum(a * b for a, b in zip(x, y))

    b = ((sumY * sumX2) - (sumX * sumXY)) / ((n * sumX2) - (sumX * sumX))
    m = ((n * sumXY) - (sumX * sumY)) / ((n * sumX2) - (sumX * sumX))
    
    print(f"x: {x}\ny: {y}\nsumX: {sumX}\nsumY: {sumY}\nsumX2: {sumX2}\nsumY2: {sumY2}\nsumXY: {sumXY}\nm: {m}\nb: {b}")
    
    return m, b

def predictVal(m,b,x):
    Y = m*x+b
    return Y


x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7.4, 7.9, 9, 11.3])

m, b = linearR(x, y)

x_line = np.linspace(min(x), max(x), 100)
y_line = m * x_line + b

pred_x = 3.5  
pred_y = predictVal(m, b, pred_x)

plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x_line, y_line, color='red', label='Regression line')
plt.scatter(pred_x, pred_y, color='green', marker='o', label='Predicted point')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Linear Regression")
plt.legend()
plt.grid()
plt.show()

