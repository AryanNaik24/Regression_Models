#Code written by AryanNaik24
#Formulas used from https://online.stat.psu.edu/stat462/node/132/
#Example idea taken from https://www.scribbr.com/statistics/multiple-linear-regression/


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

def multipleLinearReg(X, Y):
    # formula B = (X'X)^(-1) X'Y
    X = np.column_stack((np.ones(X.shape[0]), X))
 
    B = pinv(X.T @ X) @ X.T @ Y
    
    print(f"Coefficients: {B}")
    return B

def predictVal(B, X):
    X = np.column_stack((np.ones(X.shape[0]), X))
    return X @ B

X = np.array([
    [5, 30], [10, 25], [15, 20], [20, 15], [25, 10],
    [30, 5], [35, 8], [40, 12], [45, 18], [50, 22]
])  
Y = np.array([18, 17, 15, 14, 12, 10, 9, 8, 7, 6])  

B = multipleLinearReg(X, Y)

biking_range = np.linspace(0, 60, 100)
smoking_levels = [5, 15, 30]  
colors = ['red', 'green', 'blue']

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], Y, color='black', label='Data points')

for smoke, color in zip(smoking_levels, colors):
    predicted_Y = predictVal(B, np.column_stack((biking_range, np.full_like(biking_range, smoke))))
    plt.plot(biking_range, predicted_Y, color=color, label=f'Smoking {smoke}%')

plt.xlabel("Biking to work (% of population)")
plt.ylabel("Heart disease (% of population)")
plt.title("Rates of heart disease as a function of biking to work and smoking")
plt.legend(title="Smoking (% of population)")
plt.grid()
plt.show()
