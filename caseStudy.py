# Making imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Book1.csv')
X = data.iloc[:, 3]
Y = data.iloc[:, 4]
# X = data.iloc[:, 1]
# Y = data.iloc[:, 2]
plt.xlabel("Population")
plt.ylabel("GDP")   
plt.scatter(X, Y)
plt.show()

X_mean = np.mean(X)
Y_mean = np.mean(Y)

num = 0
den = 0
for i in range(len(X)):
    num += (X[i] - X_mean)*(Y[i] - Y_mean)
    den += (X[i] - X_mean)**2
m = num / den
c = Y_mean - m*X_mean

print (m, c)

Y_pred = m*X + c

print(Y_pred)
plt.scatter(X, Y)
plt.xlabel("Population")
plt.ylabel("GDP")
# plt.scatter(X, Y_pred, color='red')
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')
plt.show()