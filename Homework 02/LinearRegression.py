import numpy as np 
import pandas as pd
import math
import datetime
import os
import matplotlib.pyplot as plt

file_name = "data.csv"
path = os.path.abspath(os.getcwd()) + '\\' + file_name
data = pd.read_csv(path).to_numpy()

N = len(data)

dates = data[:,1]
days_numbering = data[:,2]
rates = data[:,3]

X = days_numbering
Y_truth = rates

def predicted_y(X, Y):
    X = np.array(X).reshape(len(X), 1)
    
    n, k = X.shape

    # Appending a vector of ones to the X matrix.
    ones = np.ones((n,1))
    X = np.hstack((ones, X))

    # Calculating the Beta values.
    temp_mat = np.matrix(np.dot(X.T, X), dtype='float')
    term_1 = np.linalg.inv(temp_mat)
    term_2 = np.dot(X.T, Y)
    beta_vals = np.dot(term_1, term_2)

    # Calculating the predicted Y values.
    y_pred = np.array(np.dot(X, beta_vals.T))

    # Calculating the standard error
    error = np.subtract(Y.reshape(len(Y), 1), y_pred)
    variance = np.dot(error.T, error) / (n - k - 1)
    standard_error = math.sqrt(variance) / math.sqrt(n)


    temp_mat = np.matrix(np.dot(X.T, X), dtype='float')
    var_y_hat = np.multiply(variance, np.linalg.inv(temp_mat))
 
    return y_pred, standard_error

y_pred, standard_error = predicted_y(X, Y_truth)

print("Standard error: {}".format(standard_error))

plt.plot(days_numbering, y_pred)
plt.scatter(days_numbering, rates, color='green')
plt.show()