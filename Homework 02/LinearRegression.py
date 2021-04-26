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

# isArrayNumeric: given an array, returns the indecies of the elements
# the are not numbers.
def isArrayNumeric(array):
    indexes = []
    for i in range(0, len(array)):
        if not isinstance(array[i], (float, int)):
            indexes.append(i)
    return indexes

# The linear regression function
# X: An array of the independent variables of the data.
# Y: An array of the dependent variables of the data.
def LinearRegression(X, Y):

    # Handling bad inputs.
    # If the first parameter is not an array.
    if not (isinstance(X, list) or isinstance(X, np.ndarray)):
        raise Exception("Invalid input type")
        return None, None, None, None

    # If the second parameter is not an array.
    if not (isinstance(Y, list) or isinstance(Y, np.ndarray)):
        raise Exception("Invalid input type")
        return None, None, None, None

    # If the two inputs are of different shapes.
    if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        if X.shape != Y.shape:
            raise Exception("Unmatched input shapes")
            return None, None, None, None

    elif isinstance(X, list) and isinstance(Y, list):
        if len(X) != len(Y):
            raise Exception("Unmatched input lengths")
            return None, None, None, None
    # If the two inputs are of different types.
    else:
        raise Exception("Parameter types mismatched, both parameter should be of the same type!")
        return None, None, None, None
            
    # Deleting the NaN values from both X and Y.
    nan_indecies = isArrayNumeric(X)
    for i in nan_indecies:
        del X[i]
        del Y[i]

    nan_indecies = isArrayNumeric(Y)
    for i in nan_indecies:
        del X[i]
        del Y[i]
    
    # The linear regression part.
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

    # Calculating the standard error.
    error = np.subtract(Y.reshape(len(Y), 1), y_pred)
    variance = np.dot(error.T, error) / (n - k - 1)
    standard_error = math.sqrt(variance) / math.sqrt(n)

    # Calculating the variance of beta.
    temp_mat = np.matrix(np.dot(X.T, X), dtype='float')
    beta_variance = np.multiply(variance, np.linalg.inv(temp_mat))

    # Calculating the credible intervals with 0.95 confidence.
    standard_errors = np.divide(np.sqrt(beta_variance.diagonal().astype(float)), math.sqrt(n))
    credible_intervals = np.array([np.subtract(beta_vals, 1.96 * standard_errors), np.add(beta_vals, 1.96 * standard_errors)])

    return y_pred, standard_errors, credible_intervals[0], credible_intervals[1]

y_pred, standard_errors, interval_1, interval_2 = LinearRegression(X, Y_truth)

# Plotting the data.
plt.plot(days_numbering, y_pred)
plt.scatter(days_numbering, rates, color='green')
# plt.show()