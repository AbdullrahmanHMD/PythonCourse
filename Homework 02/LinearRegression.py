import numpy as np 
import pandas as pd
import math
import datetime
import os
import matplotlib.pyplot as plt

from scipy.stats import sem
import statistics

# Retrieving data from the csv file.
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

# credibleIntervals: finds the credible intervals given a confidence level
# returns the credible intervals, means and standard errors of samples from
# the given data set with specified sizes.
def credibleIntervals(X, Y, y_pred, confidence_level, number_of_samples, sample_size):
    n, k = X.shape
    low_interval = []
    high_interval = []

    means = []
    standard_errors = []

    i = 0
    variance = 0
    for sublist in y_pred.reshape(number_of_samples, sample_size):
        samples = []
        for point in sublist:
            error = np.subtract(Y.reshape(number_of_samples, sample_size)[i], sublist)
            variance = np.dot(error.T, error) / (sample_size - 1 - k)
            samples.append(point)

        i += 1
        standard_error = confidence_level * math.sqrt(variance / 5)
        mean = np.mean(samples)

        low = mean - standard_error 
        high = mean + standard_error

        low_interval.append(low)
        high_interval.append(high)

        means.append(mean)
        standard_errors.append(standard_error)
    
    return low_interval, high_interval, means, standard_errors
    
# The linear regression function
# X: An array of the independent variables of the data.
# Y: An array of the dependent variables of the data.
def LinearRegression(X, Y):

    # --Handling bad inputs part--------------------
    
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
            
    # Deleting the NaN values from both X and Y.
    nan_indecies = isArrayNumeric(X)
    for i in nan_indecies:
        del X[i]
        del Y[i]

    nan_indecies = isArrayNumeric(Y)
    for i in nan_indecies:
        del X[i]
        del Y[i]
    
    # --The linear regression part--------------------

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
    confidence_level = 1.96

    number_of_samples = 20
    size_of_sample = 5
    
    low_interval, high_interval, means, standard_errors = credibleIntervals(X, Y, y_pred, confidence_level, number_of_samples, size_of_sample)

    return y_pred, standard_errors, low_interval, high_interval, means

y_pred, standard_errors, low_interval, high_interval, means = LinearRegression(X, Y_truth)

# --Plotting the data-----------------------------

fig = plt.figure(figsize=(9, 6))

# Plotting the points.
plt.scatter(days_numbering, rates, color='green')

# Plotting regression line.
plt.plot(days_numbering, y_pred)

# Plotting the credible intervals.
plt.plot(days_numbering[::5], low_interval, color='r', linestyle='dashed', markerfacecolor='blue')
plt.plot(days_numbering[::5], high_interval, color='r', linestyle='dashed', markerfacecolor='blue')

plt.xlabel('Date / Day number')
plt.ylabel('USD to TL rate')
plt.show()

#-------------------------------------------------------------------

fig = plt.figure(figsize=(9, 6))

# # # Plotting the points.
plt.scatter(days_numbering, rates, color='green')

# # Plotting the credible intervals
plt.errorbar(x = days_numbering[::5], y=np.array(means), yerr=np.array(standard_errors), color='blue')

plt.xlabel('Date / Day number')
plt.ylabel('USD to TL rate')
plt.show()

