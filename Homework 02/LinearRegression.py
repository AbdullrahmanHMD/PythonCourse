import numpy as np 
import pandas as pd
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

# def predict_rate(X, Y, x_i):
#     term_1 = N * sum(np.multiply(x,y)) - sum(x) * sum(y)
#     term_2 = N * sum(X**2) - (sum(X))**2

#     m = term_1 / term_2

#     b = (sum(Y) - m * sum(X)) / N

#     f = lambda x : m * x + b
#     errors = np.subtract(y, y_pred)
#     return f(x_i), errors


def predicted_y(X, Y):
    X = np.array(X).reshape(len(X), 1)
    
    n, m = X.shape
    ones = np.ones((n,1))
    X = np.hstack((ones, X))
    temp_mat = np.matrix(np.dot(X.transpose(), X), dtype='float')
    
    term_1 = np.linalg.inv(temp_mat)
    term_2 = np.dot(X.transpose(), Y)

    y_hat = np.dot(term_1, term_2)

    y_pred = np.dot(X, y_hat.transpose())
    errors = np.subtract(Y.reshape(len(Y), 1), y_pred)
    
    return y_pred, errors


y_pred, errors = predicted_y(X, Y_truth)


plt.plot(days_numbering, y_pred)
plt.scatter(days_numbering, rates, color='green')
plt.show()