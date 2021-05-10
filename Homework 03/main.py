import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Retrieving data from the csv file.

file_name = "cses4_cut.csv"
path = os.path.abspath(os.getcwd()) + '\\' + file_name

# Retrieving the data from the csv file while execluding the first column
# which is enumerating the data.
data = pd.read_csv(path).to_numpy()[:, 1:] 

# Extracting the lables.
y_truth = X = data[:, -1]

# Converting True values to 1s and the False values to 0s.
f = lambda x : 1 if x == True else 0
y_truth_binary = np.array([f(i) for i in data[:, -1]])

# Retrieving the data points while execluding the lables.
X = data[:,: -1]

# -Filtering data:
#   A number of columns is chosen according to their impact on
#   the prediction and the others are discarded.
#
# -The columns being:
#   Gender, Education, Main Occupation, Religiosity, Race, Age.

# An array holding the indicies of the chosen features.
# Gender: 0, Education: 1, Occupation: 9, Religiosity: 23, Race: 25, Age: -1.
indecies = [0, 1, 9, 23, 25, -1]

# Constructing the traing and the test sets.
X = np.array([np.stack((X[:,i])) for i in indecies]).T
X_train, X_test, Y_train, Y_test = train_test_split(X, y_truth, test_size = 0.2, random_state=1)

print(X_train.shape)
print(X_test.shape)


