import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
# Retrieving data from the csv file.

file_name = "cses4_cut.csv"
path = os.path.abspath(os.getcwd()) + '\\' + file_name

# Retrieving the data from the csv file while execluding the first column
# which is enumerating the data.
data = pd.read_csv(path) 
features = data.columns
data = data.to_numpy()

# Extracting the lables.
y_truth = data[:, -1]

# Converting True values to 1s and the False values to 0s.
f = lambda x : 1 if x == True else 0
y_truth_binary = np.array([f(i) for i in data[:, -1]])

# Retrieving the data points while execluding the lables.
X = data[:,: -1]
features = features[:-1]

# -Filtering data:
#   A number of columns is chosen according to their impact on
#   the prediction and the others are discarded.
#
# -The columns being:
#   Gender, Education, Main Occupation, Religiosity, Race, Age.

# An array holding the indicies of the chosen features.
# Gender: D2002, Education: D2003, Occupation: D2011,
# Religiosity: D2023, Race: D2027, Age: age.
selected_features = ['D2002', 'D2003', 'D2011', 'D2023', 'D2027', 'age']
features = features.to_list()
selected_features_indecies = [features.index(i) for i in selected_features]

X = np.array([np.stack((X[:,i])) for i in selected_features_indecies]).T

from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder()

# X_encoded = [np.array(X[:,i]).reshape(-1, 1) for i in range(0, len(selected_features))]
# X_encoded = np.array([np.stack(oneHotEncoder.fit_transform(X_encoded[i])) for i in range(0, len(selected_features))]).T

# print(X_encoded[0])


# Constructing train and test sets.
# X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, y_truth_binary, test_size = 0.2, random_state=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, y_truth_binary, test_size = 0.2, random_state=1)

from sklearn.metrics import accuracy_score


#--------------------------------------------
# --| Using naive bayes classifier |---------
#--------------------------------------------

from sklearn.naive_bayes import GaussianNB

# Fitting the data into the model.
model = GaussianNB()
model.fit(X_train, Y_train)

# Prediction using the test set.
y_pred = model.predict(X_test)

nb_accuracy = accuracy_score(Y_test, y_pred)
print("Naive Bayes classifier accuracy: {}".format(nb_accuracy))

# Plotting.
# plt.scatter(X_train[:,-1], Y_train, color="blue")
# plt.show()

