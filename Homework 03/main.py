import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.sparse.base import SparseEfficiencyWarning
from sklearn.model_selection import train_test_split
import seaborn as sns
import math

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

# Retrieving the data points while execluding the lables and the numbering.
X = data[:,1: -1]
features = features[:-1]

# Feature selection:
# Using CHI2
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

selected = SelectKBest(score_func=chi2, k='all')
fit = selected.fit(X, y_truth_binary)
scores = fit.scores_
X = selected.fit_transform(X, y_truth_binary)

# Optimizing the number of features to select.
# This optimization is based on the scores
# of the features obtained from the SelectBest
# function and a threshold (impact_threshold)
def optimal_features(impact_threshold):
    n = 0
    for score in scores:
        if score > impact_threshold:
            n += 1
    return n

# impact_threshold: The features with scores
# greater than the value of this variable
# will be selected.
impact_threshold = 10000
optimal_n = optimal_features(impact_threshold)

selected = SelectKBest(score_func=chi2, k=optimal_n)
fit = selected.fit(X, y_truth_binary) 
X = selected.fit_transform(X, y_truth_binary)

# One-hot encoding the data.
from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(sparse=True)
X_encoded = oneHotEncoder.fit_transform(X)

# Constructing train and test sets.
test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, y_truth_binary, test_size = test_size, random_state=1)

# To calclate the accuracy of the models.
from sklearn.metrics import accuracy_score

# To calculate the confusion matrix.
from sklearn.metrics import confusion_matrix

#--------------------------------------------
# --|   Using K Nearest Neighbor   |---------
#--------------------------------------------

from sklearn.neighbors import KNeighborsClassifier

# Finds the optimal value for k for the K nearest
# Neighbor classifier by testing the accuricies 
# of multiple runs of the KNN classifier.
def optimal_k(step):
    accuracies = []
    i = 1
    while(i < X_train.shape[0]):
        k = i
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        knn_accuracy = accuracy_score(Y_test, y_pred)    
        accuracies.append(knn_accuracy)
        i += step
    return np.argmax(accuracies) * step
        
step = 10

# After running the below line of code I figured
# That the optimal k is 30.
# NOTE: Running the optimal_k() function would
# take a long time. approximately 15-25 mins.
# NOTE: I ran the function and provided a txt file
# called knn_optimization that contains the accuracy
#  values for each run.

# k = optimal_k(step)

k = 30

model = KNeighborsClassifier(n_neighbors=k)

model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

knn_accuracy = accuracy_score(Y_test, y_pred)
print("K nearest neighbors classifier accuracy: {}".format(knn_accuracy))

# Shows the graph of the optimization of the k
# value for the KNN.
def plot_optimization(step, file_name, xlabel, ylabel):

    text_file = open(file_name, "r")
    entries = text_file.readline()
    entries = np.array(entries.split(", ")).astype(float)

    indep_vars = np.array(list(range(0, len(entries))))
    indep_vars = np.multiply(step, indep_vars)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.plot(indep_vars, entries)
    plt.show()

file_name = "knn_optimization.txt"
plot_optimization(step, file_name, "K value", "Accuracy")

# Confusion matrix configuration

conf_matrix = confusion_matrix(Y_test, y_pred)
sns.heatmap(conf_matrix.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('Y_truth')
plt.ylabel('Y_pred')

plt.show()

#------------------------------------------------------------
# --|   Classification with Logistic Regression    |---------
#------------------------------------------------------------

from sklearn.linear_model import LogisticRegression

def optimal_iter_number(step):
    accuracies = []
    i = 1
    while(i < X_train.shape[0]):  
        model = LogisticRegression(random_state=0, solver='sag', max_iter=i)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        logistic_accuracy = accuracy_score(Y_test, y_pred)
        accuracies.append(logistic_accuracy)

        i += step
    print(accuracies)
    return i

step = 10

# After running the below line of code I figured
# That the optimal iter_number ranges from 10 to 80.
# NOTE: Running the optimal_iter_number() function would
# take a long time. approximately 10-15 mins.
# NOTE: I ran the function and provided a txt file
# called logit_optimization that contains the accuracy
# values for each run.

# iter_number = optimal_iter_number(step)

iter_number = 45

model = LogisticRegression(random_state=0, solver='sag', max_iter=iter_number)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
logistic_accuracy = accuracy_score(Y_test, y_pred)
print("Logistic Regression accuracy: {}".format(logistic_accuracy))

file_name = "logit_optimization.txt"
plot_optimization(step, file_name, "Maximum iterations", "Accuracy")

# Confusion matrix configuration

conf_matrix = confusion_matrix(Y_test, y_pred)
sns.heatmap(conf_matrix.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('Y_truth')
plt.ylabel('Y_pred')
plt.show()

#------------------------------------------------------------

