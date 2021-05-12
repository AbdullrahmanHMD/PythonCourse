import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import math


def show_k_optimization():
    file_name = "knn_optimization.txt"
    text_file = open(file_name, "r")
    entries = text_file.readline()
    entries = np.array(entries.split(", ")).astype(float)

    ind_vars = list(range(0, len(entries)))

    plt.plot(ind_vars, entries)
    plt.show()


show_k_optimization()