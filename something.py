
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mglearn
import numpy as np
import pandas
import matplotlib.pyplot as plt



iris_dataset = load_iris()
# print(iris_dataset.keys())

# print("Gatunki: {}".format(iris_dataset['target_names']))

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state = 0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

iris_dataframe = pandas.DataFrame(X_train, columns = iris_dataset.feature_names)

grr = pandas.plotting.scatter_matrix(iris_dataframe, c = y_train, figsize = (15, 15), marker = 'o', hist_kwds = {'bins': 20}, s = 60, alpha = 0.8, cmap = mglearn.cm3)

plt.show()



