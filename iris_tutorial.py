
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

# print("X_train shape: {}".format(X_train.shape))
# print("y_train shape: {}".format(y_train.shape))

# print("X_test shape: {}".format(X_test.shape))
# print("y_test shape: {}".format(y_test.shape))

iris_dataframe = pandas.DataFrame(X_train, columns = iris_dataset.feature_names)

grr = pandas.plotting.scatter_matrix(iris_dataframe, c = y_train, figsize = (15, 15), marker = 'o', hist_kwds = {'bins': 20}, s = 60, alpha = 0.8, cmap = mglearn.cm3)

plt.show()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(X_train, y_train)

# KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski', metric_params = None, 
#                      n_jobs = 1, n_neighbors = 1, p = 2, weights = 'uniform')

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
# print(iris_dataset['target_names'])
print("Type: {}".format(iris_dataset['target_names'][prediction]))

#testing accuracy of this prediction

y_pred = knn.predict(X_test)
print("Predictions: {}".format(y_pred))

print("Result {:.2f}".format(np.mean(y_pred == y_test))) # pretty good!






