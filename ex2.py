
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mglearn
import numpy as np
import pandas
import matplotlib.pyplot as plt



# generating dataset
X, y = mglearn.datasets.make_forge()

#generating graph

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0"], ["Class 1"], loc = 4)
plt.xlabel("First trait")
plt.ylabel("Second trait")
print("X.shape: {}".format(X.shape))

plt.show()

