
from sklearn.datasets import load_iris
import numpy as np
import pandas
import matplotlib.pyplot as plt



iris_dataset = load_iris()
print(iris_dataset.keys())

print("Gatunki: {}".format(iris_dataset['target_names']))



